"""Independently reconstruct the fresh Study 04A metrics and spot support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

import run as workflow
import score as scoring
from metric_support import ALL_SPOTS, PAIRED_SUPPORT, KEY_COLUMNS, assert_exact_key_matrix

STUDY_ROOT = Path(__file__).resolve().parent
VALIDATOR_K_NEIGHBORS = 30
VALIDATOR_MMD_SAMPLE_PER_BATCH = 1000

# Numerical routines from the independent working Study 04A validator.

def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def alpha_text(value: object) -> str:
    return f'{float(value):.2f}'

def validator_standardize(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale

def validator_neighbors_without_self(values: np.ndarray, k: int) -> np.ndarray:
    check(values.shape[0] > k, f'{values.shape[0]} rows cannot support k={k}')
    raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)
    rows: list[np.ndarray] = []
    for index, neighbors in enumerate(raw):
        without_self = neighbors[neighbors != index]
        check(without_self.size >= k, 'nearest-neighbor result lacks enough non-self rows')
        rows.append(without_self[:k])
    return np.vstack(rows)

def validator_local_batch_entropy(batch_labels: np.ndarray, neighbors: np.ndarray) -> float:
    query_fraction = (batch_labels[neighbors] == 'query').mean(axis=1)
    entropy = np.zeros(query_fraction.size, dtype=np.float64)
    for probability in (query_fraction, 1.0 - query_fraction):
        nonzero = probability > 0
        entropy[nonzero] -= probability[nonzero] * np.log(probability[nonzero])
    return float(np.mean(entropy / np.log(2.0)))

def validator_domain_purity(domain_labels: np.ndarray, valid: np.ndarray, neighbors: np.ndarray) -> float:
    neighbor_labels = domain_labels[neighbors]
    purity = (neighbor_labels == domain_labels[:, None]).mean(axis=1)
    return float(np.mean(purity[valid]))

def validator_paired_retrieval(values: np.ndarray, n_reference: int) -> tuple[float, float]:
    reference = values[:n_reference]
    query = values[n_reference:]
    check(reference.shape == query.shape, 'paired retrieval requires equal support')
    distances = pairwise_distances(query, reference, metric='euclidean')
    paired = np.diag(distances)
    minimum = distances.min(axis=1)
    tolerance = 1e-12 + 1e-09 * np.maximum(1.0, np.abs(minimum))
    top1 = float(np.mean(paired <= minimum + tolerance))
    ranks = 1 + np.sum(distances < paired[:, None] - tolerance[:, None], axis=1)
    return (top1, float(np.median(ranks)))

def validator_mmd2_rbf_biased(values: np.ndarray, n_reference: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    query_count = values.shape[0] - n_reference
    reference_index = rng.choice(n_reference, min(VALIDATOR_MMD_SAMPLE_PER_BATCH, n_reference), replace=False)
    query_index = rng.choice(query_count, min(VALIDATOR_MMD_SAMPLE_PER_BATCH, query_count), replace=False)
    reference = values[reference_index]
    query = values[n_reference + query_index]
    pooled = np.vstack([reference, query])
    distances = pairwise_distances(pooled, metric='euclidean')
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)
    value = rbf_kernel(reference, reference, gamma=gamma).mean() + rbf_kernel(query, query, gamma=gamma).mean() - 2.0 * rbf_kernel(reference, query, gamma=gamma).mean()
    return (float(max(value, 0.0)), bandwidth)

def validator_paired_expression_correlation(matrix: np.ndarray, n_reference: int, paired_spot_ids: np.ndarray) -> float:
    reference = np.asarray(matrix[:n_reference], dtype=np.float64)
    query = np.asarray(matrix[n_reference:], dtype=np.float64)
    check(reference.shape == query.shape, 'expression correlation requires equal support')
    reference -= reference.mean(axis=1, keepdims=True)
    query -= query.mean(axis=1, keepdims=True)
    numerator = np.sum(reference * query, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(query, axis=1)
    zero_variance = denominator <= 0
    if zero_variance.any():
        names = np.asarray(paired_spot_ids, dtype=str)[zero_variance].tolist()
        raise ValueError(f'paired-expression correlation has zero-variance named pairs: {names[:20]}')
    correlations = numerator / denominator
    check(np.isfinite(correlations).all(), 'paired-expression correlation is non-finite')
    return float(np.median(correlations))

def validator_representations(method: str, embedding: np.ndarray, seed: int) -> list[tuple[str, bool, np.ndarray]]:
    if method != 'GraphST':
        representation = 'native_10d_topics' if method == 'STAMP' else 'native_10d_latent'
        return [(representation, True, embedding)]
    return [('pca20_randomized_seed42', True, PCA(n_components=20, svd_solver='randomized', random_state=seed).fit_transform(embedding)), ('pca10_randomized_seed42_sensitivity', False, PCA(n_components=10, svd_solver='randomized', random_state=seed).fit_transform(embedding))]

def validator_metric_row(*, method: str, representation: str, primary: bool, embedding: np.ndarray, batch_labels: np.ndarray, domain_labels: np.ndarray, n_reference: int, expression_correlation: float | None, seed: int) -> dict[str, object]:
    values = validator_standardize(embedding)
    neighbors = validator_neighbors_without_self(values, VALIDATOR_K_NEIGHBORS)
    domain_text = pd.Series(domain_labels).astype('string').fillna('<NA>').to_numpy()
    valid_domain = ~np.isin(np.char.lower(domain_text.astype(str)), ['nan', 'none', 'unknown', '<na>'])
    check(len(np.unique(domain_text[valid_domain])) >= 2, 'fewer than two valid domains remain')
    top1, median_rank = validator_paired_retrieval(values, n_reference)
    mmd2, bandwidth = validator_mmd2_rbf_biased(values, n_reference, seed)
    reference = values[:n_reference]
    query = values[n_reference:]
    return {'method': method, 'representation': representation, 'primary_representation': primary, 'representation_dimensions': int(values.shape[1]), 'batch_asw': float(silhouette_score(values, batch_labels)), 'batch_mixing_entropy_k30': validator_local_batch_entropy(batch_labels, neighbors), 'paired_retrieval_top1': top1, 'paired_retrieval_median_rank': median_rank, 'domain_asw': float(silhouette_score(values[valid_domain], domain_text[valid_domain])), 'domain_purity_k30': validator_domain_purity(domain_text, valid_domain, neighbors), 'centroid_distance': float(np.linalg.norm(reference.mean(axis=0) - query.mean(axis=0))), 'covariance_distance': float(np.linalg.norm(np.cov(reference, rowvar=False) - np.cov(query, rowvar=False), ord='fro') / values.shape[1]), 'mmd2_rbf_biased': mmd2, 'mmd_bandwidth': bandwidth, 'paired_expression_correlation': expression_correlation}

def validator_load_support(reference_path: Path, query_path: Path, genes: list[str], *, mode: str, alpha: float) -> dict[str, object]:
    reference = ad.read_h5ad(reference_path, backed='r')
    query = ad.read_h5ad(query_path, backed='r')
    try:
        check(reference.obs_names.equals(query.obs_names), 'reference/query spot order differs')
        check(not any((gene not in reference.var_names or gene not in query.var_names for gene in genes)), 'fixed-panel support is incomplete')
        reference_matrix = reference[:, genes].X
        query_matrix = query[:, genes].X
        reference_zero = np.asarray(reference_matrix.sum(axis=1)).reshape(-1) == 0
        query_zero = np.asarray(query_matrix.sum(axis=1)).reshape(-1) == 0
        reference_zero_ids = reference.obs_names[reference_zero].astype(str).tolist()
        query_zero_ids = query.obs_names[query_zero].astype(str).tolist()
        check(not reference_zero_ids, f'reference fixed panel has raw-zero rows for {mode}/{alpha:.2f}')
        return {'n_spots': int(reference.n_obs), 'spot_ids': reference.obs_names.astype(str).to_numpy(copy=True), 'remove_pairs': query_zero, 'query_zero_count': int(query_zero.sum()), 'query_zero_ids': query_zero_ids, 'reference_zero_count': 0, 'reference_zero_ids': reference_zero_ids, 'reference_path': str(reference_path.resolve()), 'query_path': str(query_path.resolve())}
    finally:
        reference.file.close()
        query.file.close()


def validate_ladder(run, config):
    manifest = pd.read_csv(run / "simulations/manifest.csv")
    cells = manifest.set_index(["mode", "alpha"], verify_integrity=True)
    expected = {(mode, float(alpha)) for mode in config["simulation"]["modes"]
                for alpha in config["simulation"]["alpha_levels"]}
    check(set(cells.index) == expected and len(cells) == 14, "The 14-cell simulation ladder changed")
    reference, _ = workflow.raw_input_paths(config)
    clean = ad.read_h5ad(reference, backed="r")
    zero = None
    try:
        for mode, alpha in sorted(cells.index):
            path = run / f"simulations/{mode}/alpha_{alpha:.2f}.h5ad"
            data = ad.read_h5ad(path)
            workflow.validate_count_matrix(data.X, str(path))
            check(data.obs_names.equals(clean.obs_names), f"Spot order differs for {path}")
            check(data.var_names.equals(clean.var_names), f"Gene order differs for {path}")
            check(np.array_equal(data.obsm["spatial"], clean.obsm["spatial"]), f"Geometry differs for {path}")
            provenance = data.uns["publication_reproduction"]
            check(provenance["feast_version"] == config["feast_version"]
                  and int(provenance["public_seed"]) == int(config["public_seed"]), f"FEAST provenance differs for {path}")
            if alpha == 0:
                if zero is None:
                    zero = data.X.copy()
                else:
                    check(workflow.sparse_exact(zero, data.X), "The two alpha-zero count matrices differ")
    finally:
        clean.file.close()
    return len(cells)


def reconstruct_candidate(candidate, supports, *, seed):
    n_spots = supports["n_spots"]
    removed = supports["remove_pairs"]
    subsets = [(ALL_SPOTS, np.ones(n_spots, dtype=bool))]
    if removed.any():
        subsets.append((PAIRED_SUPPORT, ~removed))
    for variant, keep in subsets:
        mask = np.tile(keep, 2)
        n_ref = int(keep.sum())
        labels = candidate["domain_labels"][mask]
        batch = candidate["batch_labels"][mask]
        expression = None
        if candidate["normalized"] is not None:
            expression = validator_paired_expression_correlation(
                candidate["normalized"][mask], n_ref, supports["spot_ids"][keep],
            )
        for representation, primary, embedding in validator_representations(candidate["method"], candidate["embedding"][mask], seed):
            row = validator_metric_row(
                method=candidate["method"], representation=representation, primary=primary, embedding=embedding,
                batch_labels=batch, domain_labels=labels, n_reference=n_ref, expression_correlation=expression, seed=seed,
            )
            row.update(
                support_variant=variant, n_reference=n_ref, n_query=n_ref,
                removed_paired_spots=int(n_spots - n_ref),
                removed_named_pairs=json.dumps(supports["query_zero_ids"] if variant == PAIRED_SUPPORT else []),
                fixed_panel_zero_reference_rows=0,
                fixed_panel_zero_reference_spot_ids="[]",
                fixed_panel_zero_query_rows=supports["query_zero_count"],
                fixed_panel_zero_query_spot_ids=json.dumps(supports["query_zero_ids"]),
            )
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "config.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    run = args.run_dir.resolve() if args.run_dir else config["_output_dir"]
    output = args.output_dir or run / "validation"
    simulations = validate_ladder(run, config)
    genes = scoring.load_panel(run / "panel/panel.csv")
    metrics = pd.read_csv(run / "metrics/atomic_metrics.csv")
    support_audit = pd.read_csv(run / "metrics/support_audit.csv").set_index(["mode", "alpha"], verify_integrity=True)
    seed = int(config["public_seed"])
    rows = []
    zero_cases = {}
    for mode in config["simulation"]["modes"]:
        root = run / "simulations" / mode
        for alpha in config["simulation"]["alpha_levels"]:
            alpha = float(alpha)
            if alpha == 0:
                continue
            supports = validator_load_support(root / "alpha_0.00.h5ad", root / f"alpha_{alpha:.2f}.h5ad", genes,
                                              mode=mode, alpha=alpha)
            saved = support_audit.loc[mode, alpha]
            ids = supports["query_zero_ids"]
            check(ids == json.loads(saved.fixed_panel_zero_query_spot_ids)
                  and ids == json.loads(saved.paired_support_removed_named_pairs), "Support-audit barcode mismatch")
            check(len(ids) == saved.paired_support_removed_paired_spots, "Paired exclusion count changed")
            if ids:
                zero_cases[mode, alpha_text(alpha)] = ids
            for method in workflow.METHOD_ARTIFACTS:
                candidate_id = f"{config['configuration_id']}__{method}__{mode}__alpha_{alpha:.2f}"
                path = run / f"methods/{method}/{mode}/alpha_{alpha:.2f}"
                candidate = scoring.load_candidate(path, method, candidate_id, config, supports)
                candidate["method"] = method
                rows.extend(dict(row, mode=mode, alpha=alpha) for row in reconstruct_candidate(candidate, supports, seed=seed))
            print(f"Recomputed {mode}/alpha_{alpha:.2f}", flush=True)
    reconstructed = pd.DataFrame(rows)
    key_set = scoring.expected_keys(config, zero_cases)
    assert_exact_key_matrix(reconstructed, key_set, "independent reconstruction")
    assert_exact_key_matrix(metrics, key_set, "staged metrics")
    expected = reconstructed.set_index(list(KEY_COLUMNS), verify_integrity=True).sort_index()
    observed = metrics.set_index(list(KEY_COLUMNS), verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(
        observed[expected.columns], expected, check_dtype=False, rtol=1e-8, atol=1e-10,
    )
    num = expected.select_dtypes(include="number").columns
    deltas = (observed[num] - expected[num]).abs().max().dropna().to_dict()
    output.mkdir(parents=True, exist_ok=False)
    reconstructed.to_csv(output / "independently_reconstructed_metrics.csv", index=False)
    summary = {
        "status": "passed", "simulations": simulations, "alpha_zero_count_spot_gene_geometry_equal": True,
        "fixed_panel_genes": len(genes), "validated_method_candidates": 36,
        "independently_reconstructed_rows": len(rows),
        "full_support_rows": int(reconstructed.support_variant.eq(ALL_SPOTS).sum()),
        "paired_sensitivity_rows": int(reconstructed.support_variant.eq(PAIRED_SUPPORT).sum()),
        "zero_library_cases": [{"mode": mode, "alpha": float(alpha), "query_barcodes": ids}
                               for (mode, alpha), ids in zero_cases.items()],
        "maximum_absolute_numeric_differences": deltas,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {len(rows)} same-slice metric rows")


if __name__ == "__main__":
    main()
