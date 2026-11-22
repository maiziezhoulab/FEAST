"""Build Figure 4 source tables from the staged simulations and fitted embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/feast-final-numba")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

N_SPOTS = 3611
PUBLIC_SEED = 42
METHODS = ("GraphST", "STAMP", "scVI")
MODE = "diagonal_affine"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
K = 30
UMAP_SETTINGS = {"n_neighbors": 30, "min_dist": 0.3, "metric": "euclidean",
                 "random_state": PUBLIC_SEED, "n_jobs": 1}
DOMAIN_ORDER = [f"Layer_{i}" for i in range(1, 7)] + ["WM"]


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)

def standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    scale = array.std(axis=0)
    scale[scale == 0] = 1.0
    return (array - array.mean(axis=0)) / scale

def uncorrected_pca(counts: np.ndarray) -> tuple[np.ndarray, int]:
    totals = counts.sum(axis=1, keepdims=True)
    nonzero = totals[:, 0] > 0
    scale = np.zeros_like(totals, dtype=np.float32)
    scale[nonzero, 0] = 10000.0 / totals[nonzero, 0]
    log_normalized = np.log1p(counts * scale)
    values = PCA(n_components=20, svd_solver='randomized', random_state=PUBLIC_SEED).fit_transform(log_normalized)
    return (standardize(values), int((~nonzero).sum()))

def umap_projection(values: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(**UMAP_SETTINGS)
    coordinates = reducer.fit_transform(values)
    if coordinates.shape != (2 * N_SPOTS, 2) or not np.isfinite(coordinates).all():
        raise ValueError('UMAP projection is invalid')
    return coordinates

def validate_result_order(result_path: Path, expected_ids: np.ndarray) -> None:
    result = ad.read_h5ad(result_path, backed='r')
    try:
        batches = result.obs['batch'].astype(str).to_numpy(copy=True)
        source_ids = result.obs['source_spot_id'].astype(str).to_numpy(copy=True)
    finally:
        result.file.close()
    expected_batches = np.array(['ref'] * N_SPOTS + ['query'] * N_SPOTS)
    expected_source_ids = np.concatenate([expected_ids, expected_ids])
    if not np.array_equal(batches, expected_batches):
        raise ValueError(f'Batch order changed in {result_path}')
    if not np.array_equal(source_ids, expected_source_ids):
        raise ValueError(f'Paired spot order changed in {result_path}')

def predict_and_verify(values, reference_labels, query_labels):
    """Only embeddings and reference labels enter the prediction step."""
    n_ref = len(reference_labels)
    if values.shape[0] != n_ref + len(query_labels) or not np.isfinite(values).all():
        raise ValueError('Invalid standardized embedding support')
    classifier = KNeighborsClassifier(n_neighbors=K, weights='uniform', metric='minkowski', p=2, n_jobs=1)
    classifier.fit(values[:n_ref], reference_labels)
    predicted = classifier.predict(values[n_ref:])
    indices = classifier.kneighbors(values[n_ref:], return_distance=False)
    classes = np.unique(reference_labels)
    counts = np.column_stack([(reference_labels[indices] == label).sum(axis=1) for label in classes])
    if not np.array_equal(predicted, classes[counts.argmax(axis=1)]):
        raise ValueError('Prediction disagrees with explicit 30-neighbor majority vote')
    score = float(f1_score(query_labels, predicted, labels=DOMAIN_ORDER, average='macro', zero_division=0))
    per_layer = []
    for label in DOMAIN_ORDER:
        tp = int(np.sum((query_labels == label) & (predicted == label)))
        fp = int(np.sum((query_labels != label) & (predicted == label)))
        fn = int(np.sum((query_labels == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        per_layer.append({'layer': label, 'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / denominator if denominator else 0.0})
    reconstructed = float(np.mean([row['f1'] for row in per_layer]))
    if not np.isclose(score, reconstructed, atol=1e-12, rtol=0):
        raise ValueError('Macro-F1 differs from independent per-layer reconstruction')
    return (predicted, score, per_layer)

REFERENCE_DISPLAY_REGION = {'log_mean_reference': (-3.0, -1.5), 'log_variance_reference': (-2.25, -1.35), 'zero_proportion_reference': (0.8, 0.91)}

DISPLAY_LIMITS = {'log_mean': (-4.3, -1.2), 'log_variance': (-3.5, -0.5), 'zero_proportion': (0.72, 1.02)}

def load_cloud(path: Path, alpha: float) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Load stored gene parameters and express them on the retained axes."""
    dataset = ad.read_h5ad(path, backed='r')
    try:
        columns = [f'{coordinate}_{suffix}' for coordinate in ('theta_mu', 'theta_omega', 'theta_pi0') for suffix in ('ref', 'batch')]
        missing = sorted(set(columns) - set(dataset.var.columns))
        if missing:
            raise ValueError(f'Parameter-cloud coordinates are missing from {path}: {missing}')
        deformation = dataset.uns.get('batch_deformation')
        if not isinstance(deformation, dict) or not np.isclose(float(deformation['alpha']), alpha):
            raise ValueError(f'Batch-deformation metadata is invalid in {path}')
        cloud = dataset.var.loc[:, columns].copy()
    finally:
        dataset.file.close()
    cloud.columns = [column.replace('_batch', '').replace('_ref', '_reference') for column in cloud.columns]
    for suffix in ('_reference', ''):
        log_mean = cloud[f'theta_mu{suffix}'].to_numpy(dtype=float)
        cloud[f'log_mean{suffix}'] = log_mean
        cloud[f'log_variance{suffix}'] = log_mean + cloud[f'theta_omega{suffix}'].to_numpy(dtype=float)
        cloud[f'zero_proportion{suffix}'] = 1.0 / (1.0 + np.exp(-cloud[f'theta_pi0{suffix}'].to_numpy(dtype=float)))
    numeric = cloud.loc[:, ['log_mean', 'log_variance', 'zero_proportion']]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite parameter-cloud coordinate in {path}')
    return (cloud, {'D': [float(value) for value in deformation['D']], 'b': [float(value) for value in deformation['b']]})

CONDITION_ORDER = ['raw', 'sim_0.00', 'sim_0.50', 'sim_1.00']

CONDITION_X = {'raw': -0.55, 'sim_0.00': 0.0, 'sim_0.50': 0.5, 'sim_1.00': 1.0}

BATCH_METRICS = [('conditional_batch_asw_macro', 'Layer-conditioned batch ASW', '↓'), ('conditional_batch_entropy_macro', 'Layer-conditioned batch entropy', '↑'), ('conditional_mmd2_macro', 'Layer-conditioned MMD²', '↓')]

PRESERVATION_METRICS = [('layer_asw', 'Layer separation ASW', '↑'), ('layer_purity_k', 'Layer purity (k=30)', '↑'), ('reference_to_query_layer_macro_f1', 'Reference→query layer macro-F1', '↑'), ('spatial_embedding_knn_overlap_k', 'Spatial–embedding kNN overlap', '↑'), ('within_section_knn_preservation_k', 'Within-section kNN preservation', '↑'), ('query_stability_knn_jaccard_k', 'Query kNN stability vs α=0', '↑'), ('query_stability_distance_spearman', 'Query distance stability vs α=0', '↑'), ('query_stability_linear_cka', 'Query linear CKA vs α=0', '↑')]

PLOT_METRICS = BATCH_METRICS + PRESERVATION_METRICS[:2]

def build_plot_data(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[metrics['analysis_role'].isin(['primary', 'uncorrected_baseline']) & metrics['condition'].isin(CONDITION_ORDER)].copy()
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        for metric, label, direction in PLOT_METRICS:
            records.append({'condition': row.condition, 'condition_role': row.condition_role, 'alpha': row.alpha, 'x_position': CONDITION_X[row.condition], 'method': row.method, 'method_seed': row.method_seed, 'representation': row.representation, 'analysis_role': row.analysis_role, 'metric': metric, 'metric_label': label, 'preferred_direction': direction, 'value': float(getattr(row, metric)), 'raw_anchor_connected_to_simulation_ladder': False, 'cross_section_exact_spot_pairing_used': False, 'composite_score_used': False, 'winner_ranking_authorized': False})
    long_data = pd.DataFrame(records).sort_values(['metric', 'method', 'x_position', 'method_seed'], na_position='last')
    seed_summary = long_data.groupby(['condition', 'condition_role', 'alpha', 'x_position', 'method', 'representation', 'analysis_role', 'metric', 'metric_label', 'preferred_direction'], dropna=False, sort=False)['value'].agg(seed_mean='mean', seed_sd='std', seed_min='min', seed_max='max', seed_n='size').reset_index().sort_values(['metric', 'method', 'x_position'])
    return (long_data, seed_summary)


def load_inputs(run):
    paths = {"simulation_manifest": run / "simulations/manifest.csv", "panel": run / "panel/panel.csv"}
    manifest = pd.read_csv(paths["simulation_manifest"]).set_index(["mode", "alpha"], verify_integrity=True)
    genes = pd.read_csv(paths["panel"]).sort_values("panel_order").gene.astype(str).tolist()
    if len(genes) != 2000 or len(set(genes)) != 2000:
        raise ValueError("Expected 2,000 unique ordered fixed-panel genes")
    return manifest, genes


def simulation_path(run, manifest, alpha, mode=MODE):
    saved = Path(manifest.loc[(mode, alpha), "file"])
    expected = Path("simulations") / mode / f"alpha_{alpha:.2f}.h5ad"
    if saved.parts[-len(expected.parts):] != expected.parts:
        raise ValueError(f"Unexpected simulation manifest path for alpha {alpha}: {saved}")
    return run / expected


def read_slice(path, genes):
    data = ad.read_h5ad(path, backed="r")
    try:
        return (
            dense(data[:, genes].X).astype(np.float32, copy=False),
            data.obs_names.astype(str).to_numpy(copy=True),
            data.obs["ground_truth"].astype(str).to_numpy(copy=True),
            np.asarray(data.obsm["spatial"], dtype=np.float64).copy(),
        )
    finally:
        data.file.close()


def uncorrected_pair(reference, query):
    ref_counts, ids, labels, xy = reference
    counts, query_ids, query_labels, query_xy = query
    if len(ids) != N_SPOTS or set(labels) != set(DOMAIN_ORDER):
        raise ValueError("Unexpected same-slice spot or layer support")
    if not (np.array_equal(ids, query_ids) and np.array_equal(labels, query_labels)
            and np.array_equal(xy, query_xy)):
        raise ValueError("Same-slice spot order, layer or coordinate support changed")
    return uncorrected_pca(np.vstack([ref_counts, counts]))


def embedding_table(values, ids, labels, alpha):
    xy = umap_projection(values)
    return pd.DataFrame({
        "mode": MODE, "alpha": alpha, "source_spot_id": np.tile(ids, 2),
        "batch": np.repeat(["ref", "query"], len(ids)), "ground_truth": np.tile(labels, 2),
        "umap_1": xy[:, 0], "umap_2": xy[:, 1],
    })


def write_ladder(run, output, manifest, genes):
    plot_data = []
    clouds = []
    zero_rows = {}
    limits = DISPLAY_LIMITS | {"log_variance": (-3.9, -0.5)}
    columns = list(limits)
    ref_path = simulation_path(run, manifest, 0.0)
    reference = read_slice(ref_path, genes)
    reference_cloud, _ = load_cloud(ref_path, 0.0)
    selected = np.logical_and.reduce([
        reference_cloud[coordinate].between(lower, upper)
        for coordinate, (lower, upper) in REFERENCE_DISPLAY_REGION.items()
    ])
    selected_genes = reference_cloud.index[selected]
    if selected_genes.empty:
        raise ValueError("The registered display region is empty")
    reference_columns = [f"{column}_reference" for column in columns]
    for alpha in ALPHAS:
        path = simulation_path(run, manifest, alpha)
        values, zero_rows[str(alpha)] = uncorrected_pair(reference, read_slice(path, genes))
        print(f"Projecting diagonal_affine alpha={alpha:g}", flush=True)
        plot_data.append(embedding_table(values, reference[1], reference[2], alpha))
        cloud, _ = load_cloud(path, alpha)
        if not cloud[reference_columns].equals(reference_cloud[reference_columns]):
            raise ValueError(f"Reference gene parameters differ at alpha={alpha:g}")
        frame = cloud.loc[selected_genes, columns].rename_axis("gene").reset_index()
        for column, (lower, upper) in limits.items():
            if not frame[column].between(lower, upper).all():
                raise ValueError(f"The registered display limits would clip {column} at alpha={alpha:g}")
        clouds.append(frame.assign(mode=MODE, alpha=alpha))
    pd.concat(plot_data, ignore_index=True).to_csv(output / "uncorrected_embedding_alpha_ladder_plot_data.csv.gz", index=False)
    pd.concat(clouds, ignore_index=True).to_csv(output / "uncorrected_embedding_alpha_ladder_parameter_cloud_plot_data.csv", index=False)
    (output / "uncorrected_embedding_alpha_ladder_provenance.json").write_text(json.dumps({
        "mode": MODE, "alphas": ALPHAS, "ordered_gene_panel": "same_slice/panel/panel.csv",
        "simulation_manifest": "same_slice/simulations/manifest.csv",
        "representation": "Library size 10000, log1p, randomized PCA-20 and column standardization",
        "pca_seed": PUBLIC_SEED, "umap": UMAP_SETTINGS, "zero_library_rows": zero_rows,
        "parameter_reference_region": REFERENCE_DISPLAY_REGION, "parameter_display_limits": limits,
        "selected_gene_count": len(selected_genes), "camera": {"elevation": 12, "azimuth": -45},
    }, indent=2) + "\n")


def trained_embedding(run, primary, method, ids, labels, mode=MODE):
    row = primary.loc[primary.method.eq(method) & primary["mode"].eq(mode) & primary.alpha.eq(1)]
    if len(row) != 1:
        raise ValueError(f"Expected one primary {mode} endpoint for {method}")
    candidate = run / "methods" / method / mode / "alpha_1.00"
    with np.load(candidate / "embeddings.npz", allow_pickle=False) as payload:
        values = np.asarray(payload["embedding"], dtype=np.float64)
    if method == "GraphST":
        values = PCA(n_components=20, svd_solver="randomized", random_state=PUBLIC_SEED).fit_transform(values)
    if values.shape != (2 * N_SPOTS, 20 if method == "GraphST" else 10) or not np.isfinite(values).all():
        raise ValueError(f"Primary embedding dimensions or values differ for {method}")
    result_path = candidate / "result.h5ad"
    validate_result_order(result_path, ids)
    data = ad.read_h5ad(result_path, backed="r")
    try:
        if not np.array_equal(data.obs.dlpfc_layer.astype(str), np.tile(labels, 2)):
            raise ValueError(f"Layer order differs in {result_path}")
    finally:
        data.file.close()
    return standardize(values)


def write_context(run, output, manifest, genes):
    reference = read_slice(simulation_path(run, manifest, 0.0), genes)
    query = read_slice(simulation_path(run, manifest, 1.0), genes)
    uncorrected, zero_rows = uncorrected_pair(reference, query)
    _, ids, labels, xy = reference
    _, _, query_labels, _ = query
    metrics = pd.read_csv(run / "metrics/atomic_metrics.csv")
    primary = metrics.loc[metrics.analysis_role.eq("primary")]
    context, predictions, scores, layers = [], [], [], []
    for method in ("Uncorrected", *METHODS):
        values = uncorrected if method == "Uncorrected" else trained_embedding(run, primary, method, ids, labels)
        plot_name = "Uncorrected input" if method == "Uncorrected" else method
        context.append(embedding_table(values, ids, labels, 1.0).assign(representation=plot_name))
        predicted, score, per_layer = predict_and_verify(values, labels, query_labels)
        predictions.append(pd.DataFrame({
            "mode": MODE, "alpha": 1.0, "method": method, "source_spot_id": ids,
            "x": xy[:, 0], "y": xy[:, 1], "ground_truth": query_labels, "predicted_layer": predicted,
            "correct": query_labels == predicted,
        }))
        scores.append({"mode": MODE, "alpha": 1.0, "method": method, "n_reference": len(ids),
                       "n_query": len(ids), "k": K, "macro_f1": score})
        layers.extend({"mode": MODE, "alpha": 1.0, "method": method, **row} for row in per_layer)
        print(f"{method}: verified macro-F1 {score:.6f}", flush=True)
    pd.concat(context, ignore_index=True).to_csv(output / "representative_embedding_context_data.csv", index=False)
    for name, frame in (
        ("predictions", pd.concat(predictions, ignore_index=True)),
        ("scores", pd.DataFrame(scores)),
        ("per_layer_scores", pd.DataFrame(layers)),
    ):
        frame.to_csv(output / f"same_slice_layer_transfer_spatial_{name}.csv", index=False)
    (output / "same_slice_layer_transfer_spatial_provenance.json").write_text(json.dumps({
        "mode": MODE, "alpha": 1.0, "seed": PUBLIC_SEED, "umap": UMAP_SETTINGS,
        "classifier": "Uniform vote among the 30 Euclidean-nearest reference spots",
        "standardized_dimensions": {"Uncorrected": 20, "GraphST": 20, "STAMP": 10, "scVI": 10},
        "uncorrected_zero_library_rows": zero_rows, "labels_for_training": "reference cortical labels",
        "labels_for_evaluation": "query cortical labels", "spots_per_batch": len(ids),
    }, indent=2) + "\n")


def write_design_table(same_slice, real_slices, output):
    same = json.loads((same_slice / "simulations/provenance.json").read_text())
    inputs = {row["role"]: row["path"] for row in same["inputs"]}
    section = Path(inputs["clean_reference"]).stem
    conditions = pd.read_csv(real_slices / "inputs/condition_manifest.csv")
    if set(conditions.condition) != {"raw", "sim_0.00", "sim_0.50", "sim_1.00", "sim_1.50"}:
        raise ValueError("Unexpected 04B conditions for the design table")
    prepare = json.loads((real_slices / "prepare_provenance.json").read_text())
    reference, query = (Path(row["path"]).stem for row in prepare["inputs"][:2])
    table = pd.DataFrame([
        ("Reference", f"{section} · simulated at α=0", f"{reference} · raw section"),
        ("Query", f"{section} · FEAST generated", f"{query} · raw or FEAST generated"),
        ("Conditions shown", "α = 0.25, 0.5, 0.75, 1.0", "Raw anchor; α = 0, 0.5, 1.0"),
        ("Spot pairing", "Known one-to-one pairs", "No cross-section pairs"),
    ], columns=["Setting", "Same-slice", "Two-slice"])
    table.to_csv(output / "design_table.csv", index=False)


def write_real_slice_data(run, output):
    validation = json.loads((run / "validation/summary.json").read_text())
    qualification = json.loads((run / "inputs/qualification.json").read_text())
    if (validation["stage"] != "complete"
            or any(validation[key]["status"] != "passed" for key in ("prepared", "candidates", "scores"))
            or validation["scores"]["independently_reconstructed_rows"] != 65
            or qualification["status"] != "passed"
            or qualification["cross_section_spot_pairing_declared"]):
        raise ValueError("The 04B inputs and 65 atomic rows must pass validation before plotting")
    metrics = pd.read_csv(run / "metrics/atomic_metrics.csv")
    roles = metrics.analysis_role.value_counts().to_dict()
    if roles != {"primary": 45, "representation_sensitivity": 15, "uncorrected_baseline": 5}:
        raise ValueError(f"Unexpected two-slice atomic metrics support: {roles}")
    rows, summary = build_plot_data(metrics)
    registered = pd.read_csv(run / "metrics/seed_summary_long.csv")
    keys = ["condition", "method", "representation", "metric"]
    fields = ["seed_mean", "seed_min", "seed_max", "seed_n"]
    trained = summary.loc[summary.method.ne("Uncorrected")]
    matched = trained.merge(registered[keys + fields], on=keys, suffixes=("", "_registered"), validate="one_to_one")
    if len(matched) != len(trained):
        raise ValueError("Two-slice plotted means and ranges do not cover the registered score summary")
    for field in fields:
        if not np.allclose(matched[field], matched[f"{field}_registered"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Two-slice seed summary differs for {field}")
    rows.to_csv(output / "b_method_seed_rows.csv", index=False)
    summary.to_csv(output / "b_seed_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-slice", type=Path, required=True)
    parser.add_argument("--real-slices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--section", choices=["all", "ladder", "context", "design", "real_slices"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    same = args.output_dir / "same_slice"
    real = args.output_dir / "real_slices"
    if args.section in {"all", "ladder", "context"}:
        same.mkdir(exist_ok=True)
        manifest, genes = load_inputs(args.same_slice)
        if args.section in {"all", "ladder"}:
            write_ladder(args.same_slice, same, manifest, genes)
        if args.section in {"all", "context"}:
            write_context(args.same_slice, same, manifest, genes)
    if args.section in {"all", "design"}:
        write_design_table(args.same_slice, args.real_slices, args.output_dir)
    if args.section in {"all", "real_slices"}:
        real.mkdir(exist_ok=True)
        write_real_slice_data(args.real_slices, real)


if __name__ == "__main__":
    main()
