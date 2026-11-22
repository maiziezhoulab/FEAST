"""Build Figure 5a/b source tables and compare all registered scores with the article run."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from conditional_resampling_baseline import conditional_donor_indices, selected_jobs
from evaluate_resampling_metrics import HEADLINE_METRICS
from workflow import STUDY_DIR, direction_id, expected_job_path, inspect_inputs

CASES = (
    ("A", "dlpfc", "cross_slice", "151676", "151675", "GFAP", "within_donor"),
    ("B", "dlpfc", "mask_half", "151676", "151676", "MBP", "within_slice"),
    ("C", "merfish", "cross_slice", "Zhuang-ABCA-1.006", "Zhuang-ABCA-1.007", "Frzb", "donor_not_declared"),
    ("D", "merfish", "mask_half", "Zhuang-ABCA-1.007", "Zhuang-ABCA-1.007", "Igfbp4", "within_slice"),
)
ID = ["dataset", "mode", "direction"]


def marker(data, gene):
    return np.asarray(data[:, [gene]].layers["counts"].sum(axis=1)).ravel()


def frames(config, paths, panels, results, output, metrics, genes):
    inputs = {key: ad.read_h5ad(path) for key, path in paths.items()}
    rng = np.random.default_rng(config["public_seed"])
    by_task = {(dataset, mode, source, target): (col, gene)
               for col, dataset, mode, source, target, gene, _ in CASES}
    draw_zero = metrics.loc[metrics.replicate.eq(0)].set_index(ID, verify_integrity=True)
    data_rows, selections, donors = [], [], []
    for job in selected_jobs(config):
        name = direction_id(job)
        source, target = inputs[job.dataset, job.source], inputs[job.dataset, job.target]
        label = config["datasets"][job.dataset]["annotation_key"]
        path = expected_job_path(results / "final", job)
        record = json.loads((path / "provenance.json").read_text())
        assert not record["target_expression_loaded_by_generation"]
        support = record["support"]
        source_xy, target_xy = source.obsm["spatial"], target.obsm["spatial"]
        source_label, target_label = source.obs[label].astype(str), target.obs[label].astype(str)
        in_source = source_label.isin(support["eligible_labels"]).to_numpy()
        in_target = target_label.isin(support["eligible_labels"]).to_numpy()
        split = np.nan
        if job.mode == "mask_half":
            split = float(np.median(source_xy[:, 0]))
            in_source &= source_xy[:, 0] <= split
            in_target &= target_xy[:, 0] > split
        assert in_source.sum() == support["source_retained_spots"]
        assert in_target.sum() == support["target_retained_spots"]
        selected = np.flatnonzero(in_source)
        donor = selected[conditional_donor_indices(source_label.iloc[selected].to_numpy(),
                                                   target_label.loc[in_target].to_numpy(), rng)]
        source_panel = panels[job.dataset]
        source_library = np.asarray(source[:, source_panel].layers["counts"].sum(axis=1)).ravel()
        target_library = np.asarray(target[in_target, source_panel].layers["counts"].sum(axis=1)).ravel()
        expected_w1 = draw_zero.loc[(job.dataset, job.mode, name), "library_size_wasserstein"]
        replay_w1 = wasserstein_distance(target_library, source_library[donor])
        np.testing.assert_allclose(replay_w1, expected_w1, rtol=1e-10, atol=1e-10)
        donors.append(pd.DataFrame({"dataset": job.dataset, "mode": job.mode, "direction": name,
                                    "target_obs_id": target.obs_names[in_target], "donor_obs_id": source.obs_names[donor],
                                    "donor_library_size": source_library[donor], "target_library_size": target_library}))
        key = (job.dataset, job.mode, job.source, job.target)
        if key not in by_task:
            continue
        col, gene = by_task[key]
        generated = ad.read_h5ad(path / "generated.h5ad")
        assert generated.obs_names.equals(target.obs_names[in_target])
        np.testing.assert_array_equal(generated.obsm["spatial"], target_xy[in_target])
        np.testing.assert_array_equal(generated.obs[label].astype(str), target_label.loc[in_target])
        assert generated.var_names.tolist() == source_panel
        source_counts, target_counts = marker(source, gene), marker(target, gene)
        for method, ids, xy, mask, values in [
            ("source", source.obs_names, source_xy, in_source, source_counts[in_source]),
            ("truth", target.obs_names, target_xy, in_target, target_counts[in_target]),
            ("feast", target.obs_names, target_xy, in_target, marker(generated, gene)),
            ("resampling", target.obs_names, target_xy, in_target, source_counts[donor]),
        ]:
            points = pd.DataFrame({"column": col, "gene": gene, "method": method, "spot_id": ids,
                                   "x": xy[:, 0], "y": xy[:, 1], "retained": mask})
            points.loc[mask, "count"] = values
            data_rows.append(points)
        score = genes.loc[genes.direction.eq(name) & genes.gene.eq(gene)]
        assert len(score) == 1
        selections.append({"column": col, "gene": gene, "dataset": job.dataset, "mode": job.mode,
                           "direction": name, "source": job.source, "target": job.target,
                           "pearson": score.pearson.iloc[0], "source_support": int(in_source.sum()),
                           "target_support": int(in_target.sum()), "split": split, "draw_0_library_w1": replay_w1})
    points = pd.concat(data_rows, ignore_index=True)
    assert len(selections) == len(CASES)
    points.to_csv(output / "map_points.csv.gz", index=False)
    pd.concat(donors, ignore_index=True).to_csv(output / "draw_0_donors.csv.gz", index=False)
    pd.DataFrame(selections).sort_values("column").to_csv(output / "map_selection.csv", index=False)
    eligible = []
    for col, dataset, mode, source, _, gene, stratum in CASES:
        candidate_genes = [gene] if col != "B" else [gene, "SCGB2A2", "TUBA1B", "MT-CO2"]
        frame = genes.loc[genes.dataset.eq(dataset) & genes["mode"].eq(mode) &
                          genes.gene.isin(candidate_genes) & genes.donor_stratum.eq(stratum)]
        if col == "B":
            frame = frame.loc[frame.source.eq(source)]
        eligible.append(frame.assign(column=col))
    pd.concat(eligible).to_csv(output / "map_candidates.csv", index=False)
    return float(np.log1p(points["count"].max()))


def prepare(zenodo, result_root=None):
    study = STUDY_DIR.name
    inputs = zenodo / "inputs" / study
    results = result_root or zenodo / "results" / study
    output = results / "figure05_sources"
    output.mkdir(parents=True, exist_ok=True)
    config, paths, panels = inspect_inputs(STUDY_DIR / "config.yaml", inputs / "input_files.csv", inputs)
    baseline = Path("baselines/conditional_metric_comparison")
    summarize = Path("scores/summary.csv")
    old = inputs / "article_comparison"
    dtype = {"source": str, "target": str}
    data = pd.read_csv(results / baseline / "method_metrics.csv", dtype=dtype)
    pairs = pd.read_csv(results / baseline / "feast_comparison.csv", dtype=dtype)
    assert len(data) == 143 and not data.duplicated(ID + ["method", "replicate"]).any()
    assert len(pairs) == 52 and not pairs.duplicated(ID + ["metric"]).any()
    assert set(pairs.metric) == set(HEADLINE_METRICS)
    derived = data.loc[data.method.ne("feast")].groupby(ID)[list(HEADLINE_METRICS)].mean().stack()
    claimed = pairs.set_index(ID + ["metric"]).baseline_mean
    np.testing.assert_allclose(derived.loc[claimed.index], claimed, rtol=1e-10, atol=1e-12)
    pairs.to_csv(output / "panel_b_pairs.csv", index=False)
    scores = pd.read_csv(results / "scores/per_gene_metrics.csv.gz", dtype=dtype)
    primary = scores.loc[scores.primary_setting]
    vmax = frames(config, paths, panels, results, output, data, primary)
    expected = pd.read_csv(old / "scores/per_gene_metrics.csv.gz", dtype=dtype)
    assert len(scores) == len(expected) == 587367 and len(primary) == 161007
    assert not scores.duplicated(["job_id", "gene"]).any()
    historical_primary = expected.loc[expected.primary_setting]
    finite = int(np.isfinite(primary.pearson).sum())
    historical_finite = int(np.isfinite(historical_primary.pearson).sum())
    score_keys = ID + ["assignment_randomness", "source", "target"]
    new = pd.read_csv(results / summarize, dtype=dtype)
    hist = pd.read_csv(old / summarize, dtype=dtype)
    comparison = new.merge(hist, on=score_keys, suffixes=("_current", "_article"), validate="one_to_one")
    assert len(comparison) == len(new) == len(hist) == 45
    for field in ("n_target_spots", "n_genes", "target_retained_fraction", "n_reference_observed_genes"):
        np.testing.assert_array_equal(comparison[f"{field}_current"], comparison[f"{field}_article"])
    comparison.to_csv(output / "score_comparison.csv", index=False)
    metric_comparison = pairs.merge(pd.read_csv(old / baseline / "feast_comparison.csv", dtype=dtype),
                                   on=ID + ["metric"], suffixes=("_current", "_article"), validate="one_to_one")
    assert len(metric_comparison) == 52
    metric_comparison.to_csv(output / "paired_metric_comparison.csv", index=False)
    report = {"predictions": len(new), "primary_pairs": len(primary.job_id.unique()), "method_evaluations": len(data),
              "primary_gene_rows": len(primary), "current_finite_pearson": finite, "article_finite_pearson": historical_finite,
              "draw": 0, "draw_seed": config["public_seed"], "draw_stream_tasks": 13, "unclipped_log1p_max": vmax}
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    prepare(args.zenodo, args.results)
