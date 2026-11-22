"""Verify and organize the 45 existing scored jobs for Supplementary Figure S4."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = "05_2d_conditional_transfer"
IDENTITY = ["job_id", "dataset", "mode", "source", "target", "direction", "assignment_randomness"]
KEYS = ["job_id", "gene"]
SCORE_FIELDS = ["median_gene_pearson", "moran_corr"]
DIRECTIONS = {
    "DLPFC · within donor": ("151675_to_151676", "151676_to_151675"),
    "DLPFC · cross donor": ("151670_to_151675", "151675_to_151670", "151670_to_151676", "151676_to_151670"),
    "DLPFC · half-slice": tuple(f"{sid}_low_x_to_high_x" for sid in ("151670","151675","151676")),
    "MERFISH · cross-slice": ("Zhuang-ABCA-1.006_to_Zhuang-ABCA-1.007", "Zhuang-ABCA-1.007_to_Zhuang-ABCA-1.006"),
    "MERFISH · half-slice": tuple(f"Zhuang-ABCA-1.{sid}_low_x_to_high_x" for sid in ("006","007")),
}


def verify_scores(summary, per_gene):
    computed = []
    for job, group in per_gene.groupby("job_id"):
        pair = group[["target_moran_i","generated_moran_i"]].dropna()
        value = np.corrcoef(pair.target_moran_i, pair.generated_moran_i)[0,1]
        computed.append({"job_id":job, "median_gene_pearson":group.pearson.median(),
                         "moran_corr":value, "n_finite_pearson":int(np.isfinite(group.pearson).sum()),
                         "n_gene_rows":len(group)})
    computed = pd.DataFrame(computed).set_index("job_id", verify_integrity=True)
    stored = summary.set_index("job_id", verify_integrity=True).loc[computed.index]
    np.testing.assert_allclose(computed[SCORE_FIELDS], stored[SCORE_FIELDS], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(computed.n_gene_rows, stored.n_genes)
    return computed[["n_finite_pearson","n_gene_rows"]]


def prepare(zenodo):
    results = zenodo / "results" / STUDY
    comparison = zenodo / "inputs" / STUDY / "article_comparison"
    output = results / "figure_s4"
    output.mkdir(parents=True, exist_ok=True)
    data_types = {"source":str,"target":str,"gene":str}
    score_root = results / "scores"
    report = json.loads((results / "study05_complete_validation.json").read_text())
    assert report["validated_jobs"] == 45 and report["scores"]["per_gene_rows"] == 587367
    tasks = pd.read_csv(score_root / "summary.csv", dtype=data_types)
    support = pd.read_csv(score_root / "support_audit.csv", dtype=data_types)
    per_gene = pd.read_csv(score_root / "per_gene_metrics.csv.gz", dtype=data_types)
    assert len(tasks) == len(support) == 45 and len(per_gene) == 587367
    assert not per_gene.duplicated(KEYS).any()
    pd.testing.assert_frame_equal(tasks[IDENTITY].sort_values("job_id").reset_index(drop=True),
                                   per_gene[IDENTITY].drop_duplicates().sort_values("job_id").reset_index(drop=True))
    counts = verify_scores(tasks, per_gene)
    tasks = tasks.merge(support[IDENTITY + ["target_candidate_spots","target_retained_spots",
                                          "excluded_target_label_counts_json"]], on=IDENTITY, validate="one_to_one")
    tasks = tasks.join(counts, on="job_id")
    np.testing.assert_array_equal(tasks.n_target_spots, tasks.target_retained_spots)
    np.testing.assert_allclose(tasks.target_retained_fraction,
                               tasks.target_retained_spots/tasks.target_candidate_spots, rtol=0, atol=1e-15)
    historical = pd.read_csv(comparison / "scores/summary.csv", dtype=data_types)
    compared = tasks.merge(historical[IDENTITY + SCORE_FIELDS], on=IDENTITY, suffixes=("_current","_article"), validate="one_to_one")
    for field in SCORE_FIELDS:
        np.testing.assert_array_equal(compared[f"{field}_current"], compared[f"{field}_article"])
    compared[IDENTITY + [f"{f}_{suffix}" for f in SCORE_FIELDS for suffix in ("current","article")]].to_csv(
        output / "score_comparison.csv", index=False)

    primary = tasks.loc[tasks.primary_setting].copy()
    task_order = pd.DataFrame([{"direction":key, "task_group":group}
                              for group, directions in DIRECTIONS.items() for key in directions])
    primary = task_order.merge(primary, on="direction", validate="one_to_one")
    assert len(primary) == 13
    exclusions = []
    for row in tasks.itertuples(index=False):
        excluded = json.loads(row.excluded_target_label_counts_json)
        assert sum(excluded.values()) == row.target_candidate_spots-row.target_retained_spots
        exclusions.extend({"job_id":row.job_id,"direction":row.direction,"dataset":row.dataset,"mode":row.mode,
                           "assignment_randomness":row.assignment_randomness,"label":label,"excluded_spots":n}
                          for label,n in excluded.items())
    primary_genes = per_gene.loc[per_gene.primary_setting].copy()
    assert len(primary_genes) == 161007
    primary_genes["pearson_defined"] = np.isfinite(primary_genes.pearson)
    old_genes = pd.read_csv(comparison / "scores/per_gene_metrics.csv.gz", dtype=data_types)
    old_finite = int(np.isfinite(old_genes.loc[old_genes.primary_setting, "pearson"]).sum())
    finite = int(primary_genes.pearson_defined.sum())
    direction_rows = tasks.loc[tasks["mode"].eq("cross_slice")].sort_values(["dataset","direction","assignment_randomness"])
    assert len(direction_rows) == 40 and direction_rows.groupby("direction").size().eq(5).all()
    primary.to_csv(output / "task_summary.csv", index=False)
    primary_genes.to_csv(output / "gene_data.csv.gz", index=False)
    direction_rows.to_csv(output / "ar_data.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output / "support_by_class.csv", index=False)
    fields = ["direction","target_retained_spots","target_candidate_spots","n_finite_pearson"]
    (output / "provenance.json").write_text(json.dumps({
        "study05_scores": str(score_root.relative_to(zenodo)), "seed":2026, "primary_assignment_randomness":.3,
        "jobs":45, "primary_tasks":13, "primary_gene_rows":int(len(primary_genes)), "pearson_defined":finite,
        "pearson_undefined":int(len(primary_genes))-finite, "archived_pearson_defined":old_finite,
        "fixed_seed_sensitivity_rows":40, "metric_pairs_match_archive":True,
        "task_support":primary[fields].to_dict("records"),
    },indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path(__file__).resolve().parents[2] / "Zenodo")
    prepare(parser.parse_args().zenodo)
