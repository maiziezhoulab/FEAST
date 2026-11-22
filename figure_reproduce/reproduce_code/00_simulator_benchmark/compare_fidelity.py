"""Compare Figure 2 inputs and results with the corrected article release."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIGURE_SCRIPTS = Path(__file__).resolve().parents[2] / "reproduce_figures/Scripts/Figure_02"
sys.path.insert(0, str(FIGURE_SCRIPTS))
from plot import METRIC_SPECS, build_long_table, build_median_table, load_metrics

SCORE_KEYS = ["simulator", "sample"]
SPOT_KEYS = ["sample", "gene", "spot_id"]
SOURCE_KEYS = ["sample", "gene", "source"]
SUPPORT_COLUMNS = ["identity_flag", "n_common_genes", "n_paired_spots", "spot_pairing", "n_moran_genes"]
METRICS = [spec["column"] for spec in METRIC_SPECS] + ["gene_mean_wasserstein", "gene_variance_wasserstein"]
STEM = "simulator_spatial_comparison"


def read(path, keys):
    return pd.read_csv(path, float_precision="round_trip", dtype={"spot_id": str}).set_index(keys, verify_integrity=True)


def compare(before, after, columns):
    if set(before.index) != set(after.index):
        raise ValueError("Historical and new tables have different key sets")
    old = before.loc[after.index, columns]
    new = after[columns]
    joined = old.add_suffix("_old").join(new.add_suffix("_new"))
    for column in columns:
        if pd.api.types.is_numeric_dtype(new[column]) and not pd.api.types.is_bool_dtype(new[column]):
            joined[f"{column}_delta"] = new[column] - old[column]
    return joined


def map_residual_summary(residuals):
    records = []
    for (sample, gene), section in residuals.groupby(level=["sample", "gene"], sort=False):
        for source in ("FEAST", "SRTsim", "scCube"):
            abs_residual = section[f"{source}_log1p_residual"].abs().to_numpy()
            records.append({
                "sample": sample,
                "gene": gene,
                "source": source,
                "spot_count": len(section),
                "mean_abs_residual": np.mean(abs_residual),
                "q99_abs_residual": np.percentile(abs_residual, 99),
                "zero_residual_fraction": np.mean(abs_residual == 0),
            })
    return pd.DataFrame(records).set_index(SOURCE_KEYS)


def write_comparison(results, output):
    article = results / "article"
    old_path = article / "metrics/simulator_quality_metrics.csv"
    new_path = results / "metrics/simulator_quality_metrics.csv"
    old = read(old_path, SCORE_KEYS)
    new = read(new_path, SCORE_KEYS)
    retained = new.index.get_level_values("simulator") != "FEAST_reference_rank"
    if len(new) != 60 or retained.sum() != 48:
        raise ValueError("Expected 12 fresh FEAST and 48 retained comparator results")
    pd.testing.assert_frame_equal(old.loc[new.index[retained], new.columns], new.loc[retained], check_exact=True)
    scores = compare(old, new, METRICS + SUPPORT_COLUMNS)
    scores.insert(0, "result_source", np.where(retained, "retained article result", "fresh FEAST run"))

    median_keys = ["source_column", "method"]
    previous_medians = build_median_table(build_long_table(load_metrics(old_path))).set_index(median_keys)
    archived_medians = read(article / "figure2/metric_medians.csv", median_keys).loc[previous_medians.index]
    np.testing.assert_array_equal(previous_medians["panel_rank"], archived_medians["panel_rank"])
    np.testing.assert_allclose(previous_medians["median_score"], archived_medians["median_score"], rtol=1e-13, atol=1e-16)
    medians = read(results / "figure2/metric_medians.csv", median_keys)
    computed_medians = build_median_table(build_long_table(load_metrics(new_path))).set_index(median_keys)
    pd.testing.assert_frame_equal(computed_medians.loc[medians.index], medians, check_exact=True)
    median_change = compare(previous_medians, medians, ["median_score", "panel_rank", "n_available"])

    old_maps, new_maps = article / "figure2", results / "figure2"
    limits = compare(
        read(old_maps / f"{STEM}_color_limits.csv", ["sample", "gene"]),
        read(new_maps / f"{STEM}_color_limits.csv", ["sample", "gene"]),
        ["expression_vmax", "residual_vmin", "residual_vmax"],
    )
    match_columns = ["spot_pairing", "paired_spots", "spots_reordered", "matched_coordinate_max_abs_difference"]
    matching = compare(
        read(old_maps / f"{STEM}_spot_matching.csv", SOURCE_KEYS),
        read(new_maps / f"{STEM}_spot_matching.csv", SOURCE_KEYS),
        match_columns,
    )
    previous_residuals = read(old_maps / f"{STEM}_residuals.csv.gz", SPOT_KEYS)
    fresh_residuals = read(new_maps / f"{STEM}_residuals.csv.gz", SPOT_KEYS)
    if len(fresh_residuals) != 500_956:
        raise ValueError("Expected full spot support for the six Figure 2b sections")
    reference_and_retained = ["x", "y", "reference_log1p", "SRTsim_log1p_residual", "scCube_log1p_residual"]
    pd.testing.assert_frame_equal(
        previous_residuals.loc[fresh_residuals.index, reference_and_retained],
        fresh_residuals[reference_and_retained], check_exact=True,
    )
    spatial = compare(
        map_residual_summary(previous_residuals), map_residual_summary(fresh_residuals),
        ["spot_count", "mean_abs_residual", "q99_abs_residual", "zero_residual_fraction"],
    )

    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "old_vs_new_metrics.csv")
    median_change.to_csv(output / "old_vs_new_medians.csv")
    limits.to_csv(output / "old_vs_new_color_limits.csv")
    matching.to_csv(output / "old_vs_new_spot_matching.csv")
    spatial.to_csv(output / "old_vs_new_spatial_residual_summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_comparison(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
