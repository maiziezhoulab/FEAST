"""Compare the Figure 4 atomic metrics, support and manuscript endpoints."""

import argparse
import json
from pathlib import Path

import pandas as pd

SAME_KEYS = ["method", "mode", "alpha", "representation", "support_variant"]
REAL_KEYS = ["method", "condition", "method_seed", "representation", "analysis_role"]
METRICS_A = (
    ("batch_asw", False), ("mmd2_rbf_biased", False),
    ("batch_mixing_entropy_k30", True), ("paired_retrieval_top1", True),
    ("domain_asw", True), ("domain_purity_k30", True),
)
METRICS_B = (
    ("conditional_batch_asw_macro", False), ("conditional_mmd2_macro", False),
    ("conditional_batch_entropy_macro", True), ("layer_asw", True), ("layer_purity_k", True),
)


def compare_tables(fresh, prior, keys):
    shared_numeric = [key for key in fresh.select_dtypes(include="number") if key in prior and key not in keys]
    evidence = fresh[keys + shared_numeric].merge(
        prior[keys + shared_numeric], on=keys, how="outer", indicator="key_status",
        suffixes=("_fresh", "_prior"), validate="one_to_one",
    )
    deltas = {f"{key}_difference": evidence[f"{key}_fresh"] - evidence[f"{key}_prior"]
              for key in shared_numeric}
    return evidence.assign(**deltas)


def rank_endpoints(history, metrics, group, destination):
    keys = list(group)
    data = history[keys + ["method"] + [
        f"{metric}_{version}" for metric, _ in metrics for version in ("prior", "fresh")
    ]]
    records = []
    for values, subset in data.groupby(keys):
        for metric, higher in metrics:
            cols = ["method", f"{metric}_prior", f"{metric}_fresh"]
            rows = subset[cols].rename(columns={cols[1]: "prior", cols[2]: "fresh"}).copy()
            for version in ("prior", "fresh"):
                rows[f"rank_{version}"] = rows[version].rank(ascending=not higher, method="min")
            rows["rank_changed"] = rows.rank_fresh.ne(rows.rank_prior)
            values = values if isinstance(values, tuple) else (values,)
            rows = rows.assign(**dict(zip(keys, values)), metric=metric)
            records.append(rows)
    pd.concat(records, ignore_index=True).to_csv(destination, index=False)


def compare_same_slice(args, out):
    old = pd.read_csv(args.previous_same / "atomic_metrics.csv")
    new = pd.read_csv(args.run / "same_slice/metrics/atomic_metrics.csv")
    history = compare_tables(new, old, SAME_KEYS)
    history.to_csv(out / "same_slice_atomic_old_vs_new.csv", index=False)
    primary = history.loc[history.support_variant.eq("all_spots_primary")
                          & history.representation.ne("pca10_randomized_seed42_sensitivity")]
    if len(primary) != 36 or not primary.key_status.eq("both").all():
        raise ValueError("The 36 primary same-slice cells must match the historical design")
    endpoints = primary.loc[primary.alpha.isin([0.25, 1.0])]
    rank_endpoints(endpoints, METRICS_A, ["mode", "alpha"], out / "same_slice_metric_order.csv")
    retrieval = endpoints.pivot(
        index=["mode", "method"], columns="alpha", values=["paired_retrieval_top1_prior", "paired_retrieval_top1_fresh"],
    )
    retrieval_result = pd.DataFrame(index=retrieval.index)
    for version in ("prior", "fresh"):
        key = f"paired_retrieval_top1_{version}"
        retrieval_result[f"{version}_alpha_0_25"] = retrieval[key, 0.25]
        retrieval_result[f"{version}_alpha_1_00"] = retrieval[key, 1.0]
        retrieval_result[f"{version}_declined"] = retrieval[key, 1.0] < retrieval[key, 0.25]
    retrieval_result.to_csv(out / "same_slice_paired_retrieval.csv")

    old_layer = pd.read_csv(args.previous_transfer)
    new_layer = pd.read_csv(args.run / "plot_data/same_slice/same_slice_layer_transfer_spatial_scores.csv")
    transfer = compare_tables(new_layer, old_layer.query("mode == 'diagonal_affine' & alpha == 1"),
                              ["method", "mode", "alpha"])
    if len(transfer) != 4 or not transfer.key_status.eq("both").all():
        raise ValueError("The same-slice layer-transfer comparison must cover four display representations")
    transfer.to_csv(out / "same_slice_endpoint_layer_transfer.csv", index=False)

    common = new.loc[new.support_variant.eq("all_spots_primary") & new.method.eq("GraphST")]
    keys = ["mode", "alpha"]
    primary_rows = common.loc[common.primary_representation].set_index(keys)
    sensitivity = common.loc[~common.primary_representation].set_index(keys)
    deltas = pd.DataFrame(index=primary_rows.index)
    for key in ("domain_purity_k30", "batch_mixing_entropy_k30", "paired_retrieval_top1"):
        deltas[f"delta_pca10_minus_pca20_{key}"] = sensitivity[key] - primary_rows[key]
    deltas.to_csv(out / "same_slice_pca_sensitivity_for_article.csv")

    old_support = pd.read_csv(args.previous_same / "support_audit.csv")
    new_support = pd.read_csv(args.run / "same_slice/metrics/support_audit.csv")
    columns = ["mode", "alpha", "fixed_panel_zero_query_spot_ids", "fixed_panel_zero_query_rows"]
    old_support[columns].merge(new_support[columns], on=["mode", "alpha"], validate="one_to_one",
                               suffixes=("_prior", "_fresh")).to_csv(out / "same_slice_zero_support.csv", index=False)
    (out / "same_slice_row_counts.json").write_text(json.dumps({
        version: table.support_variant.value_counts().to_dict()
        for version, table in (("prior", old), ("fresh", new))
    }, indent=2) + "\n")


def compare_two_slice(args, out):
    old = pd.read_csv(args.previous_real / "atomic_metrics.csv")
    new = pd.read_csv(args.run / "real_slices/metrics/atomic_metrics.csv")
    history = compare_tables(new, old, REAL_KEYS)
    if len(history) != 65 or not history.key_status.eq("both").all():
        raise ValueError("The two-slice comparison must match all 65 atomic keys")
    history.to_csv(out / "two_slice_atomic_old_vs_new.csv", index=False)
    primary = history.loc[history.analysis_role.isin(["primary", "uncorrected_baseline"])]
    keys = ["condition", "method"]
    metric_columns = [f"{metric}_{version}" for metric, _ in METRICS_B for version in ("prior", "fresh")]
    summary = primary.groupby(keys, as_index=False)[metric_columns].mean()
    shown = summary.loc[summary.condition.isin(["raw", "sim_0.00", "sim_0.50", "sim_1.00"])]
    shown.to_csv(out / "two_slice_primary_means_for_article.csv", index=False)
    rank_endpoints(
        shown.loc[shown.condition.eq("sim_1.00") & shown.method.ne("Uncorrected")],
        METRICS_B, ["condition"], out / "two_slice_endpoint_order.csv",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-same", type=Path, required=True)
    parser.add_argument("--previous-real", type=Path, required=True)
    parser.add_argument("--previous-transfer", type=Path, required=True)
    args = parser.parse_args()
    output = args.run / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    compare_same_slice(args, output)
    compare_two_slice(args, output)


if __name__ == "__main__":
    main()
