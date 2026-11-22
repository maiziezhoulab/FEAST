"""Compare fresh clustering scores with the public Figure 3 benchmark rows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["slice_id", "simulation_id", "method"]
AGREEMENT = ["ARI", "NMI", "AMI"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    fresh = pd.read_csv(args.run / "metrics/fixed_panel_benchmark_metrics.csv", dtype={"slice_id": str})
    prior = pd.read_csv(args.previous, dtype={"slice_id": str})
    numeric = [column for column in fresh.select_dtypes(include="number") if column in prior]
    history = fresh.merge(prior[KEYS + numeric], on=KEYS, how="outer", indicator=True,
                          suffixes=("_fresh", "_prior"), validate="one_to_one")
    if len(history) != 252 or not history["_merge"].eq("both").all():
        raise ValueError("Clustering comparison must join all 252 atomic rows")
    output = args.run / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    for column in numeric:
        history[f"{column}_difference"] = history[f"{column}_fresh"] - history[f"{column}_prior"]
    history.drop(columns="_merge").to_csv(output / "clustering_old_vs_new.csv", index=False)

    shown = pd.read_csv(args.run / "figure3/a_scores.csv", dtype={"slice_id": str})
    scores = shown[KEYS + ["logit_shift"]].merge(history[KEYS + [
        f"{metric}_{version}" for metric in AGREEMENT for version in ("prior", "fresh")
    ]], on=KEYS, validate="one_to_one")
    near_zero = scores.loc[scores.logit_shift.eq(0)].set_index(["slice_id", "method"])
    paired = scores.loc[scores.logit_shift.ne(0)].copy()
    for metric in AGREEMENT:
        for version in ("prior", "fresh"):
            key = f"{metric}_{version}"
            group = pd.MultiIndex.from_frame(paired[["slice_id", "method"]])
            paired[f"{key}_minus_neutral"] = paired[key] - near_zero[key].reindex(group).to_numpy()
        paired[f"{metric}_direction_changed"] = (
            np.sign(paired[f"{metric}_fresh_minus_neutral"]) != np.sign(paired[f"{metric}_prior_minus_neutral"])
        )
    paired.to_csv(output / "sparsity_agreement_against_simulated_neutral.csv", index=False)

    displayed = pd.read_csv(args.run / "figure3/b_ari.csv", dtype={"slice_id": str})
    maps = scores.merge(displayed[KEYS + ["ARI", "printed_ARI"]].rename(columns={"ARI": "map_ARI"}),
                        on=KEYS, validate="one_to_one")
    if len(maps) != 15 or (maps.map_ARI - maps.ARI_fresh).abs().max() > 1e-12:
        raise ValueError("The displayed ARIs differ from the fresh score rows")
    for version in ("prior", "fresh"):
        maps[f"rank_{version}"] = maps.groupby("logit_shift")[f"ARI_{version}"].rank(method="min", ascending=False)
    maps["rank_changed"] = maps.rank_fresh.ne(maps.rank_prior)
    maps.sort_values(["logit_shift", "rank_fresh"]).to_csv(output / "figure3b_ari_and_order.csv", index=False)


if __name__ == "__main__":
    main()
