"""Compare deconvolution scores on the shared support and resolution contrasts."""

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["slice_id", "resolution", "method"]
METRICS = ("jsd_mean", "pearson_mean", "rmse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    fresh = pd.read_csv(args.run / "cell_class/scores/deconvolution_scores.csv", dtype={"slice_id": str})
    prior = pd.read_csv(args.previous, dtype={"slice_id": str})
    columns = [*METRICS, "n_retained_cell_classes", "n_scored_spots", "n_input_spots", "n_zero_library_spots_excluded"]
    history = fresh[KEYS + columns].merge(prior[KEYS + columns], on=KEYS, how="outer", indicator=True,
                                         suffixes=("_fresh", "_prior"), validate="one_to_one")
    if len(history) != 12 or not history["_merge"].eq("both").all():
        raise ValueError("Deconvolution comparison must join all 12 public method jobs")
    for column in columns:
        history[f"{column}_difference"] = history[f"{column}_fresh"] - history[f"{column}_prior"]
    output = args.run / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    history.drop(columns="_merge").to_csv(output / "deconvolution_old_vs_new.csv", index=False)
    contrast = []
    for (section, method), rows in history.groupby(["slice_id", "method"]):
        rows = rows.set_index("resolution")
        for metric in METRICS:
            prior = float(rows.loc[0.25, f"{metric}_prior"] - rows.loc[0.10, f"{metric}_prior"])
            fresh = float(rows.loc[0.25, f"{metric}_fresh"] - rows.loc[0.10, f"{metric}_fresh"])
            contrast.append({"slice_id": section, "method": method, "metric": metric,
                             "delta_0_25_minus_0_10_prior": prior, "delta_0_25_minus_0_10_fresh": fresh,
                             "direction_changed": (fresh > 0) != (prior > 0) or (fresh == 0) != (prior == 0)})
    pd.DataFrame(contrast).to_csv(output / "resolution_contrasts.csv", index=False)


if __name__ == "__main__":
    main()
