"""Compare angular and placement recovery with the public fixed-plate rows."""

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["method", "alteration", "angle_degrees"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-spateo", type=Path, required=True)
    parser.add_argument("--previous-paste2", type=Path, required=True)
    args = parser.parse_args()
    prior = pd.concat([pd.read_csv(path) for path in (args.previous_spateo, args.previous_paste2)])
    fresh = pd.concat([
        pd.read_csv(args.run / "spateo_scores/alignment_metrics.csv"),
        pd.read_csv(args.run / "paste2/scores/alignment_metrics.csv"),
    ])
    numeric = [column for column in fresh.select_dtypes(include="number") if column in prior and column not in KEYS]
    evidence = fresh[KEYS + ["job_id"] + numeric].merge(prior[KEYS + numeric], how="outer", indicator=True,
                                                       on=KEYS, validate="one_to_one", suffixes=("_fresh", "_prior"))
    if len(evidence) != 56 or not evidence["_merge"].eq("both").all():
        raise ValueError("Alignment comparison must join all 56 public jobs")
    for column in numeric:
        evidence[f"{column}_difference"] = evidence[f"{column}_fresh"] - evidence[f"{column}_prior"]
    output = args.run / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    evidence.drop(columns="_merge").to_csv(output / "alignment_old_vs_new.csv", index=False)
    core = []
    for name, keys, bound in [
        ("baseline_both_methods", evidence.alteration.eq("baseline"), 0.03),
        ("spateo_all_conditions", evidence.method.eq("spateo"), 0.03),
        ("variance_both_methods", evidence.alteration.eq("variance_2.0"), 0.02),
    ]:
        subset = evidence.loc[keys]
        record = {"comparison": name, "bound_degrees": bound, "rows": len(subset)}
        for version in ("prior", "fresh"):
            maximum = subset[f"rotation_recovery_error_{version}"].max()
            record[f"{version}_maximum_degrees"] = maximum
            record[f"{version}_within_bound"] = maximum <= bound
        core.append(record)
    pd.DataFrame(core).to_csv(output / "article_rotation_bounds.csv", index=False)
    evidence.loc[
        evidence.angle_degrees.eq(75) & evidence.alteration.isin(["mean_0.5", "sparsity_0.5"]),
        KEYS + [f"{column}_{version}" for column in
                ("retained_fraction", "rotation_recovery_error", "mean_spatial_error_spot_units")
                for version in ("prior", "fresh")],
    ].to_csv(output / "article_75_degree_endpoints.csv", index=False)


if __name__ == "__main__":
    main()
