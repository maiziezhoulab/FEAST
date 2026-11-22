"""Tabulate the archived calibration and the verified comparison distributions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_archive import METRICS, STUDY


def summarize(zenodo, source_dir=None):
    input_dir = zenodo / "inputs" / STUDY
    results = zenodo / "results" / STUDY
    derived = source_dir or results / "figure05_sources"
    construction = results / "reconstruction"
    plan = json.loads((construction / "plan.json").read_text())
    calibration_source = json.loads((input_dir / "article_settings/calibration_source_plan.json").read_text())
    preflight = json.loads((construction / "preflight.json").read_text())
    assert preflight["calibration_reestimated"] is False
    grid = []
    for name, arm in plan["densities"].items():
        gap = arm["spec"]["gap"]
        record = arm["assignment_randomness_preflight"]
        assert record == calibration_source["densities"][name]["assignment_randomness_preflight"]
        estimates = record["representative_estimates"]
        assert len(estimates) == 3 and record["reference_only"]
        for fold in estimates:
            scores = fold["scores"]
            np.testing.assert_allclose([row["ar"] for row in scores], np.arange(11)*.05, atol=1e-13, rtol=0)
            best = min(scores, key=lambda row: (-row["moran_correlation"], row["ar"]))
            assert best["ar"] == fold["assignment_randomness"]
            grid.extend({"gap": gap, "holdout": fold["holdout"], "selected_ar": fold["assignment_randomness"],
                         "median_ar": record["observed_assignment_randomness"], **score} for score in scores)
        assert np.median([fold["assignment_randomness"] for fold in estimates]) == record["observed_assignment_randomness"]
        assert all(row["seed"] == 2026 + 10000*gap + row["target_index"] for row in arm["targets"])
    assert len(grid) == 99
    pd.DataFrame(grid).to_csv(derived / "archived_calibration.csv", index=False)
    metrics = pd.read_csv(derived / "panel_d_metrics.csv")
    supported = pd.read_csv(derived / "panel_d_support.csv")
    records = []
    for metric in METRICS:
        paired = metrics.merge(supported.loc[supported.metric.eq(metric), ["gap", "target_slice"]],
                               on=["gap", "target_slice"], validate="many_to_one")
        for (gap, method), group in paired.groupby(["gap", "method"]):
            values = group[metric].dropna().to_numpy()
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            within = values[(values >= q1-1.5*(q3-q1)) & (values <= q3+1.5*(q3-q1))]
            records.append({"metric": metric, "gap": gap, "method": method, "n_targets": len(values),
                            "q1": q1, "median": median, "q3": q3,
                            "lower_whisker": within.min(), "upper_whisker": within.max()})
    pd.DataFrame(records).to_csv(derived / "panel_d_summary.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    p.add_argument("--sources", type=Path)
    args = p.parse_args()
    summarize(args.zenodo, args.sources)
