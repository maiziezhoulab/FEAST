"""Export the realized FEAST count diagnostics for the registered factor profiles."""

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

FACTORS = {"mean": 1.0, "variance": 1.0, "sparsity_logit_shift": 0.0}
FIELDS = ("mean", "variance", "zero_prop")
KEYS = ["slice_id", "factor", "simulation_id", "nominal_level"]


def load_profile_data(study):
    run = study / "run"
    manifest = pd.read_csv(run / "simulation_manifest.csv", dtype={"slice_id": str})
    if len(manifest) != 81 or not manifest.status.eq("ok").all():
        raise ValueError("The 81 registered simulations must complete before assembling the profiles")
    rows = []
    for row in manifest.itertuples(index=False):
        if row.alteration_type not in FACTORS and row.simulation_id != "baseline":
            continue
        data = ad.read_h5ad(run / row.file_path, backed="r")
        try:
            realized = data.uns["alteration_diagnostics"]["realized_stage_achieved_change"]
            observed = {key: float(realized[key]) for key in FIELDS}
        finally:
            data.file.close()
        factors = FACTORS if row.simulation_id == "baseline" else {row.alteration_type: float(row.alteration_value)}
        for factor, nominal in factors.items():
            rows.append({
                "slice_id": row.slice_id, "factor": "sparsity" if factor == "sparsity_logit_shift" else factor,
                "simulation_id": row.simulation_id, "nominal_level": nominal, **observed,
            })
    profile = pd.DataFrame(rows).sort_values(KEYS)
    if len(profile) != 63 or profile.duplicated(KEYS).any():
        raise ValueError("Expected 63 realized factor-profile rows")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    data = load_profile_data(args.run_dir)
    output = args.run_dir / "run"
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "intervention_profile_data.csv", index=False)
    if args.previous:
        prior = pd.read_csv(args.previous, dtype={"slice_id": str})
        compared = data.merge(prior[KEYS + list(FIELDS)], on=KEYS, how="outer", indicator=True,
                              suffixes=("_fresh", "_prior"), validate="one_to_one")
        if len(compared) != 63 or not compared["_merge"].eq("both").all():
            raise ValueError("The fresh and public count diagnostics must cover the same 63 conditions")
        for metric in FIELDS:
            compared[f"{metric}_difference"] = compared[f"{metric}_fresh"] - compared[f"{metric}_prior"]
        destination = args.run_dir / "comparison"
        destination.mkdir(parents=True, exist_ok=True)
        compared.drop(columns="_merge").to_csv(destination / "realized_profile_old_vs_new.csv", index=False)


if __name__ == "__main__":
    main()
