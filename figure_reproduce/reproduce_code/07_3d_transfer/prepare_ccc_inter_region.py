#!/usr/bin/env python3
"""Recalculate the approved regional-participation JSD using cross-region mass."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

from ccc_current_io import AGES, DATA, REGIONS


def prepare(source: Path, output: Path) -> None:
    provenance = json.loads((source / "provenance.json").read_text())
    edges = pd.read_parquet(source / "LR_region_edges.parquet")
    previous = pd.read_csv(source / "LR_redistribution_metrics.csv").set_index("LR")
    status = pd.read_csv(source / "shared_LR_status.csv")
    cross = edges.loc[edges.sender_region.ne(edges.receiver_region)].copy()
    assert len(status) == 85 and len(previous) == 45
    assert not cross.sender_region.eq(cross.receiver_region).any()
    assert np.isfinite(cross.W).all() and cross.W.ge(0).all()
    metrics, profiles, checks = [], [], []
    for key, old in previous.iterrows():
        ligand, receptor = key.split("--")
        selected = cross.loc[cross.ligand.eq(ligand) & cross.receptor.eq(receptor)]
        stage_profiles, support = [], {}
        for stage, label in zip(AGES, ("early", "late")):
            stage_edges = selected.loc[selected.stage.eq(stage)]
            assert len(stage_edges) == len(REGIONS) * (len(REGIONS) - 1)
            outgoing = stage_edges.groupby("sender_region").W.sum().reindex(REGIONS)
            incoming = stage_edges.groupby("receiver_region").W.sum().reindex(REGIONS)
            total = float(stage_edges.W.sum())
            assert total > 0, f"No inter-region mass for {stage} {key}; composition undefined."
            profile = (outgoing.to_numpy() + incoming.to_numpy()) / (2 * total)
            np.testing.assert_allclose(profile.sum(), 1, atol=1e-12)
            np.testing.assert_allclose(total, old[f"{label}_total_nonself_mass"] * old[f"F_inter_{label}"], rtol=1e-11)
            stage_profiles.append(profile)
            support[f"{label}_inter_region_mass"] = total
            support[f"{label}_inter_region_edges"] = int(stage_edges.n_edges.sum())
            support[f"{label}_positive_directed_region_pairs"] = int(stage_edges.W.gt(0).sum())
            checks.append({"stage": stage, "LR": key, "n_region_pairs": len(stage_edges),
                           "M_sum": float(profile.sum()), "inter_region_mass": total,
                           "mass_matches_previous_off_diagonal": True, "passed": True})
        early, late = stage_profiles
        js = float(jensenshannon(early, late, base=2) ** 2)
        midpoint = (early + late) / 2
        direct = float((rel_entr(early, midpoint).sum() + rel_entr(late, midpoint).sum()) / (2 * np.log(2)))
        np.testing.assert_allclose(js, direct, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose((late - early).sum(), 0, atol=1e-12)
        metrics.append({"ligand": ligand, "receptor": receptor, "LR": key,
                        "JSD_base2": js, "previous_all_region_JSD_rank": int(old.JSD_rank), **support})
        for index, region in enumerate(REGIONS):
            profiles.append({"ligand": ligand, "receptor": receptor, "LR": key, "region": region,
                             "M_early": early[index], "M_late": late[index],
                             "delta_M_pp": 100 * (late[index] - early[index])})
    metrics = pd.DataFrame(metrics).sort_values(["JSD_base2", "LR"], ascending=[False, True]).reset_index(drop=True)
    metrics.insert(0, "JSD_rank", np.arange(1, len(metrics) + 1))
    for stage in AGES:
        present = cross.loc[cross.stage.eq(stage)].groupby(["ligand", "receptor"]).W.sum().gt(0)
        status[f"{stage}_inter_region_nonzero"] = [int(present.get((row.ligand, row.receptor), False)) for row in status.itertuples()]
        assert status[f"{stage}_inter_region_nonzero"].sum() == 45
    output.mkdir(parents=True, exist_ok=True)
    cross.to_parquet(output / "LR_inter_region_edges.parquet", index=False, compression="zstd")
    metrics.to_csv(output / "LR_redistribution_metrics.csv", index=False)
    pd.DataFrame(profiles).to_csv(output / "LR_region_profiles.csv", index=False)
    status.to_csv(output / "shared_LR_status.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "aggregation_validation.csv", index=False)
    for name in ("region_volume_fraction_change.csv", "region_volume_fractions.csv"):
        pd.read_csv(source / name).to_csv(output / name, index=False)
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "entry_point": str(Path(__file__).resolve()),
            "input_root": str(source.resolve()), "sources": provenance["sources"], "region_order": REGIONS,
            "input_validation": "Completed-run regional W and counts were verified against CSR matrices by prepare_ccc_redistribution.py.",
            "filter": "Retain sender_region != receiver_region; exclude ALL same-region interactions, including edges between different positions.",
            "W": "Unchanged completed-run positive communication mass, summed by directed region pair; no new inference.",
            "M": "M_r_inter = (sum_{b != r} W_rb + sum_{a != r} W_ar) / (2 * sum_{a != b} W_ab).",
            "normalization": "Separately within each LR and stage, using inter-region mass only. Each retained edge contributes half to each endpoint.",
            "JSD": "Same base-2 Jensen-Shannon divergence of eight regional participation shares as before; scipy jensenshannon squared, checked with relative entropy.",
            "ranking": "All 45 pairs with inter-region mass at both stages; descending JSD, alphabetical LR tie break. No support or significance filter.",
            "candidate_count": 85, "analyzed_pair_count": len(metrics),
            "limitations": provenance["limitations"][:7] + [
                "Participation combines sending and receiving and does not retain region-pair direction or partner identity.",
                "Inter-region composition is conditional on cross-region mass; it does not measure total activity or the fraction of signaling crossing a boundary.",
                "Region voxel fractions are a visual reference, not contact-area or eligible-neighbor normalization.",
                "LR pairs sharing molecular components are not independent replicates. No developmental significance test or representative-case selection."]}
    (output / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    print(metrics[["JSD_rank", "LR", "JSD_base2", "previous_all_region_JSD_rank"]].head(15).to_string(index=False))
    print(f"Saved {len(metrics)} inter-region LR comparisons to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DATA / "redistribution")
    parser.add_argument("--output-dir", type=Path, default=DATA / "redistribution_inter_region")
    args = parser.parse_args()
    prepare(args.input_root, args.output_dir)
