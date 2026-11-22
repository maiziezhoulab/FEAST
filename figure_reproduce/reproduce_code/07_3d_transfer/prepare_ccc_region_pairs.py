#!/usr/bin/env python3
"""Compare each LR's 56 directed cross-region connections between stages."""
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
    edges = pd.read_parquet(source / "LR_inter_region_edges.parquet")
    assert edges.sender_region.ne(edges.receiver_region).all()
    assert np.isfinite(edges.W).all() and edges.W.ge(0).all()
    pairs = [(a, b) for a in REGIONS for b in REGIONS if a != b]
    pair_index = pd.MultiIndex.from_tuples(pairs, names=["sender_region", "receiver_region"])
    rows, metrics, checks = [], [], []
    for (ligand, receptor), group in edges.groupby(["ligand", "receptor"], sort=True):
        key = f"{ligand}--{receptor}"
        by_stage, totals, distributions = {}, {}, []
        for stage in AGES:
            frame = group.loc[group.stage.eq(stage)].set_index(["sender_region", "receiver_region"])
            assert frame.index.is_unique and len(frame) == 56
            frame = frame.loc[pair_index]
            total = float(frame.W.sum())
            assert total > 0, f"Undefined cross-region composition: {stage} {key}"
            distribution = frame.W.to_numpy() / total
            np.testing.assert_allclose(distribution.sum(), 1, atol=1e-12)
            by_stage[stage], totals[stage] = frame, total
            distributions.append(distribution)
            checks.append({"stage": stage, "LR": key, "n_directed_region_pairs": len(frame),
                           "share_sum": float(distribution.sum()), "inter_region_mass": total,
                           "n_positive_edges": int(frame.n_edges.sum()), "passed": True})
        early, late = distributions
        delta = 100 * (late - early)
        np.testing.assert_allclose(delta.sum(), 0, atol=1e-12)
        js = float(jensenshannon(early, late, base=2) ** 2)
        midpoint = (early + late) / 2
        direct = float((rel_entr(early, midpoint).sum() + rel_entr(late, midpoint).sum()) / (2 * np.log(2)))
        np.testing.assert_allclose(js, direct, rtol=1e-11, atol=1e-14)
        record = {"ligand": ligand, "receptor": receptor, "LR": key, "JSD_base2": js}
        for stage, label in zip(AGES, ("early", "late")):
            record[f"{label}_inter_region_mass"] = totals[stage]
            record[f"{label}_inter_region_edges"] = int(by_stage[stage].n_edges.sum())
            record[f"{label}_positive_directed_region_pairs"] = int(by_stage[stage].W.gt(0).sum())
        for label, index in (("largest_gain", int(np.argmax(delta))), ("largest_loss", int(np.argmin(delta)))):
            a, b = pairs[index]
            record.update({f"{label}_sender": a, f"{label}_receiver": b,
                           f"{label}_early_pct": 100 * early[index], f"{label}_late_pct": 100 * late[index],
                           f"{label}_delta_pp": delta[index]})
        metrics.append(record)
        for index, (a, b) in enumerate(pairs):
            early_row, late_row = by_stage[AGES[0]].iloc[index], by_stage[AGES[1]].iloc[index]
            rows.append({"ligand": ligand, "receptor": receptor, "LR": key,
                         "sender_region": a, "receiver_region": b, "region_pair": f"{a}→{b}",
                         "region_pair_order": index + 1, "W_early": early_row.W, "W_late": late_row.W,
                         "n_edges_early": int(early_row.n_edges), "n_edges_late": int(late_row.n_edges),
                         "early_share_pct": 100 * early[index], "late_share_pct": 100 * late[index],
                         "delta_share_pp": delta[index], "abs_delta_share_pp": abs(delta[index])})
    metrics = pd.DataFrame(metrics).sort_values(["JSD_base2", "LR"], ascending=[False, True]).reset_index(drop=True)
    metrics.insert(0, "JSD_rank", np.arange(1, len(metrics) + 1))
    comparison = pd.DataFrame(rows).merge(metrics[["LR", "JSD_rank"]], on="LR", validate="many_to_one")
    comparison = comparison.sort_values(["JSD_rank", "abs_delta_share_pp", "region_pair_order"], ascending=[True, False, True])
    comparison["change_rank_within_LR"] = comparison.groupby("LR").cumcount() + 1
    comparison = comparison.sort_values(["JSD_rank", "region_pair_order"])
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "LR_region_pair_changes.csv", index=False)
    metrics.to_csv(output / "LR_region_pair_metrics.csv", index=False)
    comparison.sort_values(["abs_delta_share_pp", "JSD_rank", "region_pair_order"], ascending=[False, True, True]).to_csv(
        output / "LR_region_pair_changes_by_magnitude.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "aggregation_validation.csv", index=False)
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "entry_point": str(Path(__file__).resolve()),
            "input_root": str(source.resolve()), "sources": provenance["sources"], "region_order": REGIONS,
            "region_pair_order": [f"{a}→{b}" for a, b in pairs], "candidate_count": 85,
            "analyzed_LR_count": len(metrics), "directed_region_pairs_per_LR": len(pairs),
            "comparison": "E18-derived / DevCCF E18.5 minus E14-derived / DevCCF E15.5",
            "W": "Unchanged completed-run positive communication mass and edge counts for each LR, sender region and receiver region.",
            "P": "P_ab = W_ab / sum_{c != d} W_cd, separately for every LR and stage. Only a != b; all 56 directions retained.",
            "difference": "100 * (P_ab_late - P_ab_early), percentage points. Raw W and positive-edge counts are supplied alongside shares.",
            "JSD": "Base-2 Jensen-Shannon divergence between the two 56-entry directed region-pair distributions; scipy jensenshannon squared, checked by relative entropy.",
            "ranking": "All 45 LR pairs ranked by descending directed-region-pair JSD; alphabetical LR tie break. No support or significance filtering.",
            "difference_from_participation": "Sending direction and partner identities are retained; no averaging of outgoing and incoming region marginals.",
            "limitations": provenance["limitations"][:7] + [
                "Share changes describe relative allocation within each LR's cross-region mass, not absolute activity or statistical significance.",
                "Stage volumes, label geometry and eligible neighbor opportunities differ; no contact-area, voxel-count or neighbor-opportunity correction is applied.",
                "The 40 original zero LR candidates are not assigned a probability distribution or pseudocounts.",
                "LR pairs sharing components are related entries, not independent replicates. No new inference, developmental significance test or biological-case selection."]}
    (output / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    print(metrics[["JSD_rank", "LR", "JSD_base2", "largest_gain_sender", "largest_gain_receiver", "largest_gain_delta_pp"]].head(15).to_string(index=False))
    print(f"Saved {len(comparison)} LR × directed-region-pair comparisons to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DATA / "redistribution_inter_region")
    parser.add_argument("--output-dir", type=Path, default=DATA / "region_pairs")
    args = parser.parse_args()
    prepare(args.input_root, args.output_dir)
