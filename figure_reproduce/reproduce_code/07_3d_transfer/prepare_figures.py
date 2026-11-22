"""Extract the shared Figure 6–7 communication tables from verified archives."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ccc_current_io import AGES, STUDY
from prepare_ccc_figure5 import prepare as network
from prepare_ccc_inter_region import prepare as inter_region
from prepare_ccc_merged_contacts import prepare as contacts
from prepare_ccc_observed_slices import prepare as observed
from prepare_ccc_redistribution import prepare as redistribute
from prepare_ccc_region_pairs import prepare as region_pairs
from prepare_ccc_top5_spatial import prepare as spatial

STUDY_ID = STUDY.name
INTERFACES = (("IZ", "DorsalVZ"), ("GE", "VentralVZ"))
RECEPTORS = ("Fzd2_Lrp5", "Fzd8_Lrp6")


def directional_opportunity(atlas, output):
    """Count native pairs within 35 µm and report directional mass per 1,000 pairs."""
    masses = pd.read_csv(atlas / "top5_spatial/interface_direction_support.csv").set_index(
        ["stage", "LR", "sender_region", "receiver_region"], verify_integrity=True)
    records = []
    for age in AGES:
        geometry = pd.read_parquet(atlas / "top5_spatial" / f"{age}_geometry.parquet",
                                   columns=["region", "x", "y", "z"])
        for forward in INTERFACES:
            a, b = forward
            xyz_a, xyz_b = (geometry.loc[geometry.region.eq(region), ["x", "y", "z"]].to_numpy()
                            for region in forward)
            tree_a, tree_b = cKDTree(xyz_a), cKDTree(xyz_b)
            total = tree_a.count_neighbors(tree_b, .035)
            independently_counted = tree_b.query_ball_point(xyz_a, .035, workers=2, return_length=True).sum()
            assert total == independently_counted > 0
            for receptor in RECEPTORS:
                lr = f"Wnt7a--{receptor}"
                for sender, receiver in (forward, (b, a)):
                    row = masses.loc[(age, lr, sender, receiver)]
                    records.append({"stage": age, "LR": lr, "receptor": receptor, "interface": f"{a}–{b}",
                                    "sender_region": sender, "receiver_region": receiver,
                                    "W": row.W, "n_edges": int(row.n_edges), "eligible_pairs": total,
                                    "mass_per_1000_pairs": 1000 * row.W / total})
    table = pd.DataFrame(records)
    keys = ["LR", "sender_region", "receiver_region"]
    rows = table.pivot(index=keys, columns="stage", values="mass_per_1000_pairs")
    folds = (rows[AGES[1]] / rows[AGES[0]].where(rows[AGES[0]] > 0)).rename("late_over_early")
    table = table.join(folds, on=keys)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "directional_opportunity.csv", index=False)
    return table


def prepare(zenodo, output_root=None, verified_root=None):
    results = zenodo / "results" / STUDY_ID
    inputs = zenodo / "inputs" / STUDY_ID
    archive = results / "ccc"
    verified = verified_root or results / "ccc_verified"
    proof = json.loads((verified / "verification.json").read_text())
    assert proof["status"] == "passed" and proof["matrix_count"] == 340
    output = output_root or results
    atlas = output / "ccc_atlas"
    network(STUDY / "ccc_rerun/config.json", output / "ccc_figure5",
            archive, verified, inputs / "ccc/shared_85_LR.csv")
    redistribute(archive, atlas / "redistribution", 250000, verified)
    inter_region(atlas / "redistribution", atlas / "redistribution_inter_region")
    region_pairs(atlas / "redistribution_inter_region", atlas / "region_pairs")
    spatial(archive, atlas / "region_pairs", atlas / "top5_spatial", 250000, inputs / "ccc/atlas_geometry.json")
    contacts(atlas / "region_pairs", atlas / "top5_spatial", atlas / "region_pairs_merged_contact")
    d = directional_opportunity(atlas, atlas / "wnt_biology")
    observed(inputs / "GSE269617/h5ad_region_annotated", atlas / "observed_st")

    hist = results / "comparison"
    keys = ["LR", "stage", "sender_region", "receiver_region"]
    expected = pd.read_csv(hist / "directional_source_data.csv").set_index(keys)
    actual = d.set_index(keys).loc[expected.index]
    np.testing.assert_array_equal(actual.eligible_pairs, expected.eligible_pairs)
    fields = ["W", "mass_per_1000_pairs", "late_over_early"]
    np.testing.assert_allclose(actual[fields], expected[fields], rtol=1e-10, atol=1e-12)
    for name in ("measured_RNA_source_data", "measured_receptor_codetection"):
        saved = pd.read_csv(hist / f"{name}.csv")
        recomputed = pd.read_csv(atlas / "observed_st" / f"{name}.csv")
        key = ["LR", "sample"] + (["gene", "region"] if "gene" in saved else [])
        saved = saved.set_index(key).sort_index()
        recomputed = recomputed.set_index(key).sort_index().reindex(saved.index)
        pd.testing.assert_frame_equal(recomputed[saved.columns], saved, rtol=1e-10, atol=1e-12)
    area = pd.read_csv(atlas / "region_pairs_merged_contact/LR_region_pair_changes.csv")
    np.testing.assert_allclose(area.groupby("LR").delta_share_pp.sum(), 0, rtol=0, atol=1e-11)
    report = {"complete_raw_launch": proof["launch"], "verified_candidate_matrices": proof["matrix_count"],
              "directional_rows": len(d), "historical_directions_verified": len(expected),
              "measured_means_verified": 12, "co_detection_rows_verified": 4,
              "contact_profile_columns": int(area.LR.nunique()), "contact_pairs": int(area.region_pair.nunique())}
    (atlas / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verified-root", type=Path)
    args = parser.parse_args()
    prepare(args.zenodo, args.output_root, args.verified_root)
