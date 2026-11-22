#!/usr/bin/env python3
"""Extract inter-region-only spatial marginals and exact top-five LR endpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd

from ccc_current_io import AGES, DATA, INPUTS, LIMITATIONS, REGIONS, ccc_path, input_root, run_info
from extract_ccc_interface import shared_faces

INTERFACES = (("IZ", "DorsalVZ"), ("GE", "VentralVZ"))


def extract(group, codes, xyz, chunk_rows):
    n = len(codes)
    assert tuple(group.attrs["shape"]) == (n, n)
    sender, receiver = np.zeros(n), np.zeros(n)
    mass = np.zeros((8, 8))
    counts = np.zeros((8, 8), dtype=np.int64)
    selected = []
    directions = [(a, b) for a, b in INTERFACES for a, b in ((a, b), (b, a))]
    direction_codes = [(REGIONS.index(a), REGIONS.index(b)) for a, b in directions]
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo + chunk_rows)
        ptr = group["indptr"][lo:hi + 1]
        first, last = int(ptr[0]), int(ptr[-1])
        columns = group["indices"][first:last]
        values = group["data"][first:last]
        assert np.isfinite(values).all() and (values >= 0).all()
        rows = np.repeat(np.arange(lo, hi, dtype=np.int32), np.diff(ptr))
        keep = (values > 0) & (codes[rows] != codes[columns])
        rows, columns, values = rows[keep], columns[keep], values[keep]
        keys = 8 * codes[rows] + codes[columns]
        mass += np.bincount(keys, weights=values, minlength=64).reshape(8, 8)
        counts += np.bincount(keys, minlength=64).reshape(8, 8)
        sender[lo:hi] = np.bincount(rows - lo, weights=values, minlength=hi - lo)
        receiver += np.bincount(columns, weights=values, minlength=n)
        for direction, (a, b) in enumerate(direction_codes):
            take = (codes[rows] == a) & (codes[columns] == b)
            if take.any():
                selected.append(pd.DataFrame({"sender_row": rows[take], "receiver_row": columns[take],
                                               "direction_index": np.full(take.sum(), direction, dtype=np.int8),
                                               "score": values[take]}))
    edges = pd.concat(selected, ignore_index=True)
    delta = xyz[edges.sender_row.to_numpy()] - xyz[edges.receiver_row.to_numpy()]
    edges["distance_um"] = 1000 * np.linalg.norm(delta, axis=1)
    assert edges.distance_um.le(35.001).all()
    edges["sender_region"] = pd.Categorical([directions[k][0] for k in edges.direction_index], categories=REGIONS)
    edges["receiver_region"] = pd.Categorical([directions[k][1] for k in edges.direction_index], categories=REGIONS)
    np.testing.assert_allclose(sender.sum(), mass.sum(), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(receiver.sum(), mass.sum(), rtol=1e-12, atol=1e-9)
    assert np.diag(mass).sum() == 0 and np.diag(counts).sum() == 0
    return sender, receiver, mass, counts, edges


def prepare(raw_root, comparison_root, output, chunk_rows, geometry_path=INPUTS / "ccc/atlas_geometry.json"):
    framing = ["Wnt7a--Fzd2_Lrp5", "Vegfb--Fzd2", "Wnt7a--Fzd8_Lrp5",
               "Wnt7a--Fzd8_Lrp6", "Wnt7a--Fzd2_Lrp6"]
    metrics = pd.read_csv(comparison_root / "LR_region_pair_metrics.csv").set_index("LR").loc[framing].reset_index()
    expected = pd.read_csv(comparison_root / "LR_region_pair_changes.csv")
    source_geometry = json.loads(geometry_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "top5_LR_metrics.csv", index=False)
    stages, validations, support, direction_support = {}, [], [], []
    for age in AGES:
        path = ccc_path(raw_root, age)
        with h5py.File(path, "r") as h:
            info = run_info(h)
            region = read_elem(h["obs/region"])
            codes = pd.Categorical(region, categories=REGIONS).codes.astype(np.int16)
            assert (codes >= 0).all()
            xyz = h["obsm/spatial_3d"][:]
            voxels = np.column_stack([h[f"obs/voxel_{axis}"][:] for axis in "ijk"])
            affine = np.array(source_geometry[age]["affine"])
            np.testing.assert_allclose(xyz, voxels @ affine[:3, :3].T + affine[:3, 3], atol=1e-6, rtol=0)
            frame = pd.DataFrame({"global_row": np.arange(len(region), dtype=np.int32), "region": region,
                                  "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                                  "voxel_i": voxels[:, 0].astype(np.int32),
                                  "voxel_j": voxels[:, 1].astype(np.int32),
                                  "voxel_k": voxels[:, 2].astype(np.int32)})
            frame.to_parquet(output / f"{age}_geometry.parquet", index=False, compression="zstd")
            for a, b in INTERFACES:
                subset = frame.loc[frame.region.isin([a, b])].copy()
                faces = shared_faces(subset, a, b)
                assert len(faces) > 0
                faces.to_parquet(output / f"{age}_{a}_{b}_shared_faces.parquet", index=False, compression="zstd")
            stages[age] = {"path": str(path), "n_positions": len(region),
                           "configuration_id": info["preflight"]["source_configuration_id"],
                           "feast_version": "1.0.6", "geometry": source_geometry[age]}
            for row in metrics.itertuples():
                pair = f"{row.ligand}-{row.receptor}"
                print(f"{age}: extracting {pair}", flush=True)
                sender, receiver, mass, counts, edges = extract(h[f"obsp/commot-User_db-{pair}"], codes, xyz, chunk_rows)
                saved = expected.loc[expected.LR.eq(row.LR)].set_index(["sender_region", "receiver_region"])
                label = "early" if age == AGES[0] else "late"
                for i, a in enumerate(REGIONS):
                    for j, b in enumerate(REGIONS):
                        if a == b:
                            continue
                        record = saved.loc[(a, b)]
                        np.testing.assert_allclose(mass[i, j], record[f"W_{label}"], rtol=1e-11, atol=1e-9)
                        assert counts[i, j] == record[f"n_edges_{label}"]
                total = float(mass.sum())
                keep = (sender > 0) | (receiver > 0)
                pd.DataFrame({"global_row": np.flatnonzero(keep).astype(np.int32),
                              "sender_score": sender[keep], "receiver_score": receiver[keep]}).to_parquet(
                    output / f"{age}_{pair}_inter_region_marginals.parquet", index=False, compression="zstd")
                edges.to_parquet(output / f"{age}_{pair}_interface_edges.parquet", index=False, compression="zstd")
                for role, values in [("sender", sender), ("receiver", receiver)]:
                    positive = values > 0
                    support.append({"stage": age, "LR": row.LR, "role": role, "n_positions": len(values),
                                    "n_positive_positions": int(positive.sum()), "inter_region_mass": total,
                                    "score_sum": float(values.sum()), "n_inter_region_edges": int(counts.sum())})
                for (a, b), part in edges.groupby(["sender_region", "receiver_region"], observed=True):
                    score = float(part.score.sum())
                    np.testing.assert_allclose(score, mass[REGIONS.index(a), REGIONS.index(b)], rtol=1e-12, atol=1e-9)
                    assert len(part) == counts[REGIONS.index(a), REGIONS.index(b)]
                    direction_support.append({"stage": age, "LR": row.LR, "sender_region": a, "receiver_region": b,
                                              "W": score, "share_of_all_inter_region_pct": 100 * score / total,
                                              "n_edges": len(part), "n_positive_senders": part.sender_row.nunique(),
                                              "n_positive_receivers": part.receiver_row.nunique(),
                                              "max_distance_um": float(part.distance_um.max())})
                validations.append({"stage": age, "LR": row.LR, "all_56_W_and_counts_match": True,
                                    "sender_receiver_conservation": True, "native_geometry_matches_affine": True,
                                    "all_focus_edges_within_35_um": True, "passed": True})
                print(f"{age}: {pair}, {int(counts.sum()):,} cross-region edges; all regional weights/counts verified", flush=True)
    pd.DataFrame(support).to_csv(output / "spatial_support.csv", index=False)
    pd.DataFrame(direction_support).to_csv(output / "interface_direction_support.csv", index=False)
    pd.DataFrame(validations).to_csv(output / "validation.csv", index=False)
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "entry_point": str(Path(__file__).resolve()),
            "input_root": str(raw_root.resolve()), "comparison_root": str(comparison_root.resolve()), "sources": stages,
            "selected_LRs": metrics.LR.tolist(), "region_order": REGIONS, "interfaces": INTERFACES,
            "marginals": "Exact row/column sums of positive A_ij restricted to sender_region != receiver_region; every within-region edge is excluded.",
            "sparse_tables": "Only rows with positive sender or receiver scores are stored; every omitted native position has exact zero in both fields, and remains in the geometry table.",
            "interface_edges": "All positive edges in both directions across IZ–DorsalVZ and GE–VentralVZ. Global rows index the complete native geometry.",
            "shared_faces": "Native 20-um voxel faces between the two annotated regions; anatomical, independent of LR score.",
            "new_inference": False, "limitations": LIMITATIONS}
    (output / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Top-five spatial source data complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=input_root())
    parser.add_argument("--comparison-root", type=Path, default=DATA / "region_pairs")
    parser.add_argument("--output-dir", type=Path, default=DATA / "top5_spatial")
    parser.add_argument("--chunk-rows", type=int, default=250000)
    parser.add_argument("--geometry", type=Path, default=INPUTS / "ccc/atlas_geometry.json")
    args = parser.parse_args()
    prepare(args.input_root, args.comparison_root, args.output_dir, args.chunk_rows, args.geometry)
