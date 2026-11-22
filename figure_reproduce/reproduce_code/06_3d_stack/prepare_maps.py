"""Read the archived Figure 5c illustration and the native-coordinate Figure 5e markers."""
import argparse
import json
from pathlib import Path

from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd

STUDY = "06_3d_stack"
GENES = ("Gad2", "Gfap")
PANELS = [("Observed", 0)] + [(method, gap) for gap in (3, 5, 10) for method in ("FEAST", "SpatialZ")]


def read_slice(path, genes=None, libraries=False):
    with h5py.File(path) as file:
        obs = read_elem(file["obs"])
        xy = file["obsm/spatial"][:]
        xyz = file["obsm/spatial_3d"][:]
        matrix = file["layers/counts"]
        if genes:
            names = read_elem(file["var"]).index.tolist()
            columns = np.array([names.index(gene) for gene in genes])
            order = np.argsort(columns)
            data = (matrix[:, columns[order]][:, np.argsort(order)] if isinstance(matrix, h5py.Dataset)
                    else read_elem(matrix)[:, columns].toarray())
        elif libraries:
            data = np.log1p(np.asarray(read_elem(matrix).sum(axis=1), dtype=float)).ravel()
        else:
            data = None
    return obs, xy, xyz, data


def illustration(inputs, output):
    location = inputs / "illustration_083"
    archive = json.loads((location / "provenance.json").read_text())
    plan = json.loads((location / "plan.json").read_text())
    target = next(row for row in plan["densities"]["dense_gap3"]["targets"] if row["target_slice"] == 83)
    assert target["lower_ref_slice"] == archive["lower_ref_slice"] == 82
    assert target["upper_ref_slice"] == archive["upper_ref_slice"] == 84
    assert target["reference_weights"] == archive["reference_weights"]
    assert archive["target_seed"] == 32052 and not archive["target_expression_accessed"]
    actual, measured = {}, []
    for sid in range(80, 87):
        obs, xy, xyz, library = read_slice(inputs / "h5ad" / f"Zhuang-ABCA-1.{sid:03d}.h5ad", libraries=sid != 83)
        actual[sid] = (obs, xy, xyz)
        measured.append(pd.DataFrame({"slice_id": sid, "source": "blueprint" if sid == 83 else "measured",
                                      "spot_id": obs.index, "x": xy[:, 0], "y": xy[:, 1], "z": xyz[:, 2],
                                      "log1p_library": np.nan if library is None else library}))
    obs, xy, xyz, library = read_slice(location / "generated.h5ad", libraries=True)
    observed, source_xy, source_xyz = actual[83]
    assert obs.index.equals(observed.index)
    np.testing.assert_array_equal(obs["class"].astype(str), observed["class"].astype(str))
    np.testing.assert_array_equal(xy, source_xy)
    np.testing.assert_array_equal(xyz, source_xyz)
    measured.append(pd.DataFrame({"slice_id": 83, "source": "generated", "spot_id": obs.index,
                                  "x": xy[:, 0], "y": xy[:, 1], "z": xyz[:, 2], "log1p_library": library}))
    frame = pd.concat(measured, ignore_index=True)
    frame.to_parquet(output / "illustration_083.parquet", index=False, compression="zstd")
    record = {"measured_slices": list(actual), "target": 83, "brackets": [82, 84],
              "target_spots": len(obs), "archive_FEAST": archive["feast_version"], "archive_commit": archive["feast_commit"],
              "reference_weights": archive["reference_weights"], "assignment_randomness": archive["assignment_randomness"],
              "seed": archive["target_seed"], "display_quantiles": [.02, .98],
              "source": str(location.relative_to(inputs.parents[1]) / "provenance.json")}
    (output / "illustration_083.json").write_text(json.dumps(record, indent=2) + "\n")


def prepare(zenodo, output):
    inputs = zenodo / "inputs" / STUDY
    results = zenodo / "results" / STUDY
    reconstruction = results / "reconstruction"
    plan = json.loads((reconstruction / "plan.json").read_text())
    arms = {arm["spec"]["gap"]: (name, set(arm["reference_pool"])) for name, arm in plan["densities"].items()}
    output.mkdir(parents=True, exist_ok=True)
    illustration(inputs, output)
    points = []
    for i, sid in enumerate(plan["available_slices"], 1):
        name = f"Zhuang-ABCA-1.{sid:03d}"
        real = read_slice(inputs / "h5ad" / f"{name}.h5ad", GENES)
        for method, gap in PANELS:
            reference = gap == 0 or sid in arms[gap][1]
            if reference:
                obs, _, xyz, counts = real
            else:
                root = reconstruction if method == "FEAST" else results / "spatialz"
                obs, _, xyz, counts = read_slice(root / "targets" / arms[gap][0] / name / "generated.h5ad", GENES)
                if method == "FEAST":
                    assert obs.index.equals(real[0].index)
                    np.testing.assert_array_equal(xyz, real[2])
            sampled = np.arange(len(obs)) if len(obs) <= 1200 else np.sort(
                np.random.default_rng(2026 + sid).choice(len(obs), 1200, replace=False))
            frame = pd.DataFrame(xyz[sampled], columns=["x", "y", "z"])
            frame[list(GENES)] = counts[sampled]
            points.append(frame.assign(method=method, gap=gap, target_slice=sid, spot_index=sampled,
                                       spot_id=obs.index[sampled], source_kind="real" if reference else "simulated"))
        if i % 36 == 0:
            print(f"Native marker samples: {i}/144 measured z locations", flush=True)
    frame = pd.concat(points, ignore_index=True)
    assert not frame.duplicated(["method", "gap", "target_slice", "spot_index"]).any()
    frame.to_parquet(output / "native_volume.parquet", index=False, compression="zstd")
    caps = {gene: float(np.quantile(np.log1p(frame.loc[frame[gene].gt(0), gene]), .99)) for gene in GENES}
    report = {"display_cap": caps, "sampled_positions_per_slice": 1200, "seed": "2026 + numeric section ID",
              "camera": {"elevation": 25, "azimuth": -55}, "render_coordinates": "native mm",
              "rows": len(frame), "counts_data": "layers/counts"}
    (output / "native_volume.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepare(args.zenodo, args.output or args.zenodo / "results" / STUDY / "figure05_sources")
