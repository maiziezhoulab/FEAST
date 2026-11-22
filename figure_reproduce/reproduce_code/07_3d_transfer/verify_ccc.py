"""Verify the completed ComFlow files and stream all 340 candidate CSR matrices."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd

from ccc_current_io import REGIONS, run_info
from prepare_ccc_redistribution import regional_sum
from archive_io import csr_rows

STUDY = "07_3d_transfer"


def compare_source(h, input_path, info, chunk_rows):
    spec = info["input"]
    actual = pd.Categorical(read_elem(h["obs/region"]), categories=REGIONS)
    assert len(actual) == info["n_positions"] and (actual.codes >= 0).all()
    with h5py.File(input_path) as source:
        assert read_elem(h["var"]).index.equals(read_elem(source["var"]).index)
        assert len(h["var"][h["var"].attrs["_index"]]) == 550
        labels = read_elem(source[f"obs/{spec['region_key']}"])
        keep = np.flatnonzero(labels != spec["exclude_region"]) if spec["exclude_region"] else None
        explained = pd.Series(labels).value_counts().to_dict()
        assert {key: int(val) for key, val in explained.items() if val} == info["preflight"]["region_sizes_before_exclusion"]
        source_count = source["layers/counts"]
        inference_count = h["layers/counts"]
        n_source = source_count.attrs["shape"][0]
        assert n_source == info["preflight"]["n_input"]
        assert n_source - len(actual) == info["preflight"]["n_excluded_Other"]
        source_ids = source[f"obs/{source['obs'].attrs['_index']}"].asstr()
        matrix_ids = h[f"obs/{h['obs'].attrs['_index']}"].asstr()
        xy = source[f"obsm/{spec['spatial_key']}"]
        if keep is not None:
            source_ids, xy = source_ids[:], xy[:]
            measured_counts = csr_rows(source_count, 0, n_source)
        for lo in range(0, len(actual), chunk_rows):
            hi = min(len(actual), lo + chunk_rows)
            rows = keep[lo:hi] if keep is not None else slice(lo, hi)
            np.testing.assert_array_equal(matrix_ids[lo:hi], source_ids[rows])
            np.testing.assert_array_equal(actual[lo:hi].astype(str), np.asarray(labels[rows]).astype(str))
            coordinates = xy[rows]
            np.testing.assert_array_equal(h["obsm/spatial_input"][lo:hi], coordinates)
            np.testing.assert_array_equal(h["obsm/spatial"][lo:hi],
                                          coordinates.astype(float) * spec["coordinate_scale_to_mm"])
            counts = measured_counts[rows] if keep is not None else csr_rows(source_count, lo, hi)
            assert (csr_rows(inference_count, lo, hi) != counts).nnz == 0
    return actual.codes.astype(np.int16), pd.Series(actual).value_counts().to_dict()


def verify_dataset(task):
    dataset, input_path, root, output, archive, pairs, chunk_rows = task
    folder = root / dataset
    path = folder / f"{dataset}_adata.h5ad"
    saved = pd.read_csv(folder / "LR_region_edges.csv").set_index(
        ["ligand", "receptor", "sender_region", "receiver_region"], verify_integrity=True)
    assert len(saved) == 85 * 8 * 8
    status = pd.read_csv(folder / "all_LR_status.csv").set_index("Key", verify_integrity=True)
    assert set(status.index) == {f"{ligand}--{receptor}" for ligand, receptor in pairs}
    regions = pd.MultiIndex.from_product([REGIONS, REGIONS], names=["sender_region", "receiver_region"])
    edges, checks = [], []
    with h5py.File(path) as h:
        info = run_info(h)
        recorded = json.loads((folder / "run_settings.json").read_text())
        # The sidecar includes time spent writing the final H5AD.
        assert recorded.pop("runtime_seconds") >= info["runtime_seconds"]
        assert recorded == {key: value for key, value in info.items() if key != "runtime_seconds"}
        assert info["input"] == archive["datasets"][dataset]
        assert info["dataset"] == dataset and info["seed"] == archive["seed"] == 0
        assert info["cluster_permutations"] == 100 and info["n_features"] == 550
        assert info["normalization_input"] == "layers/counts" and info["preflight"]["n_blank_features"] == 50
        assert info["settings"] == {key: archive[key] for key in info["settings"]}
        codes, sizes = compare_source(h, input_path, info, chunk_rows)
        for ligand, receptor in pairs:
            key = f"{ligand}--{receptor}"
            weights, counts, _, support = regional_sum(h[f"obsp/commot-User_db-{ligand}-{receptor}"], codes, chunk_rows)
            expected = saved.loc[(ligand, receptor)].loc[regions]
            np.testing.assert_array_equal(counts.ravel(), expected.n_edges)
            np.testing.assert_allclose(weights.ravel(), expected.W, rtol=1e-11, atol=1e-9)
            record = status.loc[key]
            active = int(support["total_nonself_mass"] + support["self_mass_excluded"] > 0)
            assert active == record.Status and counts.sum() == record.n_nonself_edges
            np.testing.assert_allclose([support["total_nonself_mass"], support["self_mass_excluded"]],
                                       [record.nonself_mass, record.self_mass], rtol=1e-11, atol=1e-9)
            edges.append(regions.to_frame(index=False).assign(ligand=ligand, receptor=receptor,
                                                             W=weights.ravel(), n_edges=counts.ravel()))
            checks.append({"Key": key, "Status": active, **support,
                           "maximum_region_mass_error": float(np.max(np.abs(weights.ravel() - expected.W)))})
    target = output / dataset
    target.mkdir(parents=True, exist_ok=True)
    pd.concat(edges, ignore_index=True).to_csv(target / "regional_edges.csv", index=False)
    table = pd.DataFrame(checks)
    table.to_csv(target / "sparse_support.csv", index=False)
    print(f"{dataset}: verified all {len(table)} LR matrices across {len(codes):,} positions", flush=True)
    return {"dataset": dataset, "positions": len(codes), "region_counts": sizes, "lr_matrices": len(table),
            "max_mass_difference": table.maximum_region_mass_error.max(), "bytes": path.stat().st_size,
            "input": os.path.relpath(input_path, root.parents[2]), "h5ad": str(path.relative_to(root)),
            "run_settings": str((folder / "run_settings.json").relative_to(root))}


def verify(zenodo, output, workers):
    inputs = zenodo / "inputs" / STUDY
    root = zenodo / "results" / STUDY / "ccc"
    launch = json.loads((root / "launch.json").read_text())
    archive = json.loads((root / "archive_config.json").read_text())
    assert launch["status"] == "complete" and launch["exit_code"] == 0
    assert launch["dataset_order"] == list(archive["datasets"])
    pairs = list(pd.read_csv(inputs / "ccc/shared_85_LR.csv")[["ligand", "receptor"]].itertuples(index=False, name=None))
    assert len(set(pairs)) == len(pairs) == 85
    tasks = []
    for name in launch["dataset_order"]:
        input_path = (root.parent / "reconstruction/unified" / f"{name[:-3]}_FEAST_3D.h5ad" if name.endswith("_3D") else
                      inputs / "GSE269617/h5ad_region_annotated" / f"{name[:-3]}.h5ad")
        tasks.append((name, input_path, root, output, archive, pairs, 250000))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(verify_dataset, tasks))
    report = {"status": "passed", "launch": str((root / "launch.json").relative_to(zenodo)),
              "matrix_count": sum(row["lr_matrices"] for row in records), "datasets": records}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    verify(args.zenodo, args.output or args.zenodo / "results" / STUDY / "ccc_verified", args.workers)
