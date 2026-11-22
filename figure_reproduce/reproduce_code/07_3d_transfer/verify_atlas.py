"""Reconcile the male-only atlas volumes with the native DevCCF voxels and plotted counts."""
import argparse
import json
import os
from pathlib import Path

import anndata as ad
from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import workflow
from archive_io import csr_rows

STUDY = "07_3d_transfer"
AGES = workflow.AGE_ORDER
GENES = ("Slc17a7", "Gad2", "Reln", "Gfap")
S = 1 / np.sqrt(2)
PLANES = (
    ("XY", [1, 0, 0], [0, 1, 0]),
    ("XZ", [1, 0, 0], [0, 0, 1]),
    ("YZ", [0, 1, 0], [0, 0, 1]),
    ("Oblique XZ 45°", [S, 0, -S], [0, 1, 0]),
    ("Oblique YZ 45°", [1, 0, 0], [0, S, -S]),
)


def verify(zenodo, output):
    inputs = zenodo / "inputs" / STUDY
    root = zenodo / "results" / STUDY / "reconstruction"
    config = workflow.load_config(Path(__file__).with_name("config_male_feast106.yaml"))
    config["_dataset_root"] = inputs.resolve()
    config["_input_manifest"] = inputs / "input_files.csv"
    config["_reference_dir"] = config["_dataset_root"] / config["paths"]["reference_dir"]
    workflow.preflight_inputs(config)
    geometry = inputs / config["paths"]["devccf_dir"]
    schema = workflow.load_region_schema(geometry / config["blueprint"]["region_schema"])
    atlas_geometry = json.loads((inputs / "ccc/atlas_geometry.json").read_text())
    archived_validation = json.loads((root / "validation.json").read_text())
    samples, axial, planes, support, levels, age_records = [], [], [], [], [], []
    for age_index, age in enumerate(AGES):
        settings = config["ages"][age]
        volume, affine = workflow.read_nifti_gz(geometry / settings["volume"])
        np.testing.assert_array_equal(affine, atlas_geometry[age]["affine"])
        np.testing.assert_allclose(np.linalg.norm(affine[:3, :3], axis=0), .02, atol=1e-8)
        blueprint = workflow.extract_blueprint_payload(volume, affine, schema, age)
        entries = workflow.validate_blueprint_payload(config, age, blueprint)
        n_levels = len(entries)
        manifest = pd.read_csv(root / age / "manifest.csv").sort_values("z_index").reset_index(drop=True)
        assert manifest.z_index.tolist() == list(range(n_levels))
        assert {path.name for path in (root / age).glob("*.h5ad")} == set(manifest.filename)
        peak = int(manifest.n_spots.idxmax())
        volume_axes = [np.unique(np.concatenate([np.atleast_1d(entry[key]) for entry in entries]))
                       for key in ("x", "y", "z_world")]
        center = np.array([values[len(values) // 2] for values in volume_axes])
        plane_rows = []
        for name, horizontal, vertical in PLANES:
            normal = np.cross(horizontal, vertical)
            offset = .4 if name == "Oblique XZ 45°" else 0.
            plane_rows.append({"age": age, "plane": name, "normal": normal, "center": center + offset * normal,
                               "horizontal": horizontal, "vertical": vertical, "thickness": .02, "n_spots": 0})
        refs = workflow.reference_artifact_contract(config, age)
        for ref in workflow.reference_paths(config, age):
            with h5py.File(ref) as file:
                assert set(read_elem(file["obs/sex"])) == {"M"}
                assert set(read_elem(file["obs/treatment"])) == {"PBS"}
                assert np.isfinite(file["obsm/spatial"][:]).all()
                names = read_elem(file["var"]).index.tolist()
                assert len(names) == 550 and sum(name.startswith("Blank-") for name in names) == 50
                assert file["layers/counts"].attrs["shape"][1] == len(names)
        genes = sorted(names)
        markers = [genes.index(gene) for gene in GENES]
        transport = workflow.frozen_transport_config(config, settings["assignment_randomness"]["value"])
        provenance = json.loads((root / age / "provenance.json").read_text())
        assert provenance["feast"]["git_commit"] == config["required_feast_commit"]
        assert provenance["feast"]["version"] == config["feast_version"]
        assert provenance["reference_artifacts"] == refs
        n_transport, max_error, cursor = 0, 0., 0
        unified = root / "unified" / f"{age}_FEAST_3D.h5ad"
        with h5py.File(unified) as h:
            assert read_elem(h["var"]).index.tolist() == genes
            assert read_elem(h["uns/source_generation_provenance/feast/git_commit"]) == config["required_feast_commit"]
            unified_count = h["layers/counts"]
            unified_x = h["X"]
            labels = np.asarray(read_elem(h["obs/region"])).astype(str)
            index = h[f"obs/{h['obs'].attrs['_index']}"].asstr()
            for row, entry in zip(manifest.itertuples(index=False), entries, strict=True):
                path = root / age / row.filename
                data = ad.read_h5ad(path)
                summary = workflow.validate_result_identity(data, age, entry, genes, transport)
                workflow.validate_publication_lineage(data, config, age, entry, transport, refs)
                assert data.uns["publication_reproduction"]["feast_git_commit"] == config["required_feast_commit"]
                n_transport += summary["transport_records"]
                max_error = max(max_error, summary["max_final_error"])
                span = slice(cursor, cursor + row.n_spots)
                coords = data.obsm["spatial_3d"]
                region = data.obs["region"].astype(str).to_numpy()
                np.testing.assert_array_equal(index[span], data.obs_names)
                np.testing.assert_array_equal(labels[span], region)
                np.testing.assert_array_equal(h["obsm/spatial_3d"][span], coords)
                for dim in "ijk":
                    np.testing.assert_array_equal(h[f"obs/voxel_{dim}"][span], data.obs[f"voxel_{dim}"])
                counts = csr_matrix(data.layers["counts"])
                for matrix in (unified_count, unified_x):
                    assert (csr_rows(matrix, span.start, span.stop) != counts).nnz == 0
                for label, size in pd.Series(region).value_counts().items():
                    support.append({"age": age, "z_index": row.z_index, "z_world": row.z_world,
                                    "region": label, "n_spots": size})
                points = pd.DataFrame(coords, columns=["x", "y", "z"])
                points = points.assign(age=age, z_index=row.z_index, spot_index=np.arange(row.n_spots),
                                       spot_id=data.obs_names, region=region)
                points[list(GENES)] = data.layers["counts"][:, markers]
                seed = 2026 + 100000 * age_index + row.z_index
                sample = np.arange(row.n_spots) if row.n_spots <= 700 else np.sort(
                    np.random.default_rng(seed).choice(row.n_spots, 700, replace=False))
                samples.append(points.iloc[sample])
                if row.z_index == peak:
                    axial.append(points)
                for plane in plane_rows:
                    delta = coords - plane["center"]
                    take = np.abs(delta @ plane["normal"]) < .01 + 1e-8
                    if take.any():
                        pos = delta[take]
                        planes.append(points.loc[take].assign(plane=plane["plane"],
                                                               u=pos @ plane["horizontal"], v=pos @ plane["vertical"]))
                        plane["n_spots"] += int(take.sum())
                levels.append({"age": age, "z_index": row.z_index, "z_world": row.z_world,
                               "n_spots": row.n_spots, "peak": row.z_index == peak, "sample_seed": seed,
                               "file": os.path.relpath(path, zenodo)})
                cursor = span.stop
            assert cursor == len(labels) == unified_count.attrs["shape"][0] == settings["expected_spots"]
        recorded = next(record for record in archived_validation["ages"] if record["age"] == age)
        assert n_transport == recorded["transport_records"] and max_error == recorded["max_final_error"]
        age_records.append({"age": age, "n_z_levels": n_levels, "n_spots": cursor, "transport_records": n_transport,
                            "max_final_error": max_error, "planes": plane_rows})
        print(f"{age}: {n_levels} levels and {cursor:,} voxels checked against volume, unified H5AD and manifests", flush=True)

    region_data = pd.DataFrame(support).sort_values(["age", "z_index", "region"]).reset_index(drop=True)
    old_support = pd.read_csv(root / "evaluation/region_coverage.csv").sort_values(
        ["age", "z_index", "region"]).reset_index(drop=True)[region_data.columns]
    pd.testing.assert_frame_equal(region_data, old_support)
    output.mkdir(parents=True, exist_ok=True)
    region_data.to_csv(output / "region_coverage.csv", index=False)
    pd.DataFrame(levels).to_csv(output / "levels.csv", index=False)
    limits = {}
    views = {"volume": pd.concat(samples), "axial": pd.concat(axial), "planes": pd.concat(planes)}
    for view, frame in views.items():
        limits[view] = {}
        for gene in GENES:
            values = np.log1p(frame[gene].to_numpy(dtype=np.float32 if view == "planes" else float))
            limits[view][gene] = float(np.quantile(values[values > 0], .99))
        frame.to_parquet(output / f"{view}.parquet", index=False, compression="zstd")
    report = {"feast_commit": config["required_feast_commit"], "ages": age_records, "display_caps": limits,
              "sampled_voxels": len(views["volume"]), "axial_voxels": len(views["axial"]),
              "multiplane_voxels": len(views["planes"]), "units": "mm"}
    (output / "provenance.json").write_text(json.dumps(report, indent=2, default=lambda value: value.tolist()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.zenodo, args.output or args.zenodo / "results" / STUDY / "figure06_sources")
