"""Verify Study 06 archives and rebuild the evidence for Figure 5c–e."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from evaluate_five_metrics import prepare_regions, score_regions
from evaluate_resampling_metrics import log_normalize
from spatialz_evaluate import prepare_observed, evaluate_slice, slice_arrays
from validate import (all_target_rows, validate_count_matrix, validate_plan_contract,
                      validate_provenance, validate_target_directory_set,
                      validate_transport_diagnostics)
from workflow import load_config

STUDY = "06_3d_stack"
KEYS = ["gap", "target_slice", "method"]
METRICS = ["conditional_expression_wasserstein", "conditional_zero_fraction_wasserstein",
           "moran_profile_correlation", "residual_moran_profile_correlation"]
COMMIT = "a79cf222f05d6ad314734ce05f7dd53110d7640f"


def read_json(path):
    return json.loads(path.read_text())


def check_coordinates(data, z):
    assert data.obs_names.is_unique and data.obs["class"].notna().all()
    xy, xyz = data.obsm["spatial"], data.obsm["spatial_3d"]
    np.testing.assert_allclose(xy, xyz[:, :2], rtol=0, atol=1e-8)
    np.testing.assert_allclose(xyz[:, 2], z, rtol=0, atol=1e-7)
    np.testing.assert_allclose(data.obs["z"], z, rtol=0, atol=1e-7)
    return xy


def verify_slice(task):
    source, rows, inputs, results, config, genes, builds = task
    dataset = config["dataset"]
    path = inputs / "h5ad" / source["filename"]
    measured = ad.read_h5ad(path)
    assert measured.shape == (source["n_obs"], source["n_vars"])
    assert measured.var_names.tolist() == genes
    xy = check_coordinates(measured, source["z"])
    validate_count_matrix(measured.layers["counts"], measured.shape)
    metrics, classes, regions, lineage = [], [], [], []
    if rows:
        observed = prepare_observed(measured, dataset)
    for row in rows:
        density, gap, sid = row["density_name"], row["gap"], row["target_slice"]
        suffix = Path("targets") / density / path.stem
        fp = results / "reconstruction" / suffix / "provenance.json"
        zp = results / "spatialz" / suffix / "run.json"
        feast, spatialz = read_json(fp), read_json(zp)
        validate_provenance(feast, row, config, {}, fp.with_name("generated.h5ad"))
        assert feast["feast_commit"] == feast["runtime"]["commit"] == COMMIT
        build = "precision" if "1.0.6-precision" in feast["runtime"]["import_path"] else "original"
        assert build == builds[(gap, sid)]
        assert spatialz["status"] == "completed" and not spatialz["target_h5ad_opened"]
        assert spatialz["target"] == {key: row[key] for key in spatialz["target"]}
        assert spatialz["source_commit"].startswith("e1b5ed0")
        assert spatialz["recipe"]["coordinates"] == {
            "input_to_working_scale": 1000.0, "input_unit": "mm", "working_unit": "um",
        }
        assert spatialz["n_generated_cells"] == int(
            spatialz["alpha"] * spatialz["lower_reference_cells"]
            + (1 - spatialz["alpha"]) * spatialz["upper_reference_cells"]
        )
        geometry = [{"x": observed["normalized"], "xy": xy, "labels": observed["labels"]}]
        for method, record_path in (("feast", fp), ("spatialz", zp)):
            data = ad.read_h5ad(record_path.with_name("generated.h5ad"))
            assert data.var_names.equals(measured.var_names)
            candidate_xy = check_coordinates(data, source["z"])
            if method == "feast":
                np.testing.assert_array_equal(data.obs_names, measured.obs_names)
                np.testing.assert_array_equal(candidate_xy, xy)
                np.testing.assert_array_equal(data.obs["class"].astype(str), observed["labels"])
                np.testing.assert_array_equal(data.obsm["spatial_3d"], measured.obsm["spatial_3d"])
                assert data.uns["de_novo"]["marginal_model"] == "core"
                local = json.loads(data.uns["local_generation_json"])
                assert set(local["primary_references"]) == set(row["reference_weights"])
                refs = set(row["reference_weights"]) | {item["artifact_id"] for item in row["donor_support"]}
                transport, error = validate_transport_diagnostics(
                    data.uns["de_novo"]["transport_diagnostics"], feast["transport"], refs,
                )
                assert transport == feast["solver_diagnostics"]["transport_records"]
            else:
                assert data.n_obs == spatialz["n_generated_cells"]
            values = validate_count_matrix(data.layers["counts"], data.shape)
            evaluated, support, _ = evaluate_slice(observed, data, dataset, genes)
            key = {"gap": gap, "target_slice": sid, "method": method}
            metrics.append({**key, **evaluated})
            classes.extend(support.assign(**key).to_dict("records"))
            geometry.append({"x": log_normalize(values)[0], "xy": candidate_xy,
                             "labels": data.obs["class"].astype(str).to_numpy()})
        specs = prepare_regions(*geometry)
        for index, method in enumerate(("feast", "spatialz"), start=1):
            regions.extend({"gap": gap, "target_slice": sid, "method": method, **record}
                           for record in score_regions(geometry[index]["x"], specs, index))
        lineage.append({"gap": gap, "target_slice": sid, "feast_build": build,
                        "seed": row["seed"], "transport_records": transport, "maximum_final_error": error,
                        "measured": str(path), "feast_provenance": str(fp), "spatialz_provenance": str(zp)})
    return metrics, classes, regions, lineage


def compare(frame, path, keys, columns):
    archived = pd.read_csv(path)
    expected = archived.loc[archived.method.isin(frame.method.unique())].set_index(keys).sort_index()
    actual = frame.set_index(keys).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    for field in columns:
        if pd.api.types.is_numeric_dtype(actual[field]):
            np.testing.assert_allclose(actual[field], expected[field], rtol=1e-10, atol=1e-10,
                                       err_msg=f"{path.name}: {field}")
        else:
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)


def verify(zenodo, output, workers):
    inputs = zenodo / "inputs" / STUDY
    results = zenodo / "results" / STUDY
    reconstruction = results / "reconstruction"
    config = load_config(reconstruction / "frozen_config.yaml")
    plan = read_json(reconstruction / "plan.json")
    plan["data_dir"] = str(inputs / "h5ad")
    manifest = pd.read_csv(reconstruction / "input_manifest.csv")
    sources = manifest.set_index("slice_id", drop=False).to_dict("index")
    validate_plan_contract(plan, config, sources)
    assert len(sources) == 144 and sum(manifest.n_obs) == 2826804
    rows = all_target_rows(plan)
    assert len(rows) == 336
    validate_target_directory_set(reconstruction, rows, "Zhuang-ABCA-1")
    validate_target_directory_set(results / "spatialz", rows, "Zhuang-ABCA-1")
    builds = pd.read_csv(inputs / "article_settings/target_builds.csv")
    assert builds.feast_build.value_counts().to_dict() == {"precision": 252, "original": 84}
    build_map = builds.set_index(["gap", "target_slice"], verify_integrity=True).feast_build.to_dict()
    tasks = [(source, [row for row in rows if row["target_slice"] == sid],
              inputs, results, config, plan["gene_names"], build_map) for sid, source in sources.items()]
    collected = [[], [], [], []]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, group in enumerate(pool.map(verify_slice, tasks), start=1):
            for field, records in zip(collected, group):
                field.extend(records)
            if i % 24 == 0:
                print(f"Study 06: {i}/144 measured slices verified", flush=True)
    metrics, classes, regions, lineage = [pd.DataFrame(records).sort_values(KEYS if "method" in records[0]
                                                                          else KEYS[:2]) for records in collected]
    archived = results / "spatialz/evaluation"
    compare(metrics, archived / "metrics.csv", KEYS, metrics.columns.difference(KEYS))
    compare(classes, archived / "class_metrics.csv", KEYS + ["class"],
            classes.columns.difference(KEYS + ["class"]))
    region_keys = KEYS + ["bin_width_mm"]
    compare(regions, results / "benchmark/regional_target_metrics.csv", region_keys,
            regions.columns.difference(region_keys))
    common = set.intersection(*(set(arm["target_pool"]) for arm in plan["densities"].values()))
    assert len(common) == 76
    expected_common = pd.read_csv(results / "benchmark/regional_target_metrics.csv")
    for _, group in expected_common.loc[expected_common.common_target].groupby(["gap", "method", "bin_width_mm"]):
        assert set(group.target_slice) == common
    regions["common_target"] = regions.target_slice.isin(common)
    draws = pd.read_csv(reconstruction / "baselines/five_reference_conditional_resampling/method_metrics.csv")
    reference = draws.loc[draws.method.eq("feast")].set_index(KEYS).sort_index()
    raw_feast = metrics.loc[metrics.method.eq("feast")].set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(reference.index, raw_feast.index)
    np.testing.assert_allclose(reference[METRICS], raw_feast[METRICS], rtol=1e-9, atol=1e-10)
    draws = draws.loc[draws.method.eq("conditional_whole_spot_resampling")]
    assert len(draws) == 3360 and draws.seed.eq(draws.replicate + 2026).all()
    keyed = draws.set_index(KEYS[:2] + ["replicate"], verify_integrity=True).sort_index()
    expected_keys = sorted((row["gap"], row["target_slice"], draw) for row in rows for draw in range(10))
    assert keyed.index.tolist() == expected_keys
    means = keyed.groupby(KEYS[:2])[METRICS].mean().reset_index().assign(method="resampling")
    comparison = pd.concat([metrics[KEYS + METRICS], means], ignore_index=True).sort_values(KEYS)
    matched_rows = []
    for metric in METRICS:
        paired = comparison.pivot(index=KEYS[:2], columns="method", values=metric).dropna()
        for gap, group in paired.groupby(level="gap"):
            matched_rows.extend({"gap": gap, "target_slice": sid, "metric": metric}
                                for _, sid in group.index)
    matched = pd.DataFrame(matched_rows)
    assert matched.groupby(["metric", "gap"]).size().unstack().to_dict("index") == {
        **{metric: {3: 14, 5: 11, 10: 18} for metric in METRICS[:2]},
        **{metric: {3: 95, 5: 114, 10: 127} for metric in METRICS[2:]},
    }
    stacks = []
    for density, arm in plan["densities"].items():
        for order, sid in enumerate(plan["available_slices"]):
            source = sources[sid]
            measured = sid in arm["reference_pool"]
            path = (inputs / "h5ad" / source["filename"] if measured else
                    reconstruction / "targets" / density / Path(source["filename"]).stem / "generated.h5ad")
            stacks.append({"gap": arm["spec"]["gap"], "slice_id": sid, "stack_index": order,
                           "z": source["z"], "source_kind": "real" if measured else "simulated",
                           "path": os.path.relpath(path, zenodo)})
    output.mkdir(parents=True, exist_ok=True)
    for column in ["measured", "feast_provenance", "spatialz_provenance"]:
        lineage[column] = lineage[column].map(lambda path: os.path.relpath(path, zenodo))
    pd.DataFrame(stacks).to_csv(output / "coverage.csv", index=False)
    for name, table in [("target_lineage", lineage), ("native_metrics", metrics), ("class_support", classes),
                        ("regional_metrics", regions), ("panel_d_metrics", comparison), ("panel_d_support", matched)]:
        table.to_csv(output / f"{name}.csv", index=False)
    report = {"status": "passed", "measured_slices": len(sources), "matched_regional_positions": sorted(common),
              "feast_targets": len(builds), "spatialz_targets": len(rows), "archived_resampling_draws": len(draws),
              "archived_builds": builds.feast_build.value_counts().to_dict(),
              "regional_median_at_0_2_mm": regions.loc[regions.common_target & regions.bin_width_mm.eq(.2)]
              .groupby(["method", "gap"]).region_mae.median().unstack().to_dict("index")}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    verify(args.zenodo, args.output or args.zenodo / "results" / STUDY / "figure05_sources", args.workers)
