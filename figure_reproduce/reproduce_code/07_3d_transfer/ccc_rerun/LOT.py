#!/usr/bin/env python3
"""Local FEAST-only adaptation of upstream LOT.py; see upstream/ for original code.

check: inspect four inputs and dependencies without inference.
smoke: run the complete pipeline on 128 real positions per input (not research data).
run: infer the complete selected input(s), sequentially, in new output directories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))

import anndata as ad
from anndata.experimental import read_elem
import commot as ct
import h5py
import numpy as np
import pandas as pd
import pyreadr
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from comflow.tools import _spatial_communication

REGIONS = ["DorsalVZ", "IZ", "CP", "VentralVZ", "GE", "SeptalVZ", "Septum", "BT"]


def read_head(h, spec, limit=4096):
    """Read a CSR row prefix for smoke tests, without loading a full 3D atlas."""
    group = h["layers/counts"]
    assert group.attrs["encoding-type"] == "csr_matrix"
    n = min(limit, int(group.attrs["shape"][0]))
    ptr = group["indptr"][:n + 1]
    end = int(ptr[-1])
    x = sparse.csr_matrix((group["data"][:end], group["indices"][:end], ptr),
                          shape=(n, int(group.attrs["shape"][1])))
    region = h["obs"][spec["region_key"]]
    categories = region["categories"].asstr()[:]
    codes = region["codes"][:n]
    index_key = h["obs"].attrs["_index"]
    ids = h["obs"][index_key].asstr()[:n]
    obs = pd.DataFrame({"region": pd.Categorical.from_codes(codes, categories),
                        "source_row": np.arange(n)}, index=pd.Index(ids, name=index_key))
    result = ad.AnnData(x, obs=obs, var=read_elem(h["var"]))
    result.layers["counts"] = x.copy()
    result.obsm["spatial"] = h["obsm"][spec["spatial_key"]][:n]
    return result


def preflight(config, lr, names):
    needed = set(g for pair in lr[["ligand", "receptor"]].to_numpy()
                 for complex_name in pair for g in complex_name.split("_"))
    assert list(lr.columns) == ["ligand", "receptor", "pathway"] and len(lr) == 85
    assert not lr.duplicated(["ligand", "receptor"]).any()
    records = {}
    for name in names:
        spec = config["datasets"][name]
        with h5py.File(spec["input"], "r") as h:
            genes = read_elem(h["var"]).index
            assert genes.is_unique and needed.issubset(set(genes)), (name, needed - set(genes))
            region = read_elem(h["obs"][spec["region_key"]])
            sizes = pd.Series(region).value_counts().to_dict()
            observed = set(str(v) for v in pd.unique(region))
            allowed = set(REGIONS) | ({spec["exclude_region"]} if spec["exclude_region"] else set())
            assert observed.issubset(allowed) and not pd.isna(region).any(), (name, observed)
            head = read_head(h, spec)
            coords = head.obsm["spatial"] * spec["coordinate_scale_to_mm"]
            assert coords.shape[1] == spec["dimensions"] and np.isfinite(coords).all()
            assert (head.X.data >= 0).all() and np.isfinite(head.X.data).all()
            source = read_elem(h["uns/source_generation_provenance"]) if spec["dimensions"] == 3 else {}
            if spec["dimensions"] == 3:
                assert source["configuration_id"] == "study07-two-reference-float32-male-feast106-v1"
                assert source["feast"]["version"] == "1.0.6"
                assert all("-m" in ref or "e14m" in ref or "e18m" in ref
                           for ref in source["reference_artifacts"])
            nearest = cKDTree(coords).query(coords, k=2)[0][:, 1]
            n = sum(int(v) for v in sizes.values())
            excluded = int(sizes.get(spec["exclude_region"], 0))
            records[name] = {"input": spec["input"], "n_input": n, "n_excluded_Other": excluded,
                             "n_analyzed": n - excluded, "n_features": len(genes),
                             "n_blank_features": int(genes.str.startswith("Blank-").sum()),
                             "region_sizes_before_exclusion": {str(k): int(v) for k, v in sizes.items()},
                             "coordinate_scale_to_mm": spec["coordinate_scale_to_mm"],
                             "coordinate_dimensions": coords.shape[1],
                             "head_nearest_neighbor_median_mm": float(np.median(nearest)),
                             "all_85_LR_components_present": True,
                             "source_configuration_id": source.get("configuration_id", "observed MERFISH"),
                             "feast_version": source.get("feast", {}).get("version", "not applicable")}
        print(f"CHECK {name}: {n - excluded:,} positions, {len(genes)} features, 85 fixed LR pairs", flush=True)
    return records


def load_input(spec, smoke):
    if smoke:
        with h5py.File(spec["input"], "r") as h:
            a = read_head(h, spec)
    else:
        a = ad.read_h5ad(spec["input"])
        a.obsm["spatial"] = np.asarray(a.obsm[spec["spatial_key"]]).copy()
    if spec["exclude_region"]:
        a = a[a.obs[spec["region_key"]] != spec["exclude_region"]].copy()
    a.obs["region"] = a.obs[spec["region_key"]].astype("category")
    if smoke:
        # A deterministic local patch is a software test, not a representative tissue sample.
        marker = np.asarray(a[:, ["Dll1", "Notch1"]].X.sum(axis=1)).ravel()
        anchor = int(np.argmax(marker))
        coords = a.obsm["spatial"]
        order = np.argsort(np.sum((coords - coords[anchor]) ** 2, axis=1), kind="stable")[:128]
        a = a[np.sort(order)].copy()
    a.obsm["spatial_input"] = np.asarray(a.obsm["spatial"]).copy()
    a.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float) * spec["coordinate_scale_to_mm"]
    a.X = a.layers["counts"].copy()
    assert a.obs_names.is_unique and a.var_names.is_unique
    return a


def regional_edges(a, lr, database_name):
    codes = pd.Categorical(a.obs.region, categories=REGIONS).codes.astype(np.int16)
    assert (codes >= 0).all()
    rows, statuses = [], []
    for ligand, receptor in lr[["ligand", "receptor"]].itertuples(index=False, name=None):
        matrix = a.obsp[f"commot-{database_name}-{ligand}-{receptor}"].tocsr()
        assert np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
        w = np.zeros((8, 8)); counts = np.zeros((8, 8), dtype=np.int64)
        for lo in range(0, a.n_obs, 250000):
            hi = min(a.n_obs, lo + 250000)
            ptr = matrix.indptr[lo:hi + 1]
            rr = np.repeat(np.arange(lo, hi), np.diff(ptr))
            cc = matrix.indices[ptr[0]:ptr[-1]]; values = matrix.data[ptr[0]:ptr[-1]]
            keep = (rr != cc) & (values > 0)
            keys = 8 * codes[rr[keep]] + codes[cc[keep]]
            w += np.bincount(keys, weights=values[keep], minlength=64).reshape(8, 8)
            counts += np.bincount(keys, minlength=64).reshape(8, 8)
        statuses.append({"Key": f"{ligand}--{receptor}", "Status": int(np.any(matrix.data > 0)),
                         "n_nonself_edges": int(counts.sum()), "nonself_mass": float(w.sum()),
                         "self_mass": float(matrix.diagonal().sum())})
        for i, sender in enumerate(REGIONS):
            for j, receiver in enumerate(REGIONS):
                rows.append({"ligand": ligand, "receptor": receptor, "sender_region": sender,
                             "receiver_region": receiver, "W": w[i, j], "n_edges": counts[i, j]})
    return pd.DataFrame(rows), pd.DataFrame(statuses)


def write_adata(a, path, info, database_name):
    """Write large marginals as arrays, restoring in-memory tables afterward."""
    tables = {}
    for role in ["sender", "receiver"]:
        key = f"commot-{database_name}-sum-{role}"
        table = a.obsm[key]
        tables[key] = table
        a.uns[key + "_columns"] = table.columns.astype(str).tolist()
        a.uns[key + "_index"] = table.index.astype(str).tolist()
        a.obsm[key] = table.to_numpy()
    a.uns["ccc_rerun_provenance_json"] = json.dumps(info)
    a.write_h5ad(path)
    for key, table in tables.items():
        a.obsm[key] = table


def infer_one(name, config, lr, output_root, smoke, preflight_record):
    started = time.monotonic()
    spec = config["datasets"][name]
    output = output_root / name
    output.mkdir(parents=True, exist_ok=False)
    a = load_input(spec, smoke)
    a.raw = a
    sc.pp.normalize_total(a, target_sum=config["normalization_target_sum"], inplace=True)
    sc.pp.log1p(a)
    checked = a.X[:min(a.n_obs, 4096)].copy()
    checked.data = np.expm1(checked.data)
    totals = np.asarray(checked.sum(axis=1)).ravel()
    np.testing.assert_allclose(totals[totals > 0], config["normalization_target_sum"], rtol=1e-6, atol=2e-5)
    permutations = 2 if smoke else config["cluster_permutations"]
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "entry_point": str(Path(__file__).resolve()),
            "dataset": name, "smoke_test_only": smoke, "n_positions": a.n_obs, "input": spec,
            "normalization_target_sum": config["normalization_target_sum"], "normalization_input": "layers/counts",
            "n_features": a.n_vars, "retain_blank_controls": True, "candidate_LR_count": len(lr),
            "candidate_filter": "fixed 85; no per-input prevalence filtering", "cluster_permutations": permutations,
            "pvalue_filter_for_network": "none", "seed": config["seed"], "preflight": preflight_record,
            "model": "custom ComFlow with receptor-TF scores, not stock COMMOT spatial_communication",
            "comflow_source": str(Path(_spatial_communication.__file__).resolve()),
            "settings": {k: config[k] for k in ["distance_threshold_mm", "co_exp_ratio", "heteromeric", "heteromeric_rule", "heteromeric_delimiter", "species", "database_name", "pathway_sum", "solver"]},
            "self_policy": "retain same-position terms in inference; exclude i=j from regional W and network metrics",
            "output_stage": "inference_running"}
    (output / "run_settings.json").write_text(json.dumps(info, indent=2) + "\n")
    pathways = pyreadr.read_r(str(ROOT / config["pathways"]))["pathways"]
    print(f"INFER {name}: {a.n_obs:,} positions; {a.obsm['spatial'].shape[1]}D; 85 LR; 0.035 mm", flush=True)
    a, model = _spatial_communication.spatial_communication(
        a, config["species"], database_name=config["database_name"], df_ligrec=lr,
        pathways=pathways, co_exp_ratio=config["co_exp_ratio"], dis_thr=config["distance_threshold_mm"],
        heteromeric=config["heteromeric"], heteromeric_rule=config["heteromeric_rule"],
        heteromeric_delimiter=config["heteromeric_delimiter"], pathway_sum=config["pathway_sum"],
        copy=True, **config["solver"])
    actual = a.uns[f"commot-{config['database_name']}-info"]["df_ligrec"]
    assert set(map(tuple, actual[["ligand", "receptor"]].to_numpy())) == set(map(tuple, lr[["ligand", "receptor"]].to_numpy()))
    regional, status = regional_edges(a, lr, config["database_name"])
    # Only 85 x 8 x 8 rows; CSV keeps the original environment dependency-free.
    regional.to_csv(output / "LR_region_edges.csv", index=False)
    status.to_csv(output / "all_LR_status.csv", index=False)
    grouped = regional.groupby(["sender_region", "receiver_region"], sort=False).W.sum()
    w = grouped.unstack().loc[REGIONS, REGIONS].to_numpy()
    total = float(w.sum()); inter = float(w[~np.eye(8, dtype=bool)].sum())
    # T=0 is reported as undefined, never replaced with an artificial distribution.
    network = {"total_nonself_mass": total, "F_inter": inter / total if total > 0 else None,
               "F_intra": float(np.trace(w)) / total if total > 0 else None,
               "n_nonzero_regional_edges_including_loops": int(np.count_nonzero(w)),
               "n_nonzero_cross_region_edges": int(np.count_nonzero(w[~np.eye(8, dtype=bool)])),
               "n_positive_nonself_LR_edges": int(status.n_nonself_edges.sum()),
               "n_LR_candidates": len(lr), "n_LR_nonzero_including_self": int(status.Status.sum())}
    print(f"REGIONAL {name}: {network}", flush=True)
    (output / "network_summary.json").write_text(json.dumps(network, indent=2) + "\n")

    # The previous execution stopped during permutations before any matrices were saved.
    # Preserve the computed matrices and model first; this changes only output order.
    info["output_stage"] = "inference_complete_before_permutations"
    info["runtime_seconds"] = time.monotonic() - started
    print(f"SAVE_INFERENCE {name}", flush=True)
    write_adata(a, output / f"{name}_inference.h5ad", info, config["database_name"])
    pd.DataFrame(model.Q, columns=model.recs).to_pickle(output / "model_Q.pkl")
    for filename, value in [("rec_tf_probability.pkl", model.receptor_tf_tables), ("rec_tf_score.pkl", model.rec_tf_score_tables)]:
        with (output / filename).open("wb") as stream:
            pickle.dump(value, stream)
    sparse.save_npz(output / "model_C.npz", model.C)
    sparse.save_npz(output / "model_a.npz", sparse.csr_matrix(model.S))
    sparse.save_npz(output / "model_b.npz", sparse.csr_matrix(model.D))
    info["output_stage"] = "permutations_running"
    (output / "run_settings.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"PERMUTATIONS {name}: total network, {permutations} permutations", flush=True)
    csv_rows = []
    ct.tl.cluster_communication(a, database_name=config["database_name"], clustering="region", n_permutations=permutations, random_seed=config["seed"])
    print(f"PERMUTATIONS {name}: total network complete", flush=True)
    for number, (ligand, receptor) in enumerate(lr[["ligand", "receptor"]].itertuples(index=False, name=None), 1):
        print(f"PERMUTATIONS {name}: LR {number}/{len(lr)} {ligand}--{receptor} start", flush=True)
        ct.tl.cluster_communication(a, database_name=config["database_name"], lr_pair=[ligand, receptor], clustering="region", n_permutations=permutations, random_seed=config["seed"])
        record = a.uns[f"commot_cluster-region-{config['database_name']}-{ligand}-{receptor}"]
        score, pvalue = record["communication_matrix"], record["communication_pvalue"]
        for sender in score.index:
            for receiver in score.columns:
                csv_rows.append({"ligand": ligand, "receptor": receptor, "sender_region": sender,
                                 "receiver_region": receiver, "stored_mean_score": score.loc[sender, receiver],
                                 "within_input_permutation_pvalue": pvalue.loc[sender, receiver]})
        print(f"PERMUTATIONS {name}: LR {number}/{len(lr)} complete", flush=True)
    pd.DataFrame(csv_rows).to_csv(output / "region_means_and_within_input_pvalues.csv", index=False)
    info["output_stage"] = "complete"
    info["runtime_seconds"] = time.monotonic() - started
    print(f"SAVE_FINAL {name}", flush=True)
    write_adata(a, output / f"{name}_adata.h5ad", info, config["database_name"])
    info["runtime_seconds"] = time.monotonic() - started
    (output / "run_settings.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"COMPLETE {name}: {output}; {info['runtime_seconds']:.1f} seconds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--dataset", default="all", help="Config dataset key, or all (sequential)")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base = args.config.resolve().parent
    config["output_root"] = str(base / config["output_root"])
    for spec in config["datasets"].values():
        spec["input"] = str(base / spec["input"])
    lr = pd.read_csv(ROOT / config["lr_database"])
    names = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    assert set(names).issubset(config["datasets"]), names
    np.random.seed(config["seed"]); random.seed(config["seed"])
    records = preflight(config, lr, names)
    if args.mode == "check":
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "datasets": records,
                  "config": config, "environment": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "anndata", "scanpy", "commot", "pyreadr", "POT"]},
                  "python": sys.executable, "comflow_source": str(Path(_spatial_communication.__file__).resolve())}
        (ROOT / "checks").mkdir(exist_ok=True)
        (ROOT / "checks/preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Preflight passed; no inference was run.", flush=True)
        return
    root = args.output_root or (ROOT.parent / "outputs/ccc_rerun_smoke" if args.mode == "smoke" else Path(config["output_root"]))
    for name in names:
        infer_one(name, config, lr, root, args.mode == "smoke", records[name])


if __name__ == "__main__":
    main()
