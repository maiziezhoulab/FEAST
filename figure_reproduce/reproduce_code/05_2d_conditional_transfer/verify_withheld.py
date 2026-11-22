"""Replay the five half-section jobs after altering their withheld counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from workflow import STUDY_DIR, declared_jobs, expected_job_path, load_config


def perturb(inputs, output, results, workers):
    config_path = STUDY_DIR / "config.yaml"
    config = load_config(config_path)
    jobs = [job for job in declared_jobs(config) if job.mode == "mask_half"]
    manifest = pd.read_csv(inputs / "input_files.csv", dtype=str)
    manifest = manifest.set_index(["dataset", "slice_id"])
    variant_dir = output / "inputs"
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant_manifest = output / "input_files.csv"
    variants = manifest.copy()
    metadata = {}
    for job in jobs:
        index = (job.dataset, job.source)
        source = inputs / manifest.loc[index, "relative_path"]
        relative = Path(job.dataset) / source.name
        variants.loc[index, "relative_path"] = str(relative)
        variant = variant_dir / relative
        variant.parent.mkdir(parents=True, exist_ok=True)
        if not variant.exists():
            data = ad.read_h5ad(source)
            x = data.obsm["spatial"][:, 0]
            split = float(np.median(x))
            factor = np.where(x > split, 17, 1)
            for key, matrix in [("X", data.X), ("counts", data.layers["counts"])]:
                varied = sparse.diags(factor, dtype=matrix.dtype) @ matrix if sparse.issparse(matrix) else matrix * factor[:, None]
                if key == "X":
                    data.X = varied
                else:
                    data.layers[key] = varied
            data.write_h5ad(variant, compression="gzip")
            metadata[job] = {"withheld_spots": int(np.count_nonzero(factor == 17)), "threshold": split,
                             "changed_count_entries": int(sparse.csr_matrix(data.layers["counts"][x > split]).count_nonzero())}
        else:
            data = ad.read_h5ad(variant)
            x = data.obsm["spatial"][:, 0]
            split = float(np.median(x))
            metadata[job] = {"withheld_spots": int(np.count_nonzero(x > split)), "threshold": split,
                             "changed_count_entries": int(sparse.csr_matrix(data.layers["counts"][x > split]).count_nonzero())}
    variants.reset_index().to_csv(variant_manifest, index=False)

    def run(job):
        generation = output / "generated"
        folder = expected_job_path(generation, job)
        if not folder.exists():
            command = [sys.executable, os.path.relpath(STUDY_DIR / "run.py"), "--config", os.path.relpath(config_path),
                       "--manifest", str(variant_manifest), "--input-dir", str(variant_dir), "--output-dir", str(generation),
                       "--mode", "mask_half", "--dataset", job.dataset, "--slice", job.source]
            subprocess.run(command, check=True)
        original = expected_job_path(results / "final", job) / "generated.h5ad"
        variant = folder / "generated.h5ad"
        original_data = ad.read_h5ad(original)
        altered_data = ad.read_h5ad(variant)
        assert original_data.obs_names.equals(altered_data.obs_names) and original_data.var_names.equals(altered_data.var_names)
        changes = [(sparse.csr_matrix(a) != sparse.csr_matrix(b)).nnz for a, b in
                   ((original_data.X, altered_data.X), (original_data.layers["counts"], altered_data.layers["counts"]))]
        assert changes == [0, 0], (job, changes)
        return {"dataset": job.dataset, "slice_id": job.source, "seed": config["public_seed"],
                "withheld_count_multiplier": 17, **metadata[job], "generated_x_changed": changes[0],
                "generated_counts_changed": changes[1]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        checks = list(pool.map(run, jobs))
    pd.DataFrame(checks).to_csv(output / "comparison.csv", index=False)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path("Zenodo"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    inp = args.zenodo / "inputs" / STUDY_DIR.name
    results = args.zenodo / "results" / STUDY_DIR.name
    perturb(inp, results / "withheld_count_validation", results, args.workers)
