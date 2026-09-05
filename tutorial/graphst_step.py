"""Run in the GraphST environment; the FEAST notebook supplies prepared input."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("R_HOME", str(Path(sys.prefix) / "lib" / "R"))

import anndata as ad
import pandas as pd
import scanpy as sc
import torch
from GraphST import GraphST as graphst_module
from GraphST.utils import clustering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GraphST requires the requested CUDA device.")
    data = sc.read_h5ad(args.input)
    n_clusters = data.obs["ground_truth"].nunique()
    # The notebook has already selected genes and applied normalize_total/log1p.
    data.var["highly_variable"] = True
    sc.pp.scale(data, zero_center=False, max_value=10)
    model = graphst_module.GraphST(
        data, device=torch.device(args.device), epochs=600,
        dim_input=data.n_vars, random_seed=2026, datatype="10X",
    )
    result = model.train()
    # GraphST's utility uses PCA seed 42 and mclust seed 2020 internally.
    clustering(result, n_clusters=n_clusters, radius=50, method="mclust", refinement=True)
    # Export only the result needed by the notebook, avoiding GraphST graph objects.
    output = ad.AnnData(X=data.X.copy(), obs=data.obs.copy(), var=data.var.copy())
    output.obs["predicted_cluster"] = pd.Categorical(result.obs["domain"].astype(str))
    output.obsm["spatial"] = data.obsm["spatial"].copy()
    output.obsm["emb"] = result.obsm["emb"].copy()
    output.write_h5ad(args.output)


if __name__ == "__main__":
    main()
