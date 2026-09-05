"""Run same-slice batch correction in an existing scvi-tools environment."""

import argparse
import os
from pathlib import Path

# The worker does not use the parent notebook's inline plotting backend.
os.environ["MPLBACKEND"] = "Agg"

import anndata as ad
import scvi
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This scVI example uses CUDA, as in Study 04.")
    scvi.settings.seed = 42
    data = ad.read_h5ad(args.input)
    scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key="batch")
    model = scvi.model.SCVI(
        data, n_latent=10, n_layers=1, n_hidden=128,
        dropout_rate=0.1, gene_likelihood="zinb",
    )
    model.train(
        max_epochs=400, early_stopping=True, accelerator="gpu", devices=1,
        plan_kwargs={"lr": 0.001}, deterministic=True,
    )
    data.obsm["X_scVI"] = model.get_latent_representation()
    data.layers["scvi_normalized"] = model.get_normalized_expression(
        transform_batch="ref", return_numpy=True,
    )
    data.write_h5ad(args.output)


if __name__ == "__main__":
    main()
