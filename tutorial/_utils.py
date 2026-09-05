"""Small data and plotting helpers; FEAST calls stay in the notebooks."""

from pathlib import Path
import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def data_root():
    return Path(os.environ.get("FEAST_TUTORIAL_DATA", Path(__file__).parent / "data"))


def load_counts(path, label_key=None):
    """Read the supplied count matrix without filtering or normalizing it."""
    data = ad.read_h5ad(path)
    if "counts" in data.layers:
        data.X = data.layers["counts"].copy()
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
    assert np.isfinite(values).all() and (values >= 0).all(), "Counts must be finite and nonnegative."
    assert np.allclose(values, np.rint(values)), "Supply raw counts, not normalized expression."
    assert data.obs_names.is_unique and data.var_names.is_unique
    assert "spatial" in data.obsm
    if label_key is not None:
        assert label_key in data.obs and data.obs[label_key].notna().all()
    return data


def gene_summary(data):
    matrix = data.X.astype(np.float64)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    square = matrix.multiply(matrix) if sparse.issparse(matrix) else matrix ** 2
    return pd.DataFrame({
        "mean": mean,
        "variance": np.asarray(square.mean(axis=0)).ravel() - mean ** 2,
        "zero_fraction": 1 - np.asarray((matrix > 0).mean(axis=0)).ravel(),
    }, index=data.var_names)


def cache_reference_counts(data):
    """Prepare once the float32 count array used by FEAST's empirical decoder."""
    counts = data.layers["counts"] if "counts" in data.layers else data.X
    if sparse.issparse(counts):
        counts = counts.toarray()
    data.layers["counts"] = np.asarray(counts, dtype=np.float32)


def spatial_panel(ax, xy, values, title, categorical=False, vmax=None):
    xy = np.asarray(xy)
    if categorical:
        labels = pd.Series(values).astype(str).to_numpy()
        categories = sorted(set(labels))
        colors = plt.get_cmap("tab20", max(len(categories), 2))
        for i, label in enumerate(categories):
            selected = labels == label
            ax.scatter(*xy[selected, :2].T, s=3, color=colors(i), label=label, rasterized=True)
        ax.legend(markerscale=2, fontsize=6, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    else:
        artist = ax.scatter(*xy[:, :2].T, c=np.asarray(values), s=3,
                            cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
        ax.figure.colorbar(artist, ax=ax, fraction=0.035, pad=0.02)
    ax.set(title=title, aspect="equal")
    ax.invert_yaxis()
    ax.set_axis_off()


def gene_values(data, gene):
    values = data[:, [gene]].X
    return np.asarray(values.toarray() if sparse.issparse(values) else values).ravel()
