"""Slice-panel spatial comparison: Reference vs FEAST vs SRTsim vs scCube.

One curated sample per platform, one marker gene per panel.
Each column shares a log1p vmax across its four rows.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import sparse
from layout import FIGURES, INPUTS, RESULTS

OUTPUT_DIR = FIGURES / 'supporting'
OUTPUT_STEM = 'slice_panel_spatial_comparison'
REFERENCE_DIR = INPUTS / 'reference'
FEAST_SIM_DIR = RESULTS / 'run/simulations'
PREVIOUS_FEAST_DIR = None
SRTSIM_DIR = INPUTS / 'external/SRTsim'
SCCUBE_DIR = INPUTS / 'external/scCube'
SOURCES = ['Reference', 'FEAST', 'SRTsim', 'scCube']
SOURCE_LABELS = ['Real\nReference', 'FEAST', 'SRTsim', 'scCube']
PANEL_SPECS = [{'sample': 'DLPFC_151670', 'platform': '10X Visium', 'dataset': 'Human DLPFC', 'gene': 'SCGB2A2'}, {'sample': 'MERFISH_120', 'platform': 'MERFISH', 'dataset': 'Mouse brain', 'gene': 'Pvalb'}, {'sample': 'OpenST_006', 'platform': 'OpenST', 'dataset': 'Human metastatic lymph node', 'gene': 'IGKC'}, {'sample': 'Slideseq_001', 'platform': 'Slide-seq V2', 'dataset': 'Human melanoma metastasis', 'gene': 'IGLC2'}, {'sample': 'Stereoseq_E12_5_E2S1', 'platform': 'Stereo-seq', 'dataset': 'Mouse embryo', 'gene': 'Fabp7'}, {'sample': 'Xenium_LymphNode', 'platform': 'Xenium', 'dataset': 'Human lymph node', 'gene': 'CD3E'}]
PLATFORM_POINT_MULT = {'10X Visium': 2.2, 'MERFISH': 1.5, 'Slide-seq V2': 1.8, 'Stereo-seq': 1.0, 'OpenST': 1.0, 'Xenium': 0.7}
POINT_SIZE_SCALE = 2.64
POINT_CAP = 10000

def h5ad_path(source: str, sample: str) -> Path:
    if source == 'Reference':
        return REFERENCE_DIR / f'{sample}.h5ad'
    if source == 'FEAST':
        return FEAST_SIM_DIR / f'{sample}.h5ad'
    if source == 'Previous FEAST':
        return PREVIOUS_FEAST_DIR / f'{sample}.h5ad'
    if source == 'SRTsim':
        return SRTSIM_DIR / f'{sample}.h5ad'
    if source == 'scCube':
        filename = 'Slideseq_001_sccube_identity.h5ad' if sample == 'Slideseq_001' else f'{sample}.h5ad'
        return SCCUBE_DIR / filename
    raise ValueError(f'Unknown source: {source}')

def _gene_vector(data: ad.AnnData, gene: str) -> np.ndarray:
    if not data.var_names.is_unique or gene not in data.var_names:
        raise ValueError(f'{gene!r} requires an exact, unique gene identifier in {data.filename}')
    matrix = data.X
    # AnnData 0.10's sparse backed slicing uses a SciPy method removed in 1.17.
    if hasattr(matrix, 'to_memory'):
        matrix = matrix.to_memory()
    matrix = matrix[:, data.var_names.get_loc(gene)]
    values = matrix.toarray() if sparse.issparse(matrix) else matrix
    return np.asarray(values, dtype=float).ravel()

def read_gene_vector(path: Path, gene: str) -> np.ndarray:
    data = ad.read_h5ad(path, backed='r')
    try:
        return _gene_vector(data, gene)
    finally:
        data.file.close()

def _spatial_coords(data: ad.AnnData) -> np.ndarray:
    if 'spatial' in data.obsm:
        coords = np.asarray(data.obsm['spatial'])
    elif {'x', 'y'}.issubset(data.obs.columns):
        coords = data.obs[['x', 'y']].to_numpy()
    else:
        raise ValueError(f'No spatial coords in {data.filename}')
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f'Spatial coords shape {coords.shape}: {data.filename}')
    return coords[:, :2].astype(float)

def read_spatial_coords(path: Path) -> np.ndarray:
    data = ad.read_h5ad(path, backed='r')
    try:
        return _spatial_coords(data)
    finally:
        data.file.close()

def read_spatial_marker(path: Path, gene: str) -> ad.AnnData:
    """Read one expression column together with its original spot identities."""
    data = ad.read_h5ad(path, backed='r')
    try:
        values = _gene_vector(data, gene)
        coords = _spatial_coords(data)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Expected finite nonnegative expression in {path}')
        if not np.isfinite(coords).all():
            raise ValueError(f'Expected finite coordinates in {path}')
        return ad.AnnData(
            X=values[:, None],
            obs=pd.DataFrame(index=data.obs_names.copy()),
            var=pd.DataFrame(index=[gene]),
            obsm={'spatial': coords},
        )
    finally:
        data.file.close()

def point_size(n_obs: int, platform: str) -> float:
    if n_obs > 250000:
        base = 0.08
    elif n_obs > 80000:
        base = 0.12
    elif n_obs > 35000:
        base = 0.22
    elif n_obs > 15000:
        base = 0.35
    elif n_obs > 7000:
        base = 0.55
    else:
        base = 1.6
    return base * POINT_SIZE_SCALE * PLATFORM_POINT_MULT.get(platform, 1.0)

def configure_matplotlib() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def draw_figure(output_dir: Path, output_stem: str, dpi: int) -> list[Path]:
    configure_matplotlib()
    n_rows = len(SOURCES)
    n_cols = len(PANEL_SPECS)
    expression_norm = Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.75 * n_cols, 2.7 * n_rows), squeeze=False)
    for col_idx, spec in enumerate(PANEL_SPECS):
        sample = spec['sample']
        gene = spec['gene']
        platform = spec['platform']
        all_expr = []
        for source in SOURCES:
            path = h5ad_path(source, sample)
            all_expr.append(np.log1p(read_gene_vector(path, gene)))
        combined = np.concatenate(all_expr)
        vmax = float(np.nanpercentile(combined, 99.5))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(combined))
        for row_idx, source in enumerate(SOURCES):
            path = h5ad_path(source, sample)
            coords = read_spatial_coords(path)
            expr = np.log1p(read_gene_vector(path, gene))
            n_orig = len(expr)
            if POINT_CAP > 0 and n_orig > POINT_CAP:
                rng = np.random.default_rng(2026)
                keep = np.sort(rng.choice(n_orig, size=POINT_CAP, replace=False))
                coords = coords[keep]
                expr = expr[keep]
            ax = axes[row_idx, col_idx]
            sizes = np.full(len(expr), point_size(len(expr), platform))
            ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=expr / vmax, cmap='viridis', norm=expression_norm, marker='.', linewidths=0, rasterized=True)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
        top_ax = axes[0, col_idx]
        top_ax.text(0.5, 1.32, f"{spec['dataset']}\n({platform})", transform=top_ax.transAxes, ha='center', va='bottom', fontsize=12, linespacing=1.12)
        top_ax.plot([0.18, 0.82], [1.26, 1.26], transform=top_ax.transAxes, color='black', linewidth=0.7, clip_on=False)
        top_ax.text(0.5, 1.1, gene, transform=top_ax.transAxes, ha='center', va='bottom', fontsize=12, fontstyle='italic')
    plt.subplots_adjust(left=0.07, right=0.985, top=0.8, bottom=0.095, wspace=0.015, hspace=0.02)
    for row_idx, label in enumerate(SOURCE_LABELS):
        bbox = axes[row_idx, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2
        fig.text(0.012, y_center, label, rotation=90, ha='center', va='center', fontsize=13, fontweight='bold')
    cbar_ax = fig.add_axes([0.35, 0.025, 0.3, 0.01])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=expression_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    colorbar_label = 'log1p expression / q99.5\n(shared within each gene)' if n_cols <= 2 else 'Column-normalized log1p expression [value / q99.5(all sources)]'
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    paths: list[Path] = []
    for suffix, kwargs in {'pdf': {'metadata': {'Creator': 'FEAST Study 00 slice panel'}}, 'svg': {'metadata': {'Date': None, 'Creator': 'FEAST Study 00 slice panel'}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST Study 00 slice panel'}}}.items():
        path = stem.with_suffix(f'.{suffix}')
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved: {path}')
        paths.append(path)
    plt.close(fig)
    return paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--output-stem', default=OUTPUT_STEM)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--sample', action='append', choices=[spec['sample'] for spec in PANEL_SPECS], help='Show only this existing panel; repeat to select more than one.')
    parser.add_argument('--gene', action='append', help='Plot this gene for a single selected sample; repeat for multiple genes.')
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=SOURCES)
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE_DIR)
    parser.add_argument('--feast-dir', type=Path, default=FEAST_SIM_DIR)
    parser.add_argument('--feast-label', default='FEAST')
    parser.add_argument('--previous-feast-dir', type=Path, help='Include a retained FEAST run for comparison.')
    parser.add_argument('--previous-feast-label', default='Previous FEAST')
    parser.add_argument('--srtsim-dir', type=Path, default=SRTSIM_DIR)
    parser.add_argument('--sccube-dir', type=Path, default=SCCUBE_DIR)
    return parser.parse_args()

def main() -> int:
    global PANEL_SPECS, SOURCES, SOURCE_LABELS, REFERENCE_DIR, FEAST_SIM_DIR, PREVIOUS_FEAST_DIR, SRTSIM_DIR, SCCUBE_DIR
    args = parse_args()
    labels = dict(zip(SOURCES, SOURCE_LABELS))
    if args.sample:
        PANEL_SPECS = [spec for spec in PANEL_SPECS if spec['sample'] in set(args.sample)]
    if args.gene:
        if len(PANEL_SPECS) != 1:
            raise ValueError('--gene requires exactly one selected sample')
        PANEL_SPECS = [{**PANEL_SPECS[0], 'gene': gene} for gene in args.gene]
    SOURCES = args.sources
    labels['FEAST'] = args.feast_label
    if args.previous_feast_dir:
        position = SOURCES.index('FEAST') if 'FEAST' in SOURCES else len(SOURCES)
        SOURCES = [*SOURCES[:position], 'Previous FEAST', *SOURCES[position:]]
        labels['Previous FEAST'] = args.previous_feast_label
        PREVIOUS_FEAST_DIR = args.previous_feast_dir
    SOURCE_LABELS = [labels[source] for source in SOURCES]
    REFERENCE_DIR, FEAST_SIM_DIR = args.reference_dir, args.feast_dir
    SRTSIM_DIR, SCCUBE_DIR = args.srtsim_dir, args.sccube_dir
    draw_figure(args.output_dir, args.output_stem, args.dpi)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
