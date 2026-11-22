"""Combined spatial expression and simulator-error maps across six platforms."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from layout import FIGURES
from plot_delta_spatial import DELTA_CMAP, POINT_CAP as DELTA_POINT_CAP, paired_residual
from plot_slice_panel import PANEL_SPECS, POINT_CAP as EXPRESSION_POINT_CAP, PLATFORM_POINT_MULT, POINT_SIZE_SCALE, SOURCES, SOURCE_LABELS, h5ad_path, read_spatial_marker

OUTPUT_DIR = FIGURES / 'supporting'
OUTPUT_STEM = 'simulator_spatial_comparison'
DELTA_SOURCES = ('FEAST', 'SRTsim', 'scCube')
DELTA_LABELS = ('FEAST − reference', 'SRTsim − reference', 'scCube − reference')
SPOT_AREA_MULTIPLIER = 1.6
PLATFORM_SPOT_AREA_MULTIPLIER = {'MERFISH': 2.0, 'OpenST': 4.0, 'Slide-seq V2': 2.0, 'Stereo-seq': 2.0, 'Xenium': 4.0}

def configure_matplotlib() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _point_size(n_obs: int, platform: str) -> float:
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
    return base * POINT_SIZE_SCALE * PLATFORM_POINT_MULT.get(platform, 1.0) * SPOT_AREA_MULTIPLIER * PLATFORM_SPOT_AREA_MULTIPLIER.get(platform, 1.0)

def _sample_indices(n_obs: int, cap: int) -> np.ndarray | slice:
    if n_obs <= cap:
        return slice(None)
    return np.sort(np.random.default_rng(2026).choice(n_obs, size=cap, replace=False))

def _hide_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

def draw_figure(
    output_dir: Path, output_stem: str, dpi: int, source_dirs: dict[str, Path] | None = None,
) -> list[Path]:
    configure_matplotlib()
    expression_norm = Normalize(vmin=0, vmax=1)
    delta_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig = plt.figure(figsize=(17.0, 18.0))
    grid = fig.add_gridspec(9, len(PANEL_SPECS) + 1, left=0.105, right=0.99, top=0.88, bottom=0.055, width_ratios=(0.08,) + (1,) * len(PANEL_SPECS), height_ratios=(1, 1, 1, 1, 0.12, 1, 1, 1, 1), wspace=0.004, hspace=0.008)
    expression_axes = np.empty((len(SOURCES), len(PANEL_SPECS)), dtype=object)
    delta_axes = np.empty((len(DELTA_SOURCES), len(PANEL_SPECS)), dtype=object)
    residual_frames, limits, matches = [], [], []
    source_dirs = source_dirs or {}
    for col_idx, spec in enumerate(PANEL_SPECS):
        sample = str(spec['sample'])
        gene = str(spec['gene'])
        platform = str(spec['platform'])
        paths = {source: h5ad_path(source, sample) for source in SOURCES}
        for source, folder in source_dirs.items():
            paths[source] = folder / paths[source].name
        data = {source: read_spatial_marker(path, gene) for source, path in paths.items()}
        expression_by_source = {source: np.log1p(marker.X[:, 0]) for source, marker in data.items()}
        expression_vmax = float(np.nanpercentile(np.concatenate(list(expression_by_source.values())), 99.5))
        if not np.isfinite(expression_vmax) or expression_vmax <= 0:
            expression_vmax = max(values.max() for values in expression_by_source.values()) or 1.0
        reference = data['Reference']
        reference_expression = expression_by_source['Reference']
        reference_xy = reference.obsm['spatial']
        delta_by_source = {}
        for source in DELTA_SOURCES:
            residual, match = paired_residual(reference, data[source], paths[source])
            delta_by_source[source] = residual
            matches.append({
                'sample': sample, 'gene': gene, 'source': source,
                'reference_path': str(paths['Reference'].resolve()),
                'simulation_path': str(paths[source].resolve()), **match,
            })
        delta_scale = float(np.nanpercentile(np.abs(np.concatenate(list(delta_by_source.values()))), 99))
        if not np.isfinite(delta_scale) or delta_scale <= 0:
            delta_scale = 0.5
        limits.append({
            'sample': sample, 'gene': gene, 'expression_vmax': expression_vmax,
            'residual_vmin': -delta_scale, 'residual_vmax': delta_scale,
        })
        residual_frames.append(pd.DataFrame({
            'sample': sample, 'gene': gene, 'spot_id': reference.obs_names,
            'x': reference_xy[:, 0], 'y': reference_xy[:, 1],
            'reference_log1p': reference_expression,
            **{f'{source}_log1p_residual': residual for source, residual in delta_by_source.items()},
        }))
        for row_idx, source in enumerate(SOURCES):
            ax = fig.add_subplot(grid[row_idx, col_idx + 1])
            expression_axes[row_idx, col_idx] = ax
            coords = data[source].obsm['spatial']
            values = expression_by_source[source]
            keep = _sample_indices(len(values), EXPRESSION_POINT_CAP)
            ax.scatter(coords[keep, 0], coords[keep, 1], s=np.full(len(values[keep]), _point_size(len(values), platform)), c=values[keep] / expression_vmax, cmap='viridis', norm=expression_norm, marker='.', linewidths=0, rasterized=True)
            _hide_spatial_axis(ax)
        delta_keep = _sample_indices(len(reference_expression), DELTA_POINT_CAP)
        for row_idx, source in enumerate(DELTA_SOURCES):
            ax = fig.add_subplot(grid[row_idx + 5, col_idx + 1])
            delta_axes[row_idx, col_idx] = ax
            values = delta_by_source[source][delta_keep]
            order = np.argsort(np.abs(values))
            ax.scatter(reference_xy[delta_keep][order, 0], reference_xy[delta_keep][order, 1], c=values[order] / delta_scale, cmap=DELTA_CMAP, norm=delta_norm, s=_point_size(len(reference_expression), platform), linewidths=0, alpha=0.85, rasterized=True)
            _hide_spatial_axis(ax)
        header_axis = expression_axes[0, col_idx]
        header_axis.text(0.5, 1.3, f"{spec['dataset']}\n({platform})", transform=header_axis.transAxes, ha='center', va='bottom', fontsize=10.5, linespacing=1.08)
        header_axis.plot([0.2, 0.8], [1.24, 1.24], transform=header_axis.transAxes, color='#444444', linewidth=0.65, clip_on=False)
        header_axis.text(0.5, 1.08, f'{gene}  ·  Δ q99 ±{delta_scale:.2g}', transform=header_axis.transAxes, ha='center', va='bottom', fontsize=9.5, fontstyle='italic')
    for row_idx, label in enumerate(SOURCE_LABELS):
        bounds = expression_axes[row_idx, 0].get_position()
        fig.text(0.035, bounds.y0 + bounds.height / 2, label, rotation=90, ha='center', va='center', fontsize=10.5, fontweight='bold')
    for row_idx, label in enumerate(DELTA_LABELS):
        bounds = delta_axes[row_idx, 0].get_position()
        fig.text(0.035, bounds.y0 + bounds.height / 2, label, rotation=90, ha='center', va='center', fontsize=10.5, fontweight='bold')
    expression_cbar_axis = fig.add_subplot(grid[:4, 0])
    expression_cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=expression_norm), cax=expression_cbar_axis, orientation='vertical')
    expression_cbar.ax.yaxis.set_label_position('left')
    expression_cbar.set_label('Expression: log1p(value) / column q99.5 across all sources', fontsize=9.5, labelpad=8)
    expression_cbar.ax.tick_params(labelsize=8)
    expression_cbar_axis = fig.add_subplot(grid[5:, 0])
    delta_cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=DELTA_CMAP, norm=delta_norm), cax=expression_cbar_axis, orientation='vertical')
    delta_cbar.ax.yaxis.set_label_position('left')
    delta_cbar.set_label('Error: [log1p(simulated) − log1p(reference)] / column q99(|Δ|)', fontsize=9.5, labelpad=8)
    delta_cbar.set_ticks((-1, -0.5, 0, 0.5, 1))
    delta_cbar.ax.tick_params(labelsize=8)
    fig.suptitle('Spatial expression and simulator error maps across six platforms', y=0.993, fontsize=17, fontweight='bold')
    fig.text(0.5, 0.973, 'Top: shared expression scale within each dataset. Bottom: simulator-minus-reference error with a separate, symmetric within-dataset scale.', ha='center', va='center', fontsize=10, color='#4F4F4F')
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / output_stem
    paths: list[Path] = []
    for suffix, kwargs in {'pdf': {'dpi': dpi, 'metadata': {'Creator': 'FEAST Study 00 combined spatial comparison'}}, 'svg': {'metadata': {'Date': None, 'Creator': 'FEAST Study 00 combined spatial comparison'}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST Study 00 combined spatial comparison'}}}.items():
        path = stem.with_suffix(f'.{suffix}')
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved: {path}')
        paths.append(path)
    plt.close(fig)
    residual_path = output_dir / f'{output_stem}_residuals.csv.gz'
    limits_path = output_dir / f'{output_stem}_color_limits.csv'
    matching_path = output_dir / f'{output_stem}_spot_matching.csv'
    pd.concat(residual_frames, ignore_index=True).to_csv(
        residual_path, index=False, compression={'method': 'gzip', 'mtime': 0},
    )
    pd.DataFrame(limits).to_csv(limits_path, index=False)
    pd.DataFrame(matches).to_csv(matching_path, index=False)
    provenance_path = output_dir / f'{output_stem}_provenance.json'
    provenance_path.write_text(json.dumps({
        'figure': 'Figure 2b',
        'script': 'visualization/00_simulator_benchmark/plot_spatial_comparison.py',
        'spot_matching': 'Complete unique spot IDs or a verified coordinate bijection',
        'residual': 'log1p(simulated) - log1p(reference) after the spot join',
        'residual_scale': 'Pooled absolute residual 99th percentile within each section, before sampling',
        'expression_scale': 'Pooled log1p expression 99.5th percentile within each section',
        'render_sampling': {'expression_cap': EXPRESSION_POINT_CAP, 'residual_cap': DELTA_POINT_CAP, 'seed': 2026},
        'tables': {'residuals': residual_path.name, 'color_limits': limits_path.name, 'spot_matching': matching_path.name},
    }, indent=2) + '\n')
    return paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--output-stem', default=OUTPUT_STEM)
    parser.add_argument('--dpi', type=int, default=3000)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--feast-dir', type=Path)
    parser.add_argument('--srtsim-dir', type=Path)
    parser.add_argument('--sccube-dir', type=Path)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    directories = dict(zip(SOURCES, (args.reference_dir, args.feast_dir, args.srtsim_dir, args.sccube_dir), strict=True))
    draw_figure(args.output_dir, args.output_stem, args.dpi, {source: path for source, path in directories.items() if path})
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
