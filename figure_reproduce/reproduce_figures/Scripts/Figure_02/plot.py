"""Build publication panels for the clean Study 00 simulator benchmark."""
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
from matplotlib.patches import Patch
from layout import FIGURES, PROJECT, RESULTS
from feast_plate import register_arial_fonts

DEFAULT_METRICS_CSV = RESULTS / 'article/metrics/simulator_quality_metrics.csv'
DEFAULT_OUTPUT_DIR = FIGURES / 'supporting'
DEFAULT_OUTPUT_STEM = 'simulator_benchmark_boxplots'
DEFAULT_DECISION_JSON = RESULTS / 'article/metrics/publication_decision.json'
METHOD_LABELS = {'FEAST_reference_rank': 'FEAST', 'SRTsim': 'SRTsim', 'Splatter': 'Splatter', 'Splatter_Simple': 'Splatter Simple', 'scCube': 'scCube'}
METHOD_ORDER = ['FEAST', 'SRTsim', 'Splatter', 'Splatter Simple', 'scCube']
PALETTE = {'FEAST': '#0072B2', 'SRTsim': '#CC79A7', 'Splatter': '#D55E00', 'Splatter Simple': '#999999', 'scCube': '#E69F00'}
METRIC_SPECS = [{'column': 'input_mean_corr', 'label': 'Mean Corr ↑', 'higher_is_better': True}, {'column': 'input_variance_corr', 'label': 'Var Corr ↑', 'higher_is_better': True}, {'column': 'relative_error_mean', 'label': 'Real Error Mean ↓', 'higher_is_better': False, 'yscale': 'log'}, {'column': 'moran_i_correlation', 'label': 'Moran I Corr ↑', 'higher_is_better': True}, {'column': 'gene_zero_fraction_wasserstein', 'label': 'Zero Frac WDist ↓', 'higher_is_better': False, 'yscale': 'log'}, {'column': 'zero_mask_jaccard', 'label': 'Zero Jaccard ↑', 'higher_is_better': True}, {'column': 'cosine_divergence', 'label': 'Cos Div ↓', 'higher_is_better': False, 'yscale': 'log'}, {'column': 'library_size_wasserstein', 'label': 'Library WDist ↓', 'higher_is_better': False, 'yscale': 'log'}]

def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT))
    except ValueError:
        return str(resolved)

def load_metrics(metrics_csv: Path) -> pd.DataFrame:
    """Load the canonical atomic table and enforce its clean-run design."""
    if not metrics_csv.is_file():
        raise FileNotFoundError(f'Metric table does not exist: {metrics_csv}')
    metrics = pd.read_csv(metrics_csv, float_precision='round_trip')
    required = {'simulator', 'sample', 'identity_flag'} | {str(spec['column']) for spec in METRIC_SPECS}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f'Missing required columns in {metrics_csv}: {missing}')
    duplicate_mask = metrics.duplicated(['simulator', 'sample'], keep=False)
    if duplicate_mask.any():
        duplicate_keys = metrics.loc[duplicate_mask, ['simulator', 'sample']].drop_duplicates().to_dict('records')
        raise ValueError(f'Duplicate simulator/sample rows: {duplicate_keys}')
    observed_simulators = set(metrics['simulator'].dropna())
    expected_simulators = set(METHOD_LABELS)
    if observed_simulators != expected_simulators:
        missing_simulators = sorted(expected_simulators - observed_simulators)
        unexpected_simulators = sorted(observed_simulators - expected_simulators)
        raise ValueError(f'Metric table does not match the clean five-simulator design: missing={missing_simulators}, unexpected={unexpected_simulators}')
    sample_counts = metrics.groupby('simulator')['sample'].nunique()
    if len(set(sample_counts)) != 1 or int(sample_counts.iloc[0]) != 12:
        raise ValueError(f'Expected 12 samples per simulator, found {sample_counts.to_dict()}')
    metrics = metrics.copy()
    metrics['method'] = metrics['simulator'].map(METHOD_LABELS)
    return metrics

def load_decision(decision_json: Path) -> dict[str, object]:
    if not decision_json.is_file():
        raise FileNotFoundError(f'Decision record does not exist: {decision_json}')
    decision = json.loads(decision_json.read_text())
    if decision.get('publication_claim_authorized') is not False:
        raise ValueError('Study 00 visualization requires an unpromoted decision record')
    if 'gene_variance_wasserstein' not in decision.get('feast_aggregate_metrics_worsened', []):
        raise ValueError('Study 00 worsened-metric disposition does not match the figure')
    return decision

def build_long_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in METRIC_SPECS:
        column = str(spec['column'])
        subset = metrics[['method', 'sample', 'identity_flag', column]].dropna(subset=[column])
        for row in subset.itertuples(index=False):
            rows.append({'metric': spec['label'], 'method': row.method, 'sample': row.sample, 'score': float(getattr(row, column)), 'identity_flag': bool(row.identity_flag), 'source_column': column, 'higher_is_better': bool(spec['higher_is_better'])})
    if not rows:
        raise ValueError('The metric table contains no plottable values.')
    return pd.DataFrame(rows)

def panel_method_order(long_table: pd.DataFrame, spec: dict[str, object]) -> list[str]:
    """Order one panel by its median, best to worst, with unavailable methods last."""
    label = str(spec['label'])
    medians = long_table.loc[long_table['metric'] == label].groupby('method', sort=False)['score'].median()
    tie_break = {method: index for index, method in enumerate(METHOD_ORDER)}
    available = [method for method in METHOD_ORDER if method in medians.index and pd.notna(medians[method])]
    if bool(spec['higher_is_better']):
        available.sort(key=lambda method: (-float(medians[method]), tie_break[method]))
    else:
        available.sort(key=lambda method: (float(medians[method]), tie_break[method]))
    return available + [method for method in METHOD_ORDER if method not in available]

def build_median_table(long_table: pd.DataFrame) -> pd.DataFrame:
    """Record the exact panel-specific ordering used by the renderer."""
    records: list[dict[str, object]] = []
    for spec in METRIC_SPECS:
        label = str(spec['label'])
        subset = long_table.loc[long_table['metric'] == label]
        medians = subset.groupby('method', sort=False)['score'].median()
        counts = subset.groupby('method', sort=False)['score'].count()
        for rank, method in enumerate(panel_method_order(long_table, spec), start=1):
            records.append({'metric': label, 'source_column': str(spec['column']), 'higher_is_better': bool(spec['higher_is_better']), 'panel_rank': rank, 'method': method, 'median_score': float(medians[method]) if method in medians.index else np.nan, 'n_available': int(counts.get(method, 0))})
    return pd.DataFrame(records)

def configure_matplotlib() -> None:
    register_arial_fonts()
    plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial', 'mathtext.fallback': None, 'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def draw_boxplot_panel(ax: plt.Axes, long_table: pd.DataFrame, spec: dict[str, object]) -> None:
    label = str(spec['label'])
    subset = long_table.loc[long_table['metric'] == label]
    panel_order = panel_method_order(long_table, spec)
    method_rows = [subset.loc[subset['method'] == method].sort_values('sample') for method in panel_order]
    plotted = [(position, method, rows) for position, (method, rows) in enumerate(zip(panel_order, method_rows), start=1) if not rows.empty]
    boxplot = ax.boxplot([rows['score'].to_numpy() for _, _, rows in plotted], positions=[position for position, _, _ in plotted], patch_artist=True, widths=0.52, showfliers=False, medianprops={'color': '#D97946', 'linewidth': 1.2}, boxprops={'linewidth': 0.8, 'color': '#444444'}, whiskerprops={'linewidth': 0.8, 'color': '#444444'}, capprops={'linewidth': 0.8, 'color': '#444444'})
    for patch, (_, method, _) in zip(boxplot['boxes'], plotted):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.72)
    for position, method, rows in plotted:
        scores = rows['score'].to_numpy()
        jitter = np.zeros(1) if len(scores) == 1 else np.linspace(-0.13, 0.13, len(scores))
        x_positions = np.full(len(scores), position) + jitter
        color = PALETTE[method]
        ax.scatter(x_positions, scores, s=20, c=color, edgecolors='white', linewidths=0.4, alpha=0.9, zorder=3)
    for position, rows in enumerate(method_rows, start=1):
        if rows.empty:
            ax.text(position, 0.04, 'NA', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=10, color='#777777')
    ax.set_xlim(0.5, len(panel_order) + 0.5)
    ax.set_title(label, fontsize=15, fontweight='bold', pad=9)
    ax.set_xticks(range(1, len(panel_order) + 1))
    ax.set_xticklabels(panel_order, rotation=36, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', color='#DDDDDD', linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine_name in ('left', 'bottom'):
        spine = ax.spines[spine_name]
        spine.set_linewidth(0.8)
        spine.set_color('#555555')
    scores = subset['score'].dropna()
    data_min = float(scores.min())
    data_max = float(scores.max())
    if spec.get('yscale') == 'log':
        positive = scores[scores > 0]
        if positive.empty:
            raise ValueError(f'Log-scale metric {label} has no positive values')
        ax.set_yscale('log')
        ax.set_ylim(float(positive.min()) * 0.65, float(positive.max()) * 1.35)
    elif bool(spec['higher_is_better']):
        span = max(data_max - data_min, abs(data_max) * 0.1, 1e-06)
        lower = data_min - 0.1 * span
        if data_min >= 0:
            lower = max(-0.02, lower)
        upper = data_max + 0.1 * span
        if data_max <= 1.0:
            upper = min(1.02, upper)
        ax.set_ylim(lower, upper)
    else:
        span = max(data_max - data_min, abs(data_max) * 0.1, 1e-06)
        lower = 0 if data_min >= 0 else data_min - 0.15 * span
        ax.set_ylim(lower, data_max + 0.15 * span)

def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths: list[Path] = []
    formats = {'pdf': {'metadata': {'CreationDate': None, 'ModDate': None, 'Creator': 'FEAST publication visualization'}}, 'svg': {'metadata': {'Date': None, 'Creator': 'FEAST publication visualization'}}, 'png': {'dpi': dpi, 'metadata': {'Software': 'FEAST publication visualization'}}}
    for suffix, kwargs in formats.items():
        path = output_dir / f'{stem}.{suffix}'
        fig.savefig(path, bbox_inches='tight', **kwargs)
        print(f'Saved: {path}')
        paths.append(path)
    return paths

def draw_consolidated_figure(long_table: pd.DataFrame, output_dir: Path, output_stem: str, dpi: int) -> list[Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.75))
    axes = axes.flatten()
    for ax, spec in zip(axes, METRIC_SPECS):
        draw_boxplot_panel(ax, long_table, spec)
    for row_start in (0, 4):
        axes[row_start].set_ylabel('Metric value', fontsize=16)
    method_handles = [Patch(facecolor=PALETTE[method], edgecolor='none', label=method, alpha=0.85) for method in METHOD_ORDER]
    n_datasets = long_table['sample'].nunique()
    fig.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=5, frameon=False, fontsize=12, handlelength=0.9, handletextpad=0.5)
    fig.suptitle(f'Simulator benchmark across {n_datasets} spatial transcriptomics datasets', fontsize=20, fontweight='bold', y=0.995)
    fig.text(0.5, 0.955, 'Each panel is ordered by its median (best → worst); unavailable methods are last. Orders are panel-specific, not an overall ranking.', ha='center', va='top', fontsize=12, color='#4F4F4F')
    fig.tight_layout(rect=(0, 0.13, 1, 0.93), w_pad=1.6, h_pad=1.8)
    paths = save_figure(fig, output_dir, output_stem, dpi)
    plt.close(fig)
    return paths

def draw_individual_panels(long_table: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for spec in METRIC_SPECS:
        configure_matplotlib()
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.5))
        draw_boxplot_panel(ax, long_table, spec)
        ax.set_ylabel('Metric value', fontsize=15)
        safe_name = str(spec['label']).replace(' ', '_').replace('↑', 'up').replace('↓', 'down')
        paths.extend(save_figure(fig, output_dir, f'panel_{safe_name}', dpi))
        plt.close(fig)
    return paths

def write_provenance(metrics_csv: Path, decision_json: Path, decision: dict[str, object], metrics: pd.DataFrame, long_table: pd.DataFrame, output_dir: Path, output_paths: list[Path]) -> Path:
    metrics_provenance = metrics_csv.parent / 'provenance.json'
    payload = {'schema_version': 1, 'figure_set': 'study00_simulator_benchmark', 'source': {'metrics_csv': repository_path(metrics_csv), 'metrics_provenance': repository_path(metrics_provenance) if metrics_provenance.is_file() else None, 'decision_json': repository_path(decision_json), 'plot_script': repository_path(Path(__file__))}, 'design': {'atomic_rows': int(len(metrics)), 'samples': sorted(metrics['sample'].unique().tolist()), 'simulators': METHOD_ORDER, 'historical_feast_ot_included': False, 'panel_ordering': {'statistic': 'median', 'direction': 'metric-specific higher/lower is better', 'unavailable': 'last', 'tie_break_order': METHOD_ORDER, 'overall_ranking_claimed': False, 'orders': {str(spec['label']): panel_method_order(long_table, spec) for spec in METRIC_SPECS}}, 'metric_selection': {'displayed': [str(spec['column']) for spec in METRIC_SPECS], 'excluded': {'gene_variance_wasserstein': 'replaced by cosine_divergence per author request', 'gene_mean_wasserstein': 'mean-fidelity redundancy; retained correlation and relative error'}, 'selection_based_on_method_rank': False}}, 'scientific_disposition': {'status': decision.get('status'), 'publication_claim_authorized': False, 'figure_promotion_authorized': False, 'winner_or_composite_authorized': False, 'worsened_metric_replaced_by': 'cosine_divergence'}, 'outputs': {path.name: None for path in sorted(output_paths, key=lambda item: item.name)}}
    path = output_dir / 'figure_provenance.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Saved: {path}')
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-csv', type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument('--decision-json', type=Path, default=DEFAULT_DECISION_JSON)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--output-stem', default=DEFAULT_OUTPUT_STEM)
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    metrics = load_metrics(args.metrics_csv)
    decision = load_decision(args.decision_json)
    long_table = build_long_table(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / 'plot_data_long.csv'
    medians_path = args.output_dir / 'metric_medians.csv'
    long_table.to_csv(long_path, index=False)
    build_median_table(long_table).to_csv(medians_path, index=False)
    print(f'Saved: {long_path}')
    print(f'Saved: {medians_path}')
    output_paths = [long_path, medians_path]
    output_paths.extend(draw_consolidated_figure(long_table, args.output_dir, args.output_stem, args.dpi))
    output_paths.extend(draw_individual_panels(long_table, args.output_dir, args.dpi))
    write_provenance(args.metrics_csv, args.decision_json, decision, metrics, long_table, args.output_dir, output_paths)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
