"""Score the complete Study 04B matrix without a composite or ranking."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import metrics
import workflow
STUDY_ROOT = Path(__file__).resolve().parent
IDENTITY_COLUMNS = ['configuration_id', 'condition', 'condition_role', 'alpha', 'alpha_stratum', 'method', 'method_seed', 'representation', 'representation_dimensions', 'primary_representation', 'analysis_role']

def _load_baseline(output_root: Path, condition: str) -> dict[str, np.ndarray | int]:
    path = output_root / 'baselines' / f'{condition}.npz'
    with np.load(path, allow_pickle=True) as payload:
        return {'embedding': np.asarray(payload['embedding'], dtype=np.float64), 'batch': np.asarray(payload['batch_labels']).astype(str), 'layers': np.asarray(payload['layer_labels']).astype(str), 'source_ids': np.asarray(payload['source_spot_ids']).astype(str), 'spatial': np.asarray(payload['spatial'], dtype=np.float64), 'n_reference': int(payload['n_reference']), 'n_query': int(payload['n_query']), 'path': path}

def _method_representations(method: str, embedding: np.ndarray, scoring_seed: int) -> list[tuple[str, bool, np.ndarray]]:
    if method == 'GraphST':
        return [('pca10_randomized_seed42', True, PCA(n_components=10, svd_solver='randomized', random_state=scoring_seed).fit_transform(embedding)), ('pca20_randomized_seed42_sensitivity', False, PCA(n_components=20, svd_solver='randomized', random_state=scoring_seed).fit_transform(embedding))]
    name = 'native_10d_topics' if method == 'STAMP' else 'native_10d_latent'
    return [(name, True, np.asarray(embedding, dtype=np.float64))]

def _load_candidate_embedding(candidate: Path) -> np.ndarray:
    with np.load(candidate / 'embeddings.npz', allow_pickle=True) as payload:
        return np.asarray(payload['embedding'], dtype=np.float64)

def _stability_metrics(values: np.ndarray, anchor: np.ndarray, *, n_reference: int, config: dict) -> dict[str, float]:
    query = values[n_reference:]
    anchor_query = anchor[n_reference:]
    k = int(config['analysis']['n_neighbors'])
    seed = int(config['analysis']['scoring_seed'])
    return {'query_stability_knn_jaccard_k': metrics.mean_knn_jaccard(anchor_query, query, k=k), 'query_stability_distance_spearman': metrics.sampled_pairwise_distance_spearman(anchor_query, query, max_pairs=int(config['analysis']['pairwise_distance_sample']), seed=seed), 'query_stability_linear_cka': metrics.linear_cka(anchor_query, query)}

def build_atomic_table(config: dict, output_root: Path) -> pd.DataFrame:
    conditions = workflow.condition_lookup(config)
    scoring_seed = int(config['analysis']['scoring_seed'])
    k = int(config['analysis']['n_neighbors'])
    baseline_data = {condition: _load_baseline(output_root, condition) for condition in conditions}
    method_values: dict[tuple[str, int, str, str], np.ndarray] = {}
    for condition, method, seed in workflow.production_cells(config):
        candidate = workflow.candidate_path(output_root, condition, method, seed)
        valid, reason = workflow.candidate_is_valid(candidate, config=config, output_root=output_root, condition=condition, method=method, seed=seed)
        if not valid:
            raise ValueError(f'invalid candidate {method}/{condition}/seed_{seed}: {reason}')
        embedding = _load_candidate_embedding(candidate)
        for representation, _, values in _method_representations(method, embedding, scoring_seed):
            method_values[method, seed, condition, representation] = values
    rows = []
    for condition, condition_spec in conditions.items():
        baseline = baseline_data[condition]
        values = np.asarray(baseline['embedding'])
        anchor = np.asarray(baseline_data['sim_0.00']['embedding'])
        row = {'configuration_id': config['configuration_id'], 'condition': condition, 'condition_role': condition_spec['role'], 'alpha': condition_spec.get('alpha'), 'alpha_stratum': 'natural_anchor' if condition == 'raw' else 'interpolation' if float(condition_spec['alpha']) <= 1.0 else 'extrapolation', 'method': 'Uncorrected', 'method_seed': -1, 'representation': 'fixed_panel_log1p_pca10_seed42', 'representation_dimensions': 10, 'primary_representation': True, 'analysis_role': 'uncorrected_baseline', 'n_reference': int(baseline['n_reference']), 'n_query': int(baseline['n_query'])}
        row.update(metrics.full_metric_row(values, baseline=values, spatial=np.asarray(baseline['spatial']), batch=np.asarray(baseline['batch']), layers=np.asarray(baseline['layers']), k=k, seed=scoring_seed))
        row.update(_stability_metrics(values, anchor, n_reference=int(baseline['n_reference']), config=config))
        rows.append(row)
    for condition, method, seed in workflow.production_cells(config):
        baseline = baseline_data[condition]
        candidate = workflow.candidate_path(output_root, condition, method, seed)
        metadata_path = candidate / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        raw_embedding = _load_candidate_embedding(candidate)
        for representation, primary, values in _method_representations(method, raw_embedding, scoring_seed):
            anchor = method_values[method, seed, 'sim_0.00', representation]
            row = {'configuration_id': config['configuration_id'], 'condition': condition, 'condition_role': conditions[condition]['role'], 'alpha': conditions[condition].get('alpha'), 'alpha_stratum': 'natural_anchor' if condition == 'raw' else 'interpolation' if float(conditions[condition]['alpha']) <= 1.0 else 'extrapolation', 'method': method, 'method_seed': int(seed), 'representation': representation, 'representation_dimensions': int(values.shape[1]), 'primary_representation': bool(primary), 'analysis_role': 'primary' if primary else 'representation_sensitivity', 'n_reference': int(baseline['n_reference']), 'n_query': int(baseline['n_query']), 'candidate_configuration_id': metadata['provenance']['configuration_id'], 'candidate_path': str(candidate.resolve())}
            row.update(metrics.full_metric_row(values, baseline=np.asarray(baseline['embedding']), spatial=np.asarray(baseline['spatial']), batch=np.asarray(baseline['batch']), layers=np.asarray(baseline['layers']), k=k, seed=scoring_seed))
            row.update(_stability_metrics(values, anchor, n_reference=int(baseline['n_reference']), config=config))
            rows.append(row)
    table = pd.DataFrame(rows)
    expected_rows = 5 + 45 + 15
    if len(table) != expected_rows:
        raise RuntimeError(f'atomic table has {len(table)} rows; expected {expected_rows}')
    prohibited = {'paired_retrieval_top1', 'paired_retrieval_median_rank', 'composite_score'}
    if prohibited & set(table):
        raise RuntimeError('atomic table contains a prohibited Study 04B metric')
    return table.sort_values(['method', 'condition', 'method_seed', 'primary_representation'], ascending=[True, True, True, False]).reset_index(drop=True)

def build_seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    primary = table[(table['method'] != 'Uncorrected') & table['primary_representation']].copy()
    metric_columns = [column for column in primary.select_dtypes(include=[np.number]).columns if column not in {'alpha', 'method_seed', 'representation_dimensions', 'n_reference', 'n_query'}]
    long = primary.melt(id_vars=['condition', 'condition_role', 'alpha', 'method', 'representation'], value_vars=metric_columns, var_name='metric', value_name='value')
    summary = long.groupby(['condition', 'condition_role', 'alpha', 'method', 'representation', 'metric'], dropna=False)['value'].agg(seed_mean='mean', seed_sd='std', seed_min='min', seed_max='max', seed_n='count').reset_index()
    return summary

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_ROOT / 'config.yaml')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    run_dir = args.run_dir.resolve() if args.run_dir else config['_output_dir']
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / 'metrics'
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite metric root: {output_dir}')
    output_dir.mkdir(parents=True)
    table = build_atomic_table(config, run_dir)
    summary = build_seed_summary(table)
    table_path = output_dir / 'atomic_metrics.csv'
    summary_path = output_dir / 'seed_summary_long.csv'
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary_json = {'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'atomic_rows': len(table), 'primary_method_rows': int(((table.method != 'Uncorrected') & table.primary_representation).sum()), 'uncorrected_rows': int((table.method == 'Uncorrected').sum()), 'graphst_representation_sensitivity_rows': int(((table.method == 'GraphST') & ~table.primary_representation).sum()), 'composite_score': 'prohibited_not_computed', 'winner_ranking': 'prohibited_not_computed', 'cross_section_exact_spot_retrieval': 'not_computed_not_a_batch_removal_endpoint', 'active_manuscript_claim': None, 'decision': 'atomic_metrics_complete_independent_validation_required'}
    summary_json_path = output_dir / 'summary.json'
    summary_json_path.write_text(json.dumps(summary_json, indent=2) + '\n', encoding='utf-8')
    sources = [Path(__file__), STUDY_ROOT / 'metrics.py', STUDY_ROOT / 'workflow.py', config['_config_path']]
    provenance = {'schema_version': 1, 'configuration_id': f"{config['configuration_id']}__metrics", 'generated_utc': datetime.now(timezone.utc).isoformat(), 'inputs': [{'path': str((run_dir / 'prepare_provenance.json').resolve())}, *[{'path': str((workflow.candidate_path(run_dir, c, m, s) / 'metadata.json').resolve())} for c, m, s in workflow.production_cells(config)]], 'sources': [{'path': str(path.resolve())} for path in sources], 'outputs': [{'path': str(path.resolve())} for path in (table_path, summary_path, summary_json_path)]}
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {len(table)} atomic rows and {len(summary)} summary rows')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
