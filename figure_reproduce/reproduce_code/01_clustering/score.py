"""Validate and score all fresh Study 01 clustering outputs."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score, v_measure_score
from sklearn.preprocessing import StandardScaler
METHODS = ('GraphST', 'STAGATE_mclust', 'Leiden_unsupervised')
METRICS = ('ARI', 'NMI', 'AMI', 'Homogeneity', 'Completeness', 'V_measure', 'CHAOS', 'PAS')

def spatial_cache(spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = squareform(pdist(StandardScaler().fit_transform(spatial)))
    neighbors = np.argsort(distances, axis=1)[:, 1:11]
    return (distances, neighbors)

def metrics(truth: np.ndarray, labels: np.ndarray, distances: np.ndarray, neighbors: np.ndarray) -> dict:
    chaos = 0.0
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        if len(members) > 1:
            within = distances[np.ix_(members, members)].copy()
            np.fill_diagonal(within, np.inf)
            chaos += float(np.min(within, axis=1).sum())
    pas = np.mean([np.sum(labels[row] != labels[index]) > len(row) / 2 for index, row in enumerate(neighbors)])
    return {'ARI': adjusted_rand_score(truth, labels), 'NMI': normalized_mutual_info_score(truth, labels), 'AMI': adjusted_mutual_info_score(truth, labels), 'Homogeneity': homogeneity_score(truth, labels), 'Completeness': completeness_score(truth, labels), 'V_measure': v_measure_score(truth, labels), 'CHAOS': chaos / len(labels), 'PAS': float(pas), 'n_true_clusters': int(pd.Series(truth).nunique()), 'n_predicted_clusters': int(pd.Series(labels).nunique())}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-dir', type=Path, required=True)
    parser.add_argument('--method-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    selected_methods = tuple(args.methods)
    manifest = pd.read_csv(args.panel_dir / 'fixed_panel_manifest.csv', dtype={'slice_id': str, 'simulation_id': str})
    if len(manifest) != 84:
        raise RuntimeError('expected 84 fixed-panel rows')
    panel_provenance = json.loads((args.panel_dir / 'provenance.json').read_text())
    feast_version = panel_provenance['feast_version']
    public_seed = int(panel_provenance['public_seed'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_manifests = {}
    for method in selected_methods:
        run = pd.read_csv(args.method_dir / method / 'run_manifest.csv', dtype={'slice_id': str, 'simulation_id': str})
        if len(run) != 84 or not run.status.eq('ok').all():
            raise RuntimeError(f'{method} must have 84 successful fresh jobs')
        run_manifests[method] = run.set_index(['slice_id', 'simulation_id'])
    rows = []
    spatial_by_slice: dict[str, tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]] = {}
    for job_index, row in enumerate(manifest.itertuples(index=False), start=1):
        panel_path = args.panel_dir / row.file_path
        data = ad.read_h5ad(panel_path)
        truth = data.obs['ground_truth'].astype(str).to_numpy()
        coordinates = np.asarray(data.obsm['spatial'])
        if row.slice_id not in spatial_by_slice:
            distances, neighbors = spatial_cache(coordinates)
            spatial_by_slice[row.slice_id] = (data.obs_names.copy(), coordinates.copy(), distances, neighbors)
        else:
            expected_names, expected_coordinates, distances, neighbors = spatial_by_slice[row.slice_id]
            if not data.obs_names.equals(expected_names) or not np.array_equal(coordinates, expected_coordinates):
                raise RuntimeError(f'spot geometry changed within slice {row.slice_id}')
        for method in selected_methods:
            output = args.method_dir / method / row.slice_id / row.simulation_id
            cluster_path, metadata_path, result_path = (output / 'clusters.csv', output / 'metadata.json', output / 'result.h5ad')
            run = run_manifests[method].loc[row.slice_id, row.simulation_id]
            clusters = pd.read_csv(cluster_path, dtype={'spot_barcode': str}).set_index('spot_barcode')
            if not clusters.index.equals(data.obs_names):
                raise RuntimeError(f'cluster spot order mismatch: {cluster_path}')
            result = ad.read_h5ad(result_path, backed='r')
            try:
                if not result.obs_names.equals(data.obs_names):
                    raise RuntimeError(f'result spot order mismatch: {result_path}')
            finally:
                result.file.close()
            diagnostics = json.loads(metadata_path.read_text())
            if run.feast_version != feast_version or int(run.public_seed) != public_seed:
                raise RuntimeError(f'method manifest FEAST version or seed mismatch: {result_path}')
            if int(diagnostics.get('public_seed', -1)) != public_seed:
                raise RuntimeError(f'method seed mismatch: {metadata_path}')
            if method == 'Leiden_unsupervised':
                if diagnostics.get('selection_uses_ground_truth') is not False:
                    raise RuntimeError('Leiden selection is not label-free')
                label_informed_fields = {'ari', 'nmi', 'ami', 'ground_truth', 'adjusted_rand_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score'}
                found = {key for item in diagnostics.get('sweep', []) for key in item if key.casefold() in label_informed_fields}
                if found:
                    raise RuntimeError(f'label-informed Leiden diagnostics: {found}')
            labels = clusters['predicted_cluster'].astype(str).to_numpy()
            rows.append({'slice_id': row.slice_id, 'simulation_id': row.simulation_id, 'alteration_type': row.alteration_type, 'method': method, 'feast_version': feast_version, 'public_seed': public_seed, **metrics(truth, labels, distances, neighbors), 'status': 'ok'})
        print(f'[{job_index}/84] {row.slice_id}/{row.simulation_id}', flush=True)
    table = pd.DataFrame(rows)
    table_path = args.output_dir / 'fixed_panel_benchmark_metrics.csv'
    table.to_csv(table_path, index=False)
    summary_rows = []
    groups = {'all_simulations': table.simulation_id != 'real_baseline', 'altered_simulations': ~table.simulation_id.isin(['baseline', 'real_baseline']), 'feast_baseline': table.simulation_id == 'baseline', 'real_baseline': table.simulation_id == 'real_baseline'}
    for name, mask in groups.items():
        group = table.loc[mask].groupby('method', as_index=False)[list(METRICS)].mean()
        group.insert(0, 'group', name)
        group.insert(2, 'n_rows', table.loc[mask].groupby('method').size().to_numpy())
        summary_rows.append(group)
    summary = pd.concat(summary_rows, ignore_index=True)
    summary_path = args.output_dir / 'benchmark_summary.csv'
    summary.to_csv(summary_path, index=False)
    baseline = summary[summary.group.isin(['feast_baseline', 'real_baseline'])].pivot(index='method', columns='group', values=['ARI', 'NMI', 'AMI'])
    differences = pd.DataFrame([{'method': method, **{f'{metric}_feast_minus_real': baseline.loc[method, (metric, 'feast_baseline')] - baseline.loc[method, (metric, 'real_baseline')] for metric in ('ARI', 'NMI', 'AMI')}} for method in selected_methods])
    differences_path = args.output_dir / 'baseline_differences.csv'
    differences.to_csv(differences_path, index=False)
    provenance = {'configuration_id': panel_provenance['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'public_seed': panel_provenance['public_seed'], 'feast_version': panel_provenance['feast_version'], 'methods': list(selected_methods), 'row_count': len(table), 'selection_uses_ground_truth_for_leiden': False, 'scoring_source': str(Path(__file__).resolve()), 'outputs': {path.name: None for path in (table_path, summary_path, differences_path)}}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
