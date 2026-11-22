"""Build one raw-derived fixed gene panel per Study 01 slice."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

def select_panel(raw: ad.AnnData, n_top: int, flavor: str) -> pd.Index:
    if not raw.var_names.is_unique:
        raise ValueError('raw slice gene identifiers must be unique')
    if raw.n_vars <= n_top:
        return raw.var_names.copy()
    candidate = raw.copy()
    sc.pp.highly_variable_genes(candidate, n_top_genes=n_top, flavor=flavor)
    selected = candidate.var_names[candidate.var['highly_variable'].to_numpy()]
    if len(selected) != n_top:
        raise RuntimeError(f'expected {n_top} HVGs, selected {len(selected)}')
    return selected

def prepare(data: ad.AnnData, genes: pd.Index, target_sum: float) -> ad.AnnData:
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError('spot and gene identifiers must be unique')
    missing = genes.difference(data.var_names)
    if len(missing):
        raise ValueError(f'fixed panel is missing {len(missing)} genes')
    result = data[:, genes].copy()
    sc.pp.normalize_total(result, target_sum=target_sum)
    sc.pp.log1p(result)
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--simulation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    simulation_manifest = pd.read_csv(args.simulation_dir / 'simulation_manifest.csv', dtype={'slice_id': str})
    if len(simulation_manifest) != 81 or not simulation_manifest.status.eq('ok').all():
        raise RuntimeError('expected 81 successful fresh simulation rows')
    feast_version = str(config['feast']['version'])
    required_simulation_contract = simulation_manifest['configuration_id'].eq(config['configuration_id']).all() and simulation_manifest['public_seed'].eq(config['public_seed']).all() and simulation_manifest['feast_version'].astype(str).eq(feast_version).all() and simulation_manifest['spatial_mode'].eq('reference_rank').all() and simulation_manifest['assignment_solver'].eq('scipy').all() and simulation_manifest['assignment_blocks'].eq(False).all()
    if not required_simulation_contract:
        raise RuntimeError('simulation manifest does not match the fixed publication contract')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = config['fixed_panel']
    rows = []
    for slice_value in config['slices']:
        slice_id = str(slice_value)
        raw_path = args.raw_dir / f'{slice_id}.h5ad'
        raw = ad.read_h5ad(raw_path)
        genes = select_panel(raw, int(settings['n_top_genes']), settings['flavor'])
        panel_path = args.output_dir / 'panels' / f'{slice_id}.csv'
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'gene_id': genes}).to_csv(panel_path, index=False)
        slice_rows = simulation_manifest[simulation_manifest.slice_id == slice_id]
        candidates = [(row.simulation_id, args.simulation_dir / row.file_path, row.alteration_type) for row in slice_rows.itertuples(index=False)]
        candidates.append(('real_baseline', raw_path, 'real_baseline'))
        for simulation_id, source, alteration_type in candidates:
            source_data = raw if simulation_id == 'real_baseline' else ad.read_h5ad(source)
            prepared = prepare(source_data, genes, float(settings['target_sum']))
            prepared.uns['publication_reproduction'] = {'configuration_id': config['configuration_id'], 'feast_version': feast_version, 'public_seed': int(config['public_seed']), 'source_type': 'raw_real_baseline' if simulation_id == 'real_baseline' else 'fresh_feast_simulation', 'panel_selection': 'raw-derived Seurat-v3', 'preprocessing': f"normalize_total(target_sum={settings['target_sum']}) then log1p"}
            relative = Path('inputs') / slice_id / f'{simulation_id}.h5ad'
            output = args.output_dir / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_h5ad(output, compression='gzip')
            rows.append({'slice_id': slice_id, 'simulation_id': simulation_id, 'alteration_type': alteration_type, 'file_path': relative.as_posix(), 'n_spots': prepared.n_obs, 'n_genes': prepared.n_vars, 'status': 'ok'})
        print(f'{slice_id}: {len(candidates)} fixed-panel inputs', flush=True)
    manifest_path = args.output_dir / 'fixed_panel_manifest.csv'
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    provenance = {'configuration_id': config['configuration_id'], 'feast_version': feast_version, 'public_seed': config['public_seed'], 'panel_rule': f"raw-derived {settings['flavor']} top-{settings['n_top_genes']} per slice", 'preprocessing': f"normalize_total(target_sum={settings['target_sum']}) then log1p exactly once", 'job_count': len(rows)}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
