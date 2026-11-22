"""Validate unique shard outputs and hardlink one immutable final age volume."""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import pandas as pd
import workflow
import run as execution

def reference_genes(config: dict, age: str) -> list[str]:
    backed = ad.read_h5ad(workflow.reference_paths(config, age)[0], backed='r')
    try:
        return sorted(map(str, backed.var_names))
    finally:
        backed.file.close()

def collect_candidates(config: dict, age: str) -> dict[int, tuple[Path, dict]]:
    candidates: dict[int, tuple[Path, dict]] = {}
    age_root = config['_work_dir'] / age
    declared_paths: set[Path] = set()
    for record_path in sorted(age_root.glob('*/*.record.json')):
        row = json.loads(record_path.read_text(encoding='utf-8'))
        index = int(row['z_index'])
        if index in candidates:
            raise ValueError(f'duplicate active candidate for {age} z-index {index}')
        path = record_path.parent / str(row['filename'])
        if not path.is_file():
            raise FileNotFoundError(f'recorded candidate is missing: {path}')
        candidates[index] = (path, row)
        declared_paths.add(path.resolve())
    unrecorded = [path for path in age_root.glob('*/*.h5ad') if path.resolve() not in declared_paths and (not path.name.endswith('.tmp.h5ad'))]
    if unrecorded:
        raise ValueError(f'unrecorded shard H5ADs must be archived before consolidation: {unrecorded}')
    return candidates

def consolidate_age(config: dict, age: str) -> None:
    workflow.preflight_inputs(config, inspect_h5ad=True)
    feast = workflow.feast_identity(config)
    _, entries = workflow.load_blueprint(config, age)
    genes = reference_genes(config, age)
    ar, _ = execution.assignment_randomness(config, age)
    transport_contract = workflow.frozen_transport_config(config, ar)
    reference_artifacts = workflow.reference_artifact_contract(config, age)
    candidates = collect_candidates(config, age)
    expected = set(range(len(entries)))
    if set(candidates) != expected:
        missing = sorted(expected - set(candidates))
        extra = sorted(set(candidates) - expected)
        raise ValueError(f'{age} shard coverage incomplete; missing={missing}, extra={extra}')
    final = config['_final_dir'] / age
    staging = config['_work_dir'] / f'consolidating-{age}'
    if final.exists() or staging.exists():
        raise FileExistsError(f'refusing to overwrite final/staging directory: {final}, {staging}')
    staging.mkdir(parents=True)
    rows = []
    try:
        for index in range(len(entries)):
            source, declared = candidates[index]
            result = ad.read_h5ad(source)
            summary = workflow.validate_result_identity(result, age, entries[index], genes, transport_contract)
            workflow.validate_publication_lineage(result, config, age, entries[index], transport_contract, reference_artifacts)
            provenance = result.uns['publication_reproduction']
            filename = f"z{index:03d}__{float(entries[index]['z_world']):+.3f}.h5ad"
            destination = staging / filename
            os.link(source, destination)
            rows.append({**summary, 'configuration_id': config['configuration_id'], 'filename': filename, 'public_seed': int(config['public_seed']) + index, 'assignment_randomness': float(provenance['assignment_randomness'])})
        manifest = pd.DataFrame(rows).sort_values('z_index')
        manifest.to_csv(staging / 'manifest.csv', index=False)
        metadata = {'schema_version': 1, 'configuration_id': config['configuration_id'], 'age': age, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'n_z_levels': len(entries), 'n_spots': int(manifest['n_spots'].sum()), 'z_start': float(manifest.iloc[0]['z_world']), 'z_end': float(manifest.iloc[-1]['z_world']), 'reference_artifacts': reference_artifacts, 'transport_config': transport_contract, 'feast': feast, 'smoothing': 'none', 'all_transport_records_converged': True}
        (staging / 'provenance.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    except Exception:
        raise
    print(f'consolidated {age}: {len(entries)} slices -> {final}')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.DEFAULT_CONFIG)
    parser.add_argument('--age', choices=workflow.AGE_ORDER, required=True)
    args = parser.parse_args()
    consolidate_age(workflow.load_config(args.config), args.age)
if __name__ == '__main__':
    main()
