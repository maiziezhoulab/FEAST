"""Strictly validate complete E15.5/E18.5 final volumes and write a decision."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import pandas as pd
import workflow
import run as execution

def validate_age(config: dict, age: str) -> dict:
    _, entries = workflow.load_blueprint(config, age)
    final = config['_final_dir'] / age
    manifest_path = final / 'manifest.csv'
    provenance_path = final / 'provenance.json'
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f'{age} is not consolidated: {final}')
    manifest = pd.read_csv(manifest_path)
    expected_indices = list(range(len(entries)))
    if manifest['z_index'].astype(int).tolist() != expected_indices:
        raise ValueError(f'{age}: manifest z indices are not complete and contiguous')
    observed_files = sorted((path.name for path in final.glob('*.h5ad')))
    if observed_files != sorted(manifest['filename'].astype(str)):
        raise ValueError(f'{age}: undeclared or missing final H5AD files')
    reference = ad.read_h5ad(workflow.reference_paths(config, age)[0], backed='r')
    try:
        genes = sorted(map(str, reference.var_names))
    finally:
        reference.file.close()
    max_error = 0.0
    transport_records = 0
    ar, _ = execution.assignment_randomness(config, age)
    transport_contract = workflow.frozen_transport_config(config, ar)
    reference_artifacts = workflow.reference_artifact_contract(config, age)
    for row, entry in zip(manifest.itertuples(index=False), entries):
        path = final / str(row.filename)
        result = ad.read_h5ad(path)
        summary = workflow.validate_result_identity(result, age, entry, genes, transport_contract)
        workflow.validate_publication_lineage(result, config, age, entry, transport_contract, reference_artifacts)
        max_error = max(max_error, float(summary['max_final_error']))
        transport_records += int(summary['transport_records'])
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('configuration_id') != config['configuration_id']:
        raise ValueError(f'{age}: consolidated provenance configuration mismatch')
    return {'age': age, 'status': 'validated', 'n_z_levels': len(entries), 'n_spots': sum((int(entry['n_spots']) for entry in entries)), 'z_start': float(entries[0]['z_world']), 'z_end': float(entries[-1]['z_world']), 'transport_records': transport_records, 'max_final_error': max_error, 'all_transport_records_converged': True, 'exact_blueprint_identity': True}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.DEFAULT_CONFIG)
    parser.add_argument('--age', choices=(*workflow.AGE_ORDER, 'both'), default='both')
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    workflow.preflight_inputs(config, inspect_h5ad=True)
    feast = workflow.feast_identity(config)
    ages = workflow.AGE_ORDER if args.age == 'both' else (args.age,)
    results = [validate_age(config, age) for age in ages]
    decision = {'schema_version': 1, 'configuration_id': config['configuration_id'], 'generated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'validated' if len(results) == 2 else 'age_validated_only', 'publication_canonical': False, 'publication_status': 'candidate_identity_and_convergence_gate_passed_pending_article_metrics_visualization_and_author_review' if len(results) == 2 else 'single_age_gate_only', 'feast': feast, 'ages': results}
    decision_path = config['_final_dir'] / 'validation.json'
    if decision_path.exists():
        raise FileExistsError(f'refusing to overwrite validation decision: {decision_path}')
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(decision, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
