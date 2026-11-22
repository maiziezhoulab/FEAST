"""Verify source artifacts and regenerate expression-free DevCCF blueprints."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import workflow

def prepare_age(config: dict, age: str) -> None:
    volume = config['_devccf_dir'] / config['ages'][age]['volume']
    schema_path = config['_devccf_dir'] / config['blueprint']['region_schema']
    output = workflow.blueprint_path(config, age)
    manifest_path = workflow.blueprint_manifest_path(config, age)
    provenance_path = workflow.blueprint_provenance_path(config, age)
    for path in (output, manifest_path, provenance_path):
        if path.exists():
            raise FileExistsError(f'refusing to overwrite prepared artifact: {path}')
    data, affine = workflow.read_nifti_gz(volume)
    schema = workflow.load_region_schema(schema_path)
    payload = workflow.extract_blueprint_payload(data, affine, schema, age)
    payload['metadata'].update({'blueprint_configuration_id': config['blueprint_configuration_id']})
    entries = workflow.validate_blueprint_payload(config, age, payload)
    workflow.write_blueprint(output, payload)
    frame = pd.DataFrame((workflow.blueprint_contract_row(age, entry) for entry in entries))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    provenance = {'schema_version': 1, 'blueprint_configuration_id': config['blueprint_configuration_id'], 'age': age, 'generated_utc': datetime.now(timezone.utc).isoformat(), 'expression_source': 'none', 'smoothing': 'none', 'n_z_levels': len(entries), 'n_spots': int(frame['n_spots'].sum()), 'execution_configuration_id_at_materialization': config['configuration_id']}
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"prepared {age}: {len(entries)} levels, {int(frame['n_spots'].sum()):,} spots")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.DEFAULT_CONFIG)
    parser.add_argument('--age', choices=(*workflow.AGE_ORDER, 'both'), default='both')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    summary = workflow.preflight_inputs(config, inspect_h5ad=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.preflight_only:
        return
    ages = workflow.AGE_ORDER if args.age == 'both' else (args.age,)
    targets = []
    for age in ages:
        output = workflow.blueprint_path(config, age)
        targets.extend((output, workflow.blueprint_manifest_path(config, age), workflow.blueprint_provenance_path(config, age)))
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f'refusing partial preparation because targets exist: {existing}')
    for age in ages:
        prepare_age(config, age)
if __name__ == '__main__':
    main()
