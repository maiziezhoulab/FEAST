"""Inspect the configured reference cohort, reused geometry and runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
from matplotlib import font_manager

import run
import workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=workflow.DEFAULT_CONFIG)
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    summary = workflow.preflight_inputs(config)
    summary['configuration_id'] = config['configuration_id']
    summary['feast'] = workflow.feast_identity(config)
    summary['blueprints'] = {}
    for age in workflow.AGE_ORDER:
        _, entries = workflow.load_blueprint(config, age)
        summary['blueprints'][age] = {
            'levels': len(entries), 'spots': sum(entry['n_spots'] for entry in entries),
            'canary_indices': run.canary_indices(age, entries),
        }
        for row in summary['ages'][age]:
            data = ad.read_h5ad(config['_reference_dir'] / row['file'], backed='r')
            try:
                row['regions'] = sorted(set(data.obs[config['reference_fit']['label_key']].astype(str)))
                if not workflow.EXPECTED_REGIONS <= set(row['regions']):
                    raise ValueError(f"{row['file']}: missing blueprint modeling regions")
            finally:
                data.file.close()
    summary['fonts'] = []
    for filename in ('arial.ttf', 'arialbd.ttf'):
        path = Path(sys.prefix) / 'fonts' / filename
        if not path.is_file():
            raise FileNotFoundError(f'Arial font is missing: {path}')
        font_manager.fontManager.addfont(str(path))
        summary['fonts'].append(str(path))
    output = config['_final_dir'] / 'preflight.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
