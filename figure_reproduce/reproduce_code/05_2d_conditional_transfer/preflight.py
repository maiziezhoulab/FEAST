"""Verify all five Study 05 inputs, panels, and expression-free job supports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from workflow import STUDY_DIR, declared_jobs, input_artifact_id, inspect_inputs, prepare_job

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY_DIR / 'config.yaml')
    parser.add_argument('--manifest', type=Path, default=STUDY_DIR / 'data' / 'input_files.csv')
    parser.add_argument('--input-dir', type=Path, default=STUDY_DIR / 'data' / 'local')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir, check_counts=True)
    jobs = declared_jobs(config)
    structural_jobs = {}
    for job in jobs:
        key = (job.mode, job.dataset, job.source, job.target)
        structural_jobs.setdefault(key, job)
    support = {}
    for key, job in structural_jobs.items():
        prepared = prepare_job(config, paths, panels, job)
        support['/'.join(key)] = prepared.support
    report = {'status': 'ok', 'configuration_id': config['configuration_id'], 'inputs': {f'{dataset}/{slice_id}': {'artifact_id': input_artifact_id(config, dataset, slice_id)} for (dataset, slice_id), path in sorted(paths.items())}, 'ordered_gene_panels': {dataset: {'n_genes': len(genes)} for dataset, genes in panels.items()}, 'declared_jobs': len(jobs), 'cross_slice_jobs': sum((job.mode == 'cross_slice' for job in jobs)), 'mask_half_jobs': sum((job.mode == 'mask_half' for job in jobs)), 'structural_supports': support, 'target_expression_policy': config['target_expression_policy']}
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
