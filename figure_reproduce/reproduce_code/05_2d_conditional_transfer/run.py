"""Run one fresh Study 05 cross-slice or half-slice conditional job."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import anndata as ad
import numpy as np
import pandas as pd
from workflow import Job, STUDY_DIR, dataset_config, expected_job_path, input_artifact_id, inspect_inputs, job_id, prepare_job, require_declared_job, validate_generated

def cache_reference_counts_dense_float32(reference: ad.AnnData) -> dict:
    """Cache the immutable reference counts once for FEAST's empirical decoder.

    FEAST's public conditional path converts counts to dense float32 whenever
    it reads them.  Keeping that exact representation in the in-memory
    reference avoids repeating the same sparse-to-dense conversion for every
    label/gene without changing the values passed to FEAST.
    """
    if 'counts' not in reference.layers:
        raise KeyError('reference is missing the required counts layer')
    original = reference.layers['counts']
    original_type = f'{type(original).__module__}.{type(original).__name__}'
    if hasattr(original, 'toarray'):
        original = original.toarray()
    dense = np.asarray(original, dtype=np.float32)
    if dense.shape != reference.shape:
        raise AssertionError('cached reference counts shape changed')
    if not np.isfinite(dense).all():
        raise ValueError('reference counts must be finite before caching')
    reference.layers['counts'] = dense
    return {'strategy': 'one_time_dense_float32_public_input_cache', 'original_type': original_type, 'cached_dtype': str(dense.dtype), 'shape': [int(value) for value in dense.shape], 'value_transform': 'none'}

def feast_runtime() -> dict:
    import FEAST
    return {
        'version': str(FEAST.__version__),
        'import_path': str(Path(FEAST.__file__).resolve()),
        'source_revision': os.environ.get('FEAST_SOURCE_REVISION'),
        'interpreter': sys.executable,
        'python': sys.version.split()[0],
        'packages': {name: version(name) for name in ('numpy', 'scipy', 'POT', 'anndata', 'pandas', 'scikit-learn')},
        'threads': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cross_slice', 'mask_half'), required=True)
    parser.add_argument('--dataset', choices=('dlpfc', 'merfish'), required=True)
    parser.add_argument('--source')
    parser.add_argument('--target')
    parser.add_argument('--slice')
    parser.add_argument('--assignment-randomness', type=float)
    parser.add_argument('--config', type=Path, default=STUDY_DIR / 'config.yaml')
    parser.add_argument('--manifest', type=Path, default=STUDY_DIR / 'data' / 'input_files.csv')
    parser.add_argument('--input-dir', type=Path, default=STUDY_DIR / 'data' / 'local')
    parser.add_argument('--output-dir', type=Path, default=STUDY_DIR / 'outputs' / 'final')
    parser.add_argument('--verbose', action='store_true')
    return parser

def job_from_args(args: argparse.Namespace, config: dict) -> Job:
    if args.mode == 'cross_slice':
        if args.source is None or args.target is None or args.assignment_randomness is None:
            raise ValueError('cross_slice requires --source, --target, and --assignment-randomness')
        if args.slice is not None:
            raise ValueError('cross_slice does not accept --slice')
        job = Job('cross_slice', str(args.dataset), str(args.source), str(args.target), float(args.assignment_randomness))
    else:
        if args.slice is None or args.source is not None or args.target is not None:
            raise ValueError('mask_half requires --slice and does not accept source/target')
        randomness = float(config['mask_half']['assignment_randomness'])
        if args.assignment_randomness is not None and (not np.isclose(float(args.assignment_randomness), randomness, rtol=0.0, atol=1e-12)):
            raise ValueError('mask_half assignment randomness differs from the frozen value')
        job = Job('mask_half', str(args.dataset), str(args.slice), str(args.slice), randomness)
    require_declared_job(config, job)
    return job

def main() -> int:
    args = build_parser().parse_args()
    config, paths, panels = inspect_inputs(args.config, args.manifest, args.input_dir)
    job = job_from_args(args, config)
    final_dir = expected_job_path(args.output_dir, job)
    if final_dir.exists():
        raise FileExistsError(f'fresh-output contract forbids replacing {final_dir}')
    runtime = feast_runtime()
    if runtime['version'] != str(config['required_feast_version']):
        raise RuntimeError('imported FEAST version differs from config.yaml')
    from FEAST import __version__ as feast_version
    from FEAST.de_novo import ReferenceFitConfig, SimulationBlueprint, SimulationConfig, fit_reference, simulate_from_reference
    prepared = prepare_job(config, paths, panels, job)
    reference = prepared.reference
    reference_counts_cache = cache_reference_counts_dense_float32(reference)
    target = prepared.target_contract
    genes = prepared.genes
    support = prepared.support
    entry = dataset_config(config, job.dataset)
    annotation_key = str(entry['annotation_key'])
    spot_id_column = str(config['blueprint']['spot_id_column'])
    target_labels = target.obs[annotation_key].astype(str).to_numpy()
    blueprint_obs = pd.DataFrame({spot_id_column: target.obs_names.astype(str).to_numpy(), annotation_key: target_labels})
    blueprint = SimulationBlueprint(coordinates=np.asarray(target.obsm['spatial'], dtype=np.float64), domain_map=target_labels, obs=blueprint_obs, metadata={'artifact_id': input_artifact_id(config, job.dataset, job.target), 'expression_policy': config['target_expression_policy'], 'mode': job.mode, 'dataset': job.dataset})
    fit = config['reference_fit']
    fit_config = ReferenceFitConfig(min_gene_spots=int(fit['min_gene_spots']), min_gene_mean=float(fit['min_gene_mean']), max_gene_zero_prop=float(fit['max_gene_zero_prop']), boundary_neighbors=int(fit['boundary_neighbors']))
    transport = config['transport']
    simulation_config = SimulationConfig(epsilon=float(transport['epsilon']), sinkhorn_iter=int(transport['sinkhorn_iter']), sinkhorn_tol=float(transport['sinkhorn_tol']), unbalanced_transport=bool(transport['unbalanced_transport']), reg_m=float(transport['reg_m']), transport_nonconvergence=str(transport['transport_nonconvergence']), geometry_weight=float(transport['geometry_weight']), boundary_weight=float(transport['boundary_weight']), assignment_randomness=job.assignment_randomness, latent_clip_eps=float(transport['latent_clip_eps']), gene_chunk_size=int(transport['gene_chunk_size']), max_transport_pairs=int(transport['max_transport_pairs']), verbose=bool(args.verbose))
    work_root = args.output_dir.parent / '.work'
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f'{job_id(config, job)}-', dir=work_root))
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    model = fit_reference(reference, annotation_key, fit_config)
    if list(map(str, model.gene_names)) != genes:
        raise RuntimeError('reference fitting did not retain the exact frozen dataset panel')
    generated = simulate_from_reference(model, blueprint, config=simulation_config, random_seed=int(config['public_seed']), marginal_model='empirical_reference')
    validate_generated(generated, target, genes, config, job.dataset, require_bound_obs_names=False)
    generated.obs_names = target.obs_names.copy()
    records = validate_generated(generated, target, genes, config, job.dataset)
    configuration_id = job_id(config, job)
    source_path = paths[job.dataset, job.source]
    target_path = paths[job.dataset, job.target]
    generated.uns['feast_reproduce'] = {'configuration_id': configuration_id, 'study_configuration_id': str(config['configuration_id']), 'mode': job.mode, 'dataset': job.dataset, 'annotation_key': annotation_key, 'feast_version': str(feast_version), 'public_seed': int(config['public_seed']), 'source_artifact_id': input_artifact_id(config, job.dataset, job.source), 'target_artifact_id': input_artifact_id(config, job.dataset, job.target), 'assignment_randomness': job.assignment_randomness, 'primary_setting': bool(np.isclose(job.assignment_randomness, float(config['primary_assignment_randomness']))), 'target_expression_policy': str(config['target_expression_policy']), 'target_expression_loaded_by_generation': False, 'identity_verified_before_obs_name_rebinding': True, 'reference_counts_cache': reference_counts_cache, 'positive_transport_records': int(len(records))}
    generated_path = work_dir / 'generated.h5ad'
    diagnostics_path = work_dir / 'transport_diagnostics.csv'
    generated.write_h5ad(generated_path, compression='gzip')
    pd.DataFrame(records).to_csv(diagnostics_path, index=False)
    restored = ad.read_h5ad(generated_path)
    validate_generated(restored, target, genes, config, job.dataset)
    provenance = {'status': 'ok', 'configuration_id': configuration_id, 'study_configuration_id': config['configuration_id'], 'mode': job.mode, 'dataset': job.dataset, 'annotation_key': annotation_key, 'feast_version': str(feast_version), 'public_seed': int(config['public_seed']), 'source': job.source, 'target': job.target, 'assignment_randomness': job.assignment_randomness, 'primary_setting': bool(np.isclose(job.assignment_randomness, float(config['primary_assignment_randomness']))), 'target_expression_policy': config['target_expression_policy'], 'target_expression_loaded_by_generation': False, 'identity_verified_before_obs_name_rebinding': True, 'reference_counts_cache': reference_counts_cache, 'input_artifacts': {'source': {'artifact_id': input_artifact_id(config, job.dataset, job.source)}, 'target': {'artifact_id': input_artifact_id(config, job.dataset, job.target)}}, 'ordered_gene_panel': {'n_genes': len(genes)}, 'support': support, 'sources': {}, 'transport': transport, 'positive_transport_records': len(records), 'numerical_change': {'field': 'sinkhorn_iter', 'historical': 200, 'fresh': int(transport['sinkhorn_iter']), 'classification': 'declared_solver_configuration_change'}, 'runtime': runtime, 'started_at': started_at.isoformat(), 'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started, 'outputs': {'generated.h5ad': None, 'transport_diagnostics.csv': None}}
    provenance['argv'] = sys.argv
    provenance['simulation_config'] = asdict(simulation_config)
    (work_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f'fresh-output contract forbids replacing {final_dir}')
    os.replace(work_dir, final_dir)
    print(json.dumps({'status': 'ok', 'output_dir': str(final_dir)}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
