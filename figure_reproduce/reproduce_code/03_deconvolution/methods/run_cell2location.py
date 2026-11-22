"""Run one Cell2location job on the exact fresh FEAST simulation."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import traceback
import warnings
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
LIMITATION = 'This is an external method environment and may use a NumPy version unsupported by FEAST. FEAST is not imported or executed in this process.'

def integer_counts(matrix, label: str) -> np.ndarray:
    values = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or values.min(initial=0) < 0:
        raise RuntimeError(f'{label} has invalid count values')
    if np.max(np.abs(values - np.rint(values)), initial=0) > 1e-06:
        raise RuntimeError(f'{label} is not raw integer count data')
    return np.rint(values).astype(np.int64)

def loss_summary(model, expected_epochs: int, label: str) -> dict:
    history = getattr(model, 'history', None)
    if not isinstance(history, dict) or 'elbo_train' not in history:
        raise RuntimeError(f'{label} lacks ELBO history')
    values = np.asarray(history['elbo_train'], dtype=np.float64).reshape(-1)
    if len(values) != expected_epochs or not np.isfinite(values).all():
        raise RuntimeError(f'{label} ELBO history is incomplete or non-finite')
    return {'metric': 'elbo_train', 'epochs_expected': expected_epochs, 'epochs_observed': len(values), 'all_finite': True, 'initial': float(values[0]), 'final': float(values[-1]), 'minimum': float(values.min()), 'maximum': float(values.max()), 'interpretation': 'fixed_epoch_schedule_complete; not a convergence claim'}

def warning_rows(captured) -> list[dict]:
    rows, seen = ([], set())
    for item in captured:
        key = (item.category.__name__, str(item.message), str(item.filename), int(item.lineno))
        if key not in seen:
            seen.add(key)
            rows.append(dict(zip(('category', 'message', 'filename', 'lineno'), key)))
    return rows

def normalize_abundance_columns(abundance: pd.DataFrame, abundance_key: str, factor_names: list[str]) -> tuple[pd.DataFrame, str]:
    """Validate Cell2location's named abundance columns and remove its prefix."""
    if not isinstance(abundance, pd.DataFrame):
        raise RuntimeError('posterior abundance lacks named cell-type columns')
    expected = pd.Index(map(str, factor_names))
    if not expected.is_unique:
        raise RuntimeError('reference factor names are not unique')
    actual = pd.Index(map(str, abundance.columns))
    if not actual.is_unique:
        raise RuntimeError('posterior abundance columns are not unique')
    if actual.equals(expected):
        column_schema = 'factor_names_exact_order'
    else:
        suffix = '_cell_abundance_w_sf'
        if not abundance_key.endswith(suffix):
            raise RuntimeError(f'cannot derive Cell2location column prefix from {abundance_key!r}')
        summary_name = abundance_key[:-len(suffix)]
        prefixed = pd.Index((f'{summary_name}cell_abundance_w_sf_{name}' for name in expected))
        if not actual.equals(prefixed):
            raise RuntimeError(f'posterior abundance named cell-type support/order differs: expected {len(prefixed)} exact prefixed columns, observed {len(actual)}')
        column_schema = 'cell2location_prefixed_exact_order'
    normalized = abundance.copy()
    normalized.columns = expected
    return (normalized, column_schema)

def labels_retained_by_minimum_size(labels: pd.Series, minimum_size: int | None) -> tuple[pd.Index, np.ndarray]:
    """Return exact named reference-label support for a declared fair comparison."""
    label_counts = labels.value_counts()
    if minimum_size is None:
        retained = pd.Index(label_counts.index.astype(str))
    else:
        if minimum_size < 1:
            raise ValueError('minimum cell-type size must be positive')
        retained = pd.Index(label_counts[label_counts >= minimum_size].index.astype(str))
    keep = labels.astype(str).isin(retained).to_numpy()
    if retained.empty or not keep.any():
        raise RuntimeError('reference filtering left no cell types')
    return (retained, keep)

def abundance_total_summary(abundance: np.ndarray) -> dict[str, float]:
    totals = abundance.sum(axis=1)
    return {'min': float(totals.min()), 'median': float(np.median(totals)), 'mean': float(totals.mean()), 'max': float(totals.max())}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell-type-key', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--min-cells-per-type', type=int, help='Optionally retain only reference labels with at least this many cells.')
    parser.add_argument('--positive-library-only', action='store_true', help='Fit only spots with nonzero library size and reinsert zero rows in the output.')
    parser.add_argument('--a-factors-per-location', type=float)
    parser.add_argument('--b-groups-per-location', type=float)
    args = parser.parse_args()
    metadata_path = args.output.with_name(args.output.stem + '_metadata.json')
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import scanpy as sc
        import scvi
        import torch
        from cell2location.models import Cell2location, RegressionModel
        config = yaml.safe_load(args.config.read_text())['cell2location']
        accelerator = str(config['accelerator'])
        if accelerator == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA was requested but is not available')
            torch.cuda.set_device(0)
            torch.cuda.init()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cuda_device_index = int(torch.cuda.current_device())
            actual_device = f'cuda:{cuda_device_index}'
            cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))
            train_accelerator = 'gpu'
        elif accelerator == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            actual_device = 'cpu'
            cuda_device_name = None
            train_accelerator = 'cpu'
        else:
            raise RuntimeError(f'unsupported accelerator: {accelerator}')
        scvi.settings.seed = args.seed
        spatial_full = ad.read_h5ad(args.input)
        reference = ad.read_h5ad(args.reference)
        if args.cell_type_key not in reference.obs:
            raise RuntimeError(f'reference lacks {args.cell_type_key!r}')
        if not spatial_full.var_names.equals(reference.var_names):
            raise RuntimeError('simulation/reference gene identity or order differs')
        spatial_counts_full = integer_counts(spatial_full.X, 'spatial simulation')
        reference_counts = integer_counts(reference.X, 'reference')
        if 'n_source_cells' not in spatial_full.obs:
            raise RuntimeError('simulation lacks the declared source-cell count per aggregate location')
        source_cells_full = spatial_full.obs['n_source_cells'].to_numpy(np.float64)
        if not np.isfinite(source_cells_full).all() or np.any(source_cells_full < 0) or np.max(np.abs(source_cells_full - np.rint(source_cells_full)), initial=0) > 1e-06:
            raise RuntimeError('source-cell counts per location are invalid')
        full_spots = pd.Index([f'spot_{index}' for index in range(spatial_full.n_obs)])
        positive_mask = spatial_counts_full.sum(axis=1) > 0
        training_mask = positive_mask if args.positive_library_only else np.ones(len(full_spots), dtype=bool)
        if not training_mask.any():
            raise RuntimeError('no spots were selected for Cell2location training')
        training_spots = full_spots[training_mask]
        spatial = spatial_full[training_mask].copy()
        spatial_counts = spatial_counts_full[training_mask]
        source_cells = source_cells_full[training_mask]
        n_cells_per_location = float(source_cells.mean())
        if n_cells_per_location <= 0:
            raise RuntimeError('mean source-cell count per location is not positive')
        labels_full = reference.obs[args.cell_type_key].astype(str)
        retained_types, reference_keep = labels_retained_by_minimum_size(labels_full, args.min_cells_per_type)
        reference = reference[reference_keep].copy()
        reference_counts = reference_counts[reference_keep]
        if len(reference) != len(reference_counts):
            raise RuntimeError('reference filter did not preserve count rows')
        if (args.a_factors_per_location is None) != (args.b_groups_per_location is None):
            raise ValueError('A_factors_per_location and B_groups_per_location must be set together')
        if args.a_factors_per_location is not None and (args.a_factors_per_location <= 0 or args.b_groups_per_location <= 0):
            raise ValueError('Cell2location factor/group priors must be positive')
        reference.X = reference_counts
        reference.obs['_cell_type'] = reference.obs[args.cell_type_key].astype(str)
        reference.obs['_batch'] = 'all'
        RegressionModel.setup_anndata(reference, batch_key='_batch', labels_key='_cell_type')
        reference_model = RegressionModel(reference)
        with warnings.catch_warnings(record=True) as reference_warnings:
            warnings.simplefilter('default')
            reference_model.train(max_epochs=int(config['reference_epochs']), batch_size=int(config['reference_batch_size']), train_size=1, lr=float(config['reference_learning_rate']), accelerator=train_accelerator, device='auto')
            reference_loss = loss_summary(reference_model, int(config['reference_epochs']), 'RegressionModel')
            reference = reference_model.export_posterior(reference, sample_kwargs={'num_samples': int(config['reference_posterior_samples']), 'batch_size': int(config['reference_batch_size'])})
        if 'means_per_cluster_mu_fg' not in reference.varm:
            raise RuntimeError('reference posterior lacks means_per_cluster_mu_fg')
        factor_names = list(map(str, reference.uns['mod']['factor_names']))
        signatures = reference.varm['means_per_cluster_mu_fg'].iloc[:, :len(factor_names)].copy()
        signatures.columns = factor_names
        expected_types = set(reference.obs['_cell_type'].astype(str))
        if set(factor_names) != expected_types:
            raise RuntimeError('reference signature cell-type support differs')
        spatial.X = spatial_counts.astype(np.float32)
        gene_mask = spatial_counts.sum(axis=0) > 0
        spatial = spatial[:, gene_mask].copy()
        signatures = signatures.loc[spatial.var_names]
        spatial.obs['_batch'] = 'all'
        Cell2location.setup_anndata(spatial, batch_key='_batch')
        model_kwargs = {'N_cells_per_location': n_cells_per_location, 'detection_alpha': float(config['detection_alpha'])}
        if args.a_factors_per_location is not None:
            model_kwargs.update({'A_factors_per_location': float(args.a_factors_per_location), 'B_groups_per_location': float(args.b_groups_per_location)})
        model = Cell2location(spatial, cell_state_df=signatures, **model_kwargs)
        with warnings.catch_warnings(record=True) as spatial_warnings:
            warnings.simplefilter('default')
            model.train(max_epochs=int(config['spatial_epochs']), batch_size=None, train_size=1, lr=float(config['spatial_learning_rate']), accelerator=train_accelerator, device='auto')
            spatial_loss = loss_summary(model, int(config['spatial_epochs']), 'Cell2location')
            spatial = model.export_posterior(spatial, sample_kwargs={'num_samples': int(config['spatial_posterior_samples']), 'batch_size': None})
        if accelerator == 'cuda':
            torch.cuda.synchronize()
            peak_gpu_memory = int(torch.cuda.max_memory_allocated())
            if peak_gpu_memory <= 0:
                raise RuntimeError('Cell2location did not prove positive CUDA allocation')
        else:
            peak_gpu_memory = 0
        key = str(config['abundance_key'])
        if key not in spatial.obsm:
            raise RuntimeError(f'posterior lacks required abundance key {key}')
        abundance, abundance_column_schema = normalize_abundance_columns(spatial.obsm[key], key, factor_names)
        abundance = abundance.to_numpy()
        abundance = np.asarray(abundance, dtype=np.float64)
        if abundance.shape != (len(training_spots), len(factor_names)):
            raise RuntimeError(f'unexpected abundance shape: {abundance.shape}')
        if not np.isfinite(abundance).all() or abundance.min(initial=0) < 0:
            raise RuntimeError('posterior abundance is invalid')
        totals = abundance.sum(axis=1, keepdims=True)
        proportions = abundance.copy()
        nonzero = totals[:, 0] > 0
        proportions[nonzero] /= totals[nonzero]
        prediction = pd.DataFrame(0.0, index=full_spots, columns=factor_names)
        prediction.loc[training_spots] = proportions
        row_sums = prediction.to_numpy().sum(axis=1)
        if np.max(np.abs(row_sums[positive_mask] - 1), initial=0) > 1e-06:
            raise RuntimeError('positive-library prediction rows do not sum to one')
        zero_sums = row_sums[~positive_mask]
        if not np.all(np.isclose(zero_sums, 0, atol=1e-06) | np.isclose(zero_sums, 1, atol=1e-06)):
            raise RuntimeError('zero-library prediction mass is neither zero nor one')
        prediction.to_csv(args.output)
        packages = {}
        for package in ('numpy', 'anndata', 'scanpy', 'scvi-tools', 'cell2location', 'torch'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = 'unknown'
        metadata = {'status': 'validated_success', 'method': 'cell2location', 'public_seed': args.seed, 'actual_accelerator': accelerator, 'actual_device': actual_device, 'cuda_device_name': cuda_device_name, 'cuda_execution_verified': accelerator == 'cuda' and peak_gpu_memory > 0, 'peak_gpu_memory_allocated_bytes': peak_gpu_memory, 'input_path': str(args.input.resolve()), 'reference_path': str(args.reference.resolve()), 'output_path': str(args.output.resolve()), 'n_input_spots': len(full_spots), 'n_training_spots': len(training_spots), 'n_output_spots': len(prediction), 'n_zero_library_spots_excluded_from_scoring': int((~positive_mask).sum()), 'zero_library_policy': 'excluded from training and reinserted as zero rows; excluded from biological scoring' if args.positive_library_only else 'preserved in training/output; excluded from biological scoring', 'reference_model': reference_loss, 'spatial_model': spatial_loss, 'warnings': {'reference': warning_rows(reference_warnings), 'spatial': warning_rows(spatial_warnings), 'disposition': 'preserved_not_suppressed; review before publication'}, 'n_cells_per_location': n_cells_per_location, 'n_cells_per_location_policy': 'mean exact high-resolution source-cell assignments per training location; derived from simulation geometry without cell-type labels', 'source_cell_count_total': int(np.rint(source_cells_full).sum()), 'source_cell_count_min': int(np.rint(source_cells_full).min()), 'source_cell_count_max': int(np.rint(source_cells_full).max()), 'reference_label_filter': {'cell_type_key': args.cell_type_key, 'min_cells_per_type': args.min_cells_per_type, 'n_reference_cells_before': int(len(labels_full)), 'n_reference_cells_retained': int(reference_keep.sum()), 'n_reference_cell_types_before': int(labels_full.nunique()), 'n_reference_cell_types_retained': int(len(retained_types)), 'retained_cell_types': list(map(str, retained_types))}, 'model_prior': {'N_cells_per_location': n_cells_per_location, 'A_factors_per_location': model_kwargs.get('A_factors_per_location', 7.0), 'B_groups_per_location': model_kwargs.get('B_groups_per_location', 7.0)}, 'abundance_named_columns_validated_and_ordered': True, 'abundance_column_schema': abundance_column_schema, 'pre_normalization_abundance_total': abundance_total_summary(abundance), 'environment': {'python': platform.python_version(), 'packages': packages}, 'external_environment_limitation': LIMITATION}
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        return 0
    except Exception as exc:
        metadata_path.write_text(json.dumps({'status': 'failed_noncanonical', 'method': 'cell2location', 'error': repr(exc), 'traceback': traceback.format_exc(), 'external_environment_limitation': LIMITATION}, indent=2) + '\n')
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
