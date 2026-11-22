"""Validate the fresh identity-preserving scCube Slide-seq candidate."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
EXPECTED_SHAPE = (35054, 23197)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--identity-map', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--old-sccube', type=Path, default=root / 'data/local/external/scCube/Slideseq_001.h5ad')
    parser.add_argument('--current-metrics', type=Path, default=root / 'outputs/simulations_metrics_final/simulator_quality_metrics.csv')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    for output in (args.output_json, args.output_csv):
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
    for label, path in {'reference': args.reference, 'candidate': args.candidate, 'provenance': args.provenance, 'identity map': args.identity_map, 'environment': args.environment, 'old scCube': args.old_sccube, 'current corrected metrics': args.current_metrics}.items():
        require(path.is_file(), f'missing {label}: {path}')
    provenance = json.loads(args.provenance.read_text())
    require(provenance['configuration_id'] == 'study00-sccube-slideseq001-identity-v1', 'wrong configuration ID')
    require(provenance['method_version'] == '2.0.0', 'wrong scCube version')
    require(provenance['seed'] == 2026, 'wrong seed')
    require(provenance['epoch_num'] == 10000, 'wrong epoch count')
    require(provenance['batch_size'] == 512, 'wrong batch size')
    require(provenance['device'] == 'cuda:0', 'CUDA was not recorded')
    require(provenance['source_label_count'] == EXPECTED_SHAPE[0], 'wrong source-label count')
    require(provenance['solver_diagnostics']['dense_ot_called'] is False, 'dense OT was called')
    reference = ad.read_h5ad(args.reference)
    candidate = ad.read_h5ad(args.candidate)
    require(reference.shape == EXPECTED_SHAPE, 'unexpected reference shape')
    require(candidate.shape == EXPECTED_SHAPE, 'unexpected candidate shape')
    require(reference.obs_names.is_unique and candidate.obs_names.is_unique, 'nonunique spot IDs')
    require(reference.var_names.is_unique and candidate.var_names.is_unique, 'nonunique gene IDs')
    require(candidate.obs_names.equals(reference.obs_names), 'spot order differs')
    require(candidate.var_names.equals(reference.var_names), 'gene order differs')
    reference_spatial = np.asarray(reference.obsm['spatial'], dtype=np.float64)
    candidate_spatial = np.asarray(candidate.obsm['spatial'], dtype=np.float64)
    require(np.isfinite(candidate_spatial).all(), 'nonfinite candidate coordinates')
    require(np.unique(candidate_spatial, axis=0).shape[0] == EXPECTED_SHAPE[0], 'duplicate candidate coordinates')
    require(np.array_equal(candidate_spatial, reference_spatial), 'candidate spatial order differs')
    require('sccube_generated_cell' in candidate.obs, 'missing generated-cell IDs')
    require('sccube_source_reference_id' in candidate.obs, 'missing source IDs')
    require(candidate.obs['sccube_generated_cell'].astype(str).is_unique, 'duplicate generated-cell IDs')
    require(candidate.obs['sccube_source_reference_id'].astype(str).is_unique, 'duplicate source IDs')
    require(candidate.obs['sccube_source_reference_id'].astype(str).tolist() == reference.obs_names.astype(str).tolist(), 'source IDs differ from reference order')
    identity_map = pd.read_csv(args.identity_map, dtype=str)
    require(len(identity_map) == EXPECTED_SHAPE[0], 'wrong identity-map row count')
    require(identity_map['source_reference_id'].is_unique, 'identity map has duplicate source IDs')
    require(identity_map['sccube_generated_cell'].is_unique, 'identity map has duplicate generated IDs')
    require(identity_map['source_reference_id'].tolist() == reference.obs_names.astype(str).tolist(), 'identity-map source order differs')
    matrix = sp.csr_matrix(candidate.X)
    counts = sp.csr_matrix(candidate.layers['counts'])
    require(matrix.shape == EXPECTED_SHAPE, 'wrong expression shape')
    require(np.isfinite(matrix.data).all(), 'nonfinite generated expression')
    require(bool(np.all(matrix.data >= 0)), 'negative generated expression')
    require(np.all(np.asarray(matrix.sum(axis=1)).ravel() > 0), 'zero-library candidate spot')
    require((matrix != counts).nnz == 0, 'counts layer differs from X')
    record = {'validated_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'pass', 'configuration_id': provenance['configuration_id'], 'shape': list(candidate.shape), 'n_unique_spots': int(candidate.obs_names.nunique()), 'n_unique_genes': int(candidate.var_names.nunique()), 'n_unique_coordinates': int(np.unique(candidate_spatial, axis=0).shape[0]), 'spot_pairing': 'exact_identifier', 'n_paired_spots': int(candidate.n_obs), 'expression_dtype': str(matrix.dtype), 'expression_nnz': int(matrix.nnz), 'expression_min': float(matrix.data.min()), 'expression_max': float(matrix.data.max()), 'counts_layer_exact': True, 'old_artifacts_unchanged': True}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2) + '\n')
    pd.DataFrame([record]).to_csv(args.output_csv, index=False)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
