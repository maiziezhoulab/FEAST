"""Validate coordinate-assignment metrics and auxiliary transport diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
BOUNDED_METRICS = ('coordinate_nn_spot_accuracy', 'coordinate_nn_region_accuracy', 'moving_transport_coverage', 'transport_moving_exact_spot_recovery_rate', 'transport_moving_conditional_identity_accuracy', 'reference_active_support_fraction', 'reference_overlap_detection_precision', 'reference_overlap_detection_recall', 'reference_overlap_detection_f1', 'reference_exact_pair_precision', 'reference_exact_pair_recall', 'reference_exact_pair_f1', 'active_transport_mutual_consistency', 'ground_truth_transport_mass_fraction', 'transport_moving_region_recovery_rate', 'transport_moving_conditional_region_accuracy')
SIGNED_BOUNDED_METRICS = ('coordinate_nn_label_transfer_ari', 'transport_label_transfer_ari')

def validate_provenance(scores_dir: Path) -> bool:
    provenance = json.loads((scores_dir / 'provenance.json').read_text())
    outputs = provenance.get('outputs', {})
    required = {'alignment_metrics.csv', 'alignment_summary.csv'}
    return bool(int(provenance.get('metric_schema_version', -1)) == 3 and required.issubset(outputs) and all(((scores_dir / name).is_file() for name in required)))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spateo-scores', type=Path, required=True)
    parser.add_argument('--paste2-scores', type=Path, required=True)
    parser.add_argument('--rotation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'Zenodo/inputs/02_alignment' / 'config.yaml')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    expected = {(method, alteration, float(angle)) for method in ('spateo', 'paste2') for alteration in config['alterations'] for angle in config['angles']}
    spateo = pd.read_csv(args.spateo_scores / 'alignment_metrics.csv')
    paste2 = pd.read_csv(args.paste2_scores / 'alignment_metrics.csv')
    metrics = pd.concat([spateo, paste2], ignore_index=True)
    observed = set(zip(metrics['method'], metrics['alteration'], metrics['angle_degrees'].astype(float)))
    rotations = pd.read_csv(args.rotation_dir / 'rotation_manifest.csv')
    support = rotations[['alteration', 'angle_degrees', 'retained_fraction']].drop_duplicates()
    checked = metrics.merge(support, on=['alteration', 'angle_degrees'], how='left', suffixes=('', '_rotation'), validate='many_to_one')
    finite_columns = ('reference_median_spot_spacing', 'mean_spatial_error', 'mean_spatial_error_spot_units', 'coordinate_nn_mean_distance', 'rotation_recovery_error', 'transport_total_mass', 'transport_minimum', 'oracle_ge_correlation', 'coordinate_nn_predicted_pair_ge_correlation', 'coordinate_nn_ge_correlation_gap', 'transport_predicted_pair_ge_correlation', 'transport_ge_correlation_gap')
    valid = bool(len(metrics) == len(expected) and metrics['job_id'].is_unique and (observed == expected) and metrics['status'].eq('ok').all() and metrics['canonical_candidate'].all() and (metrics['n_aligned'] == metrics['n_moving_spots']).all() and np.isfinite(metrics[list(finite_columns)].to_numpy(float)).all() and np.isfinite(metrics[list(BOUNDED_METRICS)].to_numpy(float)).all() and np.isfinite(metrics[list(SIGNED_BOUNDED_METRICS)].to_numpy(float)).all() and ((metrics[list(BOUNDED_METRICS)] >= -1e-12) & (metrics[list(BOUNDED_METRICS)] <= 1 + 1e-12)).all().all() and ((metrics[list(SIGNED_BOUNDED_METRICS)] >= -1 - 1e-12) & (metrics[list(SIGNED_BOUNDED_METRICS)] <= 1 + 1e-12)).all().all() and (metrics['reference_median_spot_spacing'] > 0).all() and (metrics['mean_spatial_error'] >= 0).all() and (metrics['coordinate_nn_mean_distance'] >= 0).all() and np.allclose(metrics['mean_spatial_error_spot_units'], metrics['mean_spatial_error'] / metrics['reference_median_spot_spacing'], rtol=1e-12, atol=1e-12) and (metrics['transport_moving_exact_spot_recovery_rate'] <= metrics['moving_transport_coverage'] + 1e-12).all() and (metrics['transport_moving_region_recovery_rate'] <= metrics['moving_transport_coverage'] + 1e-12).all() and (metrics['transport_total_mass'] > 0).all() and (metrics['transport_minimum'] >= -1e-10).all() and np.allclose(checked['retained_fraction'], checked['retained_fraction_rotation'], rtol=0.0, atol=1e-12) and validate_provenance(args.spateo_scores) and validate_provenance(args.paste2_scores))
    rows = metrics[['method', 'job_id']].copy()
    rows.insert(0, 'kind', 'metric')
    rows['valid'] = valid
    if not valid:
        raise RuntimeError('replacement alignment metric validation failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    print(f'Validated {len(rows)} replacement metric rows')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
