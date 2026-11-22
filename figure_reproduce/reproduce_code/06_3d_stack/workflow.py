"""Shared, experiment-owned contracts for clean Study 06 execution."""
from __future__ import annotations
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import yaml
STUDY_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = STUDY_ROOT / 'config.yaml'

@dataclass(frozen=True)
class SliceInfo:
    slice_id: int
    path: Path
    z: float
    n_obs: int
    n_vars: int

@dataclass(frozen=True)
class DensitySpec:
    name: str
    gap: int
    extra_anchor_slices: list[int]
    expected_references: int
    expected_targets: int

@dataclass(frozen=True)
class TargetAssignment:
    density_name: str
    gap: int
    target_index: int
    target_slice: int
    lower_ref_slice: int
    upper_ref_slice: int
    target_z: float
    z0: float
    z1: float
    tau: float
    reference_gap_z: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_config(path: Path=CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('config must contain a YAML mapping')
    required = {'configuration_id', 'study_id', 'public_seed', 'feast_version', 'dataset', 'anchor_selection', 'densities', 'reference_fit', 'generation', 'transport', 'validation', 'evaluation'}
    missing = required - set(config)
    if missing:
        raise ValueError(f'config is missing fields: {sorted(missing)}')
    if str(config['study_id']) != '06':
        raise ValueError("study_id must remain '06'")
    validate_fixed_contract(config)
    return config

def validate_fixed_contract(config: Mapping[str, Any]) -> None:
    generation = config['generation']
    transport = config['transport']
    dataset = config['dataset']
    if int(dataset['expected_genes']) != 1122:
        raise ValueError('Study 06 must retain the declared 1,122-gene panel')
    if int(config['reference_fit']['min_gene_spots']) != 0:
        raise ValueError('reference_fit.min_gene_spots must remain zero to retain all 1,122 genes')
    if int(dataset['first_slice']) != 4 or int(dataset['last_slice']) != 150:
        raise ValueError('the corrected modeled range must remain slices 4 through 150')
    if tuple(map(int, dataset.get('excluded_conflicting_slices', []))) != (1, 2, 3):
        raise ValueError('slices 1, 2, and 3 must remain excluded from the corrected study')
    if tuple(map(int, dataset['intentionally_absent_slices'])) != (75, 94, 112):
        raise ValueError('the three absent source slices changed')
    anchors = config['anchor_selection']
    if anchors.get('unit') != 'rank_among_available_slices' or anchors.get('phase') != 'first_available_slice' or tuple(map(int, anchors.get('boundary_slices', []))) != (4, 150):
        raise ValueError('the approved rank-based anchor policy changed')
    expected = {3: (49, 95, ()), 5: (30, 114, ()), 10: (17, 127, (89,))}
    observed = {int(row['gap']): (int(row['expected_references']), int(row['expected_targets']), tuple(map(int, row['extra_anchor_slices']))) for row in config['densities']}
    if observed != expected:
        raise ValueError(f'density contract changed: {observed!r}')
    required_transport = {'sinkhorn_iter': 1000, 'sinkhorn_tol': 1e-05, 'transport_nonconvergence': 'raise', 'max_transport_pairs': 25000000, 'sinkhorn_method': 'sinkhorn_log', 'transport_backend': 'torch', 'transport_device': 'cuda:0', 'transport_dtype': 'float64'}
    for key, expected_value in required_transport.items():
        if transport.get(key) != expected_value:
            raise ValueError(f'transport.{key} must remain {expected_value!r}')
    if generation.get('target_expression_access') != 'evaluation_only':
        raise ValueError('target expression must remain evaluation-only')
    if generation.get('smoothing') is not False or generation.get('z_regularization') is not False:
        raise ValueError('Study 06 prohibits smoothing and z regularization')
    evaluation = config['evaluation']
    if evaluation.get('targets_are_biological_replicates') is not False:
        raise ValueError('the simulated targets must not be treated as biological replicates')
    baseline = evaluation.get('real_continuity_baseline', {})
    if int(baseline.get('minimum_class_spots', -1)) != 2 or int(baseline.get('minimum_z_levels', -1)) != 3:
        raise ValueError('real split-half continuity baseline contract changed')

def density_specs(config: Mapping[str, Any]) -> list[DensitySpec]:
    return [DensitySpec(**row) for row in config['densities']]

def build_assignments(slice_infos: Mapping[int, SliceInfo], spec: DensitySpec, public_seed: int) -> tuple[list[TargetAssignment], list[int]]:
    available = sorted(slice_infos, key=lambda item: (float(slice_infos[item].z), int(item)))
    if not available:
        raise ValueError('no input slices were discovered')
    references = set(available[::int(spec.gap)])
    references.update((available[0], available[-1]))
    extras = {int(item) for item in spec.extra_anchor_slices}
    missing_extras = extras - set(available)
    if missing_extras:
        raise ValueError(f'{spec.name} extra anchors are unavailable: {sorted(missing_extras)}')
    references.update(extras)
    target_ids = [item for item in available if item not in references]
    rows: list[TargetAssignment] = []
    for target_index, target_id in enumerate(target_ids):
        target_z = float(slice_infos[target_id].z)
        lower_candidates = [item for item in references if float(slice_infos[item].z) < target_z]
        upper_candidates = [item for item in references if float(slice_infos[item].z) > target_z]
        if not lower_candidates or not upper_candidates:
            raise ValueError(f'target {target_id:03d} cannot be strictly bracketed for gap {spec.gap}')
        lower = max(lower_candidates, key=lambda item: (float(slice_infos[item].z), int(item)))
        upper = min(upper_candidates, key=lambda item: (float(slice_infos[item].z), int(item)))
        z0 = float(slice_infos[lower].z)
        z1 = float(slice_infos[upper].z)
        if not z0 < target_z < z1:
            raise ValueError(f'target {target_id:03d} z is not strictly inside its reference interval')
        tau = (target_z - z0) / (z1 - z0)
        rows.append(TargetAssignment(density_name=spec.name, gap=spec.gap, target_index=target_index, target_slice=target_id, lower_ref_slice=lower, upper_ref_slice=upper, target_z=target_z, z0=z0, z1=z1, tau=float(tau), reference_gap_z=float(z1 - z0), seed=int(public_seed) + int(spec.gap) * 10000 + target_index))
    reference_ids = sorted(references)
    if len(rows) != spec.expected_targets:
        raise ValueError(f'{spec.name} expected {spec.expected_targets} targets, found {len(rows)}')
    if len(reference_ids) != spec.expected_references:
        raise ValueError(f'{spec.name} expected {spec.expected_references} references, found {len(reference_ids)}')
    if set(reference_ids) & set(target_ids):
        raise ValueError(f'{spec.name} real and simulated partitions overlap')
    if set(reference_ids) | set(target_ids) != set(available):
        raise ValueError(f'{spec.name} real and simulated partitions do not cover all available slices')
    return (rows, reference_ids)

def representative_reference_ids(reference_ids: Sequence[int], count: int) -> list[int]:
    values = sorted({int(item) for item in reference_ids})
    if len(values) <= count:
        return values
    indices = sorted({int(round(position)) for position in np.linspace(0, len(values) - 1, count)})
    return [values[index] for index in indices]

def choose_donor(label: str, target_slice: int, target_z: float, label_support: Mapping[str, Sequence[int]], slice_infos: Mapping[int, SliceInfo]) -> int:
    candidates = [int(item) for item in label_support.get(str(label), []) if int(item) != int(target_slice)]
    if not candidates:
        raise ValueError(f'no declared reference donor supports label {label!r} for target {target_slice:03d}')
    distances = {item: abs(float(slice_infos[item].z) - float(target_z)) for item in candidates}
    minimum = min(distances.values())
    tied = [item for item, distance in distances.items() if math.isclose(distance, minimum, rel_tol=0.0, abs_tol=1e-12)]
    return min(tied)

def feast_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    import FEAST
    version = str(FEAST.__version__)
    if version != str(config['feast_version']):
        raise RuntimeError(f"expected FEAST {config['feast_version']}, found {version}")
    return {'status': 'OK', 'version': version, 'import_path': str(Path(FEAST.__file__).resolve()), 'python': sys.version.split()[0]}

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))
