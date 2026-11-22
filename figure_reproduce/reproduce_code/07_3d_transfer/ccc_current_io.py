"""Paths and stored-marginal access for the completed matched-85 CCC run."""
from pathlib import Path
import json

STUDY = Path(__file__).resolve().parent
ZENODO = STUDY.parents[1] / "Zenodo"
INPUTS = ZENODO / "inputs" / STUDY.name
RESULTS = ZENODO / "results" / STUDY.name
AGES = ('E15.5', 'E18.5')
REGIONS = ('DorsalVZ', 'IZ', 'CP', 'VentralVZ', 'GE', 'SeptalVZ', 'Septum', 'BT')
DATA = RESULTS / 'ccc_atlas'


def input_root():
    config = json.loads((STUDY / 'ccc_rerun/config.json').read_text())
    return STUDY / 'ccc_rerun' / config['output_root']


def ccc_path(root, age):
    return Path(root) / f'{age}_3D' / f'{age}_3D_adata.h5ad'


def marginal(h, role, pair):
    key = f'commot-User_db-sum-{role}'
    columns = h[f'uns/{key}_columns'].asstr()[:].tolist()
    return h[f'obsm/{key}'][:, columns.index(f'{role[0]}-{pair}')]


def run_info(h):
    info = json.loads(h['uns/ccc_rerun_provenance_json'].asstr()[()])
    assert info['output_stage'] == 'complete' and not info['smoke_test_only']
    assert info['normalization_target_sum'] == 100 and info['candidate_LR_count'] == 85
    assert info['settings']['distance_threshold_mm'] == .035
    assert info['settings']['heteromeric'] and info['settings']['heteromeric_rule'] == 'min'
    return info


LIMITATIONS = [
    'Male-only FEAST 1.0.6 reconstruction followed by custom ComFlow with receptor-TF information; output keys retain the commot prefix.',
    'Both stages use the same 85 candidates, counts normalized to 100 then log1p, heteromeric=True with min rule, and a 35-um cutoff.',
    'Generated voxel positions are not measured cells or independent biological replicates. These are descriptive inferred signaling patterns, not direct signaling measurements or causal developmental tests.',
    'Source image labels E14/E18 map to DevCCF E15.5/E18.5 targets; GEO overall design additionally uses E14.5/E18.5.',
    'The 550 input features include 500 genes and 50 Blank controls. Original normalization and features are retained.',
    'The anatomy covers eight reconstructed telencephalic regions, not the whole brain. Geometry and reference-age mismatch can influence inferred patterns.',
    'Stored p-values are within-input region-label permutations, not developmental difference tests; no p-value filtering is applied to these figures.',
]
