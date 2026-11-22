#!/usr/bin/env python3
"""Rebuild the legacy ccc_figure5 tables used in Figure 6e/f."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
REGIONS = ['DorsalVZ', 'IZ', 'CP', 'VentralVZ', 'GE', 'SeptalVZ', 'Septum', 'BT']
PANELS = [('A', 'GSM8323035_E14M_1_2D', 'E14', 'Observed 2D'),
          ('B', 'E15.5_3D', 'E14', 'Reconstructed 3D'),
          ('C', 'GSM8323030_E18M_1_2D', 'E18', 'Observed 2D'),
          ('D', 'E18.5_3D', 'E18', 'Reconstructed 3D')]


def prepare(config_path, output, input_root=None, verified_root=None, lr_database=None):
    config = json.loads(config_path.read_text())
    input_root = input_root or config_path.parent / config['output_root']
    launch = json.loads((input_root / 'launch.json').read_text())
    assert launch['status'] == 'complete' and launch['exit_code'] == 0
    candidates = pd.read_csv(lr_database or config_path.parent / config['lr_database'])
    expected_pairs = set(map(tuple, candidates[['ligand', 'receptor']].to_numpy()))
    assert len(expected_pairs) == 85
    metrics, edges, ranks, validation = [], [], [], []
    sources, status_sets = {}, {}
    for panel, dataset, stage, modality in PANELS:
        folder = input_root / dataset
        info = json.loads((folder / 'run_settings.json').read_text())
        summary = json.loads((folder / 'network_summary.json').read_text())
        assert info['output_stage'] == 'complete' and not info['smoke_test_only']
        assert info['normalization_target_sum'] == 100 and info['candidate_LR_count'] == 85
        assert info['cluster_permutations'] == 100
        assert info['settings']['distance_threshold_mm'] == .035
        assert info['settings']['heteromeric'] and info['settings']['heteromeric_rule'] == 'min'
        file = folder / f'{dataset}_adata.h5ad'
        table = pd.read_csv(verified_root / dataset / 'regional_edges.csv' if verified_root else
                            folder / 'LR_region_edges.csv')
        status = pd.read_csv(folder / 'all_LR_status.csv').set_index('Key')
        assert len(table) == 85 * 8 * 8
        assert not table.duplicated(['ligand', 'receptor', 'sender_region', 'receiver_region']).any()
        assert set(map(tuple, table[['ligand', 'receptor']].drop_duplicates().to_numpy())) == expected_pairs
        assert set(table.sender_region) == set(REGIONS) and set(table.receiver_region) == set(REGIONS)
        assert np.isfinite(table.W).all() and (table.W >= 0).all()
        assert (table.n_edges >= 0).all() and (table.n_edges == np.floor(table.n_edges)).all()
        for (ligand, receptor), subset in table.groupby(['ligand', 'receptor']):
            record = status.loc[f'{ligand}--{receptor}']
            np.testing.assert_allclose(subset.W.sum(), record.nonself_mass, rtol=1e-10, atol=1e-9)
            assert subset.n_edges.sum() == record.n_nonself_edges
        aggregate = table.groupby(['sender_region', 'receiver_region'], sort=False).agg(W=('W', 'sum'), n_LR_edges=('n_edges', 'sum')).reset_index()
        w = aggregate.pivot(index='sender_region', columns='receiver_region', values='W').loc[REGIONS, REGIONS].to_numpy()
        total = float(w.sum()); inter = float(w[~np.eye(8, dtype=bool)].sum())
        assert total > 0
        np.testing.assert_allclose(total, summary['total_nonself_mass'], rtol=1e-12)
        np.testing.assert_allclose(inter / total, summary['F_inter'], rtol=1e-12)
        assert np.count_nonzero(w) == summary['n_nonzero_regional_edges_including_loops']
        assert np.count_nonzero(w[~np.eye(8, dtype=bool)]) == summary['n_nonzero_cross_region_edges']
        aggregate.insert(0, 'dataset', dataset); aggregate.insert(0, 'panel', panel)
        aggregate['share'] = aggregate.W / total
        aggregate['share_percent'] = 100 * aggregate.share
        aggregate['intra_region'] = aggregate.sender_region == aggregate.receiver_region
        np.testing.assert_allclose(aggregate.share.sum(), 1, atol=1e-12)
        edges.append(aggregate)
        for scope, subset in [('all', aggregate), ('cross_region', aggregate[~aggregate.intra_region])]:
            ranked = subset.sort_values(['share', 'sender_region', 'receiver_region'], ascending=[False, True, True]).copy()
            ranked['rank'] = np.arange(1, len(ranked) + 1); ranked['scope'] = scope
            ranks.append(ranked)
        active = set(status.index[status.Status == 1]); status_sets[dataset] = active
        metrics.append({'panel': panel, 'dataset': dataset, 'stage': stage, 'modality': modality,
                        'n_positions': info['n_positions'], 'position_unit': 'cells' if modality == 'Observed 2D' else 'voxels',
                        'n_lr_candidates': 85, 'n_active_lr_including_self': len(active),
                        'n_active_lr_nonself': int((status.nonself_mass > 0).sum()),
                        'F_inter_percent': 100 * inter / total, 'F_intra_percent': 100 * np.trace(w) / total,
                        'n_nonzero_regional_edges': int(np.count_nonzero(w)),
                        'n_nonzero_cross_region_edges': int(np.count_nonzero(w[~np.eye(8, dtype=bool)])),
                        'n_positive_nonself_LR_edges': int(table.n_edges.sum()),
                        'total_nonself_mass': total, 'self_mass_excluded': float(status.self_mass.sum()),
                        'runtime_hours': info['runtime_seconds'] / 3600})
        sources[dataset] = {'final_h5ad': str(file.relative_to(input_root)), 'raw_h5ad_included': file.is_file(), 'run_settings': info}
        validation.append({'dataset': dataset, 'complete_stage': True, 'all_85_LR_candidates_in_CSV': True,
                           'regional_mass_and_counts_match_LR_status': True,
                           'network_summary_matches_W': True, 'share_sum': float(aggregate.share.sum()), 'passed': True})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics).to_csv(output / 'network_metrics.csv', index=False)
    pd.concat(edges, ignore_index=True).to_csv(output / 'regional_edges.csv', index=False)
    pd.concat(ranks, ignore_index=True).to_csv(output / 'regional_interaction_ranks.csv', index=False)
    candidates.to_csv(output / 'LR_candidates.csv', index=False)
    pd.DataFrame(validation).to_csv(output / 'validation.csv', index=False)
    activity = candidates[['ligand', 'receptor']].copy()
    for dataset, active in status_sets.items():
        activity[dataset] = [int(f'{l}--{r}' in active) for l, r in candidates[['ligand', 'receptor']].itertuples(index=False, name=None)]
    activity.to_csv(output / 'LR_activity.csv', index=False)
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'entry_point': str(Path(__file__).resolve()),
                  'config': str(config_path.resolve()), 'input_root': str(input_root), 'sources': sources,
                  'region_order': REGIONS, 'panels': PANELS, 'launch': launch,
                  'network': 'W_ab summed over all 85 LR; share W_ab/sum_ab W_ab separately for each dataset; no LR-wise equal weighting.',
                  'self_policy': 'Same-position i=j excluded; same-region different-position communication retained as loops.',
                  'filtering': 'All positive regional sums retained; no p-value or score threshold filtering.',
                  'validation_scope': 'Completed run metadata and sparse regional sums verified against saved LR counts and masses.'
                  if verified_root else 'Completed run metadata and saved LR counts and masses.',
                  'limitations': ['Observed 2D cells and reconstructed 3D voxels differ in density, coverage and geometry; these shares do not establish a pure 3D benefit.',
                                  'One observed male PBS image per stage versus a 3D atlas built from five male reference images per stage; not an animal-replicate comparison. These observed images contributed to the reconstruction references, so this is not held-out validation.',
                                  '2D labels are marker-based annotation recovery; 3D labels use DevCCF anatomy. Spatial label structure can affect regional connectivity.',
                                  'E14/E18 source image labels map to E15.5/E18.5 DevCCF targets; GEO overall design additionally uses E14.5/E18.5.',
                                  'Matched-slice reconstructed 2D reinference (state B), eligible-pair opportunity normalization and animal-level validation are not included.',
                                  'Custom ComFlow incorporates receptor-TF information; keys retain commot-User_db names. No significance filtering or developmental differential testing.',
                                  'Regional edge counts use >0 and can include very small weights; count differences alone do not establish substantial biological connections.']}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'data_notes.txt').write_text(provenance['network'] + '\n' + provenance['self_policy'] + '\n' + provenance['filtering'] + '\n\n' + '\n'.join(provenance['limitations']) + '\n')
    print(pd.DataFrame(metrics)[['panel', 'dataset', 'F_inter_percent', 'n_nonzero_regional_edges']].to_string(index=False))
    print('Output:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=STUDY / 'ccc_rerun/config.json')
    parser.add_argument('--output-dir', type=Path, default=STUDY / 'outputs/ccc_figure5')
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--verified-root', type=Path)
    args = parser.parse_args()
    prepare(args.config, args.output_dir, args.input_root, args.verified_root)
