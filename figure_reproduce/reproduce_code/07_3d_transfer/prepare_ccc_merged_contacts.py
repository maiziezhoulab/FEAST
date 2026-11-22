#!/usr/bin/env python3
"""Contact-restricted LR profiles with the two directions of each pair summed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

from ccc_current_io import AGES, DATA

REGION_ORDER = ('CP', 'IZ', 'DorsalVZ', 'BT', 'GE', 'VentralVZ', 'SeptalVZ', 'Septum')


def physical_contacts(geometry_root):
    records = []
    for stage in AGES:
        frame = pd.read_parquet(geometry_root / f'{stage}_geometry.parquet',
                                columns=['region', 'voxel_i', 'voxel_j', 'voxel_k'])
        indices = frame[['voxel_i', 'voxel_j', 'voxel_k']].to_numpy(dtype=np.int32)
        indices -= indices.min(axis=0)
        codes = pd.Categorical(frame.region, categories=REGION_ORDER).codes + 1
        assert (codes > 0).all()
        grid = np.zeros(tuple(indices.max(axis=0) + 1), dtype=np.uint8)
        grid[tuple(indices.T)] = codes
        totals = np.zeros((8, 8), dtype=np.int64)
        for axis in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            a, b = grid[tuple(lo)], grid[tuple(hi)]
            keep = (a > 0) & (b > 0) & (a != b)
            aa, bb = a[keep].astype(np.int16) - 1, b[keep].astype(np.int16) - 1
            totals += np.bincount(8 * np.minimum(aa, bb) + np.maximum(aa, bb), minlength=64).reshape(8, 8)
        for i, a in enumerate(REGION_ORDER):
            for j in range(i + 1, 8):
                records.append({'stage': stage, 'region_a': a, 'region_b': REGION_ORDER[j],
                                'n_shared_faces': int(totals[i, j])})
    return pd.DataFrame(records)


def prepare(source, geometry_root, output):
    original = pd.read_csv(source / 'LR_region_pair_changes.csv')
    previous = pd.read_csv(source / 'LR_region_pair_metrics.csv').set_index('LR')
    provenance = json.loads((source / 'provenance.json').read_text())
    contacts = physical_contacts(geometry_root)
    contact_lookup = contacts.set_index(['stage', 'region_a', 'region_b']).n_shared_faces
    region_index = {r: i for i, r in enumerate(REGION_ORDER)}
    pair_indices = np.sort(np.column_stack([original.sender_region.map(region_index), original.receiver_region.map(region_index)]), axis=1)
    original['region_a'] = [REGION_ORDER[i] for i in pair_indices[:, 0]]
    original['region_b'] = [REGION_ORDER[i] for i in pair_indices[:, 1]]
    merged = original.groupby(['ligand', 'receptor', 'LR', 'region_a', 'region_b'], as_index=False).agg(
        unfiltered_W_early=('W_early', 'sum'), unfiltered_W_late=('W_late', 'sum'),
        unfiltered_n_edges_early=('n_edges_early', 'sum'), unfiltered_n_edges_late=('n_edges_late', 'sum'))
    for stage, label in zip(AGES, ['early', 'late']):
        merged[f'{label}_shared_faces'] = [int(contact_lookup.loc[(stage, a, b)]) for a, b in zip(merged.region_a, merged.region_b)]
        merged[f'contact_{label}'] = merged[f'{label}_shared_faces'].gt(0)
        merged[f'W_{label}'] = merged[f'unfiltered_W_{label}'].where(merged[f'contact_{label}'], 0.)
        merged[f'n_edges_{label}'] = merged[f'unfiltered_n_edges_{label}'].where(merged[f'contact_{label}'], 0)
    merged['retained_pair'] = merged.contact_early | merged.contact_late
    merged['stage_specific_contact'] = merged.contact_early != merged.contact_late
    merged['region_pair'] = merged.region_a + '–' + merged.region_b
    all_pairs = [(a, b) for i, a in enumerate(REGION_ORDER) for b in REGION_ORDER[i + 1:]]
    pair_order = {pair: i for i, pair in enumerate(all_pairs)}
    merged['pair_order'] = [pair_order[(a, b)] for a, b in zip(merged.region_a, merged.region_b)]
    rows = merged.loc[merged.retained_pair].copy()
    metrics, checks = [], []
    for key, group in rows.groupby('LR', sort=True):
        group = group.sort_values('pair_order')
        distributions, support = [], {}
        for label in ['early', 'late']:
            total = float(group[f'W_{label}'].sum())
            assert total > 0
            distribution = group[f'W_{label}'].to_numpy() / total
            distributions.append(distribution)
            raw_total = float(merged.loc[merged.LR.eq(key), f'unfiltered_W_{label}'].sum())
            expected = float(original.loc[original.LR.eq(key), f'W_{label}'].sum())
            np.testing.assert_allclose(raw_total, expected, rtol=1e-12, atol=1e-9)
            support[f'{label}_retained_mass'] = total
            support[f'{label}_excluded_no_contact_mass'] = raw_total - total
            support[f'{label}_excluded_mass_pct'] = 100 * (raw_total - total) / raw_total
            support[f'{label}_retained_edges'] = int(group[f'n_edges_{label}'].sum())
            rows.loc[group.index, f'{label}_share_pct'] = 100 * distribution
            checks.append({'LR': key, 'stage': label, 'profile_sum': float(distribution.sum()),
                           'retained_mass': total, 'source_merge_conserves_total': True, 'passed': True})
        early, late = distributions
        js = float(jensenshannon(early, late, base=2) ** 2)
        midpoint = (early + late) / 2
        independent = float((rel_entr(early, midpoint).sum() + rel_entr(late, midpoint).sum()) / (2 * np.log(2)))
        np.testing.assert_allclose(js, independent, rtol=1e-11, atol=1e-14)
        metrics.append({'LR': key, 'ligand': group.ligand.iloc[0], 'receptor': group.receptor.iloc[0],
                        'JSD_base2': js, 'original_directed_JSD_rank': int(previous.loc[key, 'JSD_rank']), **support})
    rows['delta_share_pp'] = rows.late_share_pct - rows.early_share_pct
    rows['abs_delta_share_pp'] = rows.delta_share_pp.abs()
    metrics = pd.DataFrame(metrics).sort_values(['JSD_base2', 'LR'], ascending=[False, True]).reset_index(drop=True)
    metrics.insert(0, 'JSD_rank', np.arange(1, len(metrics) + 1))
    rows = rows.merge(metrics[['LR', 'JSD_rank']], on='LR', validate='many_to_one').sort_values(['JSD_rank', 'pair_order'])
    pairs = rows[['region_a', 'region_b', 'region_pair', 'pair_order', 'early_shared_faces', 'late_shared_faces',
                  'contact_early', 'contact_late', 'stage_specific_contact']].drop_duplicates().sort_values('pair_order')
    excluded = merged.loc[~merged.retained_pair, ['region_a', 'region_b', 'region_pair', 'early_shared_faces', 'late_shared_faces']].drop_duplicates()
    np.testing.assert_allclose(rows.groupby('LR')[['early_share_pct', 'late_share_pct']].sum(), 100, atol=1e-11)
    assert not rows.duplicated(['LR', 'region_a', 'region_b']).any()
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / 'LR_region_pair_changes.csv', index=False)
    metrics.to_csv(output / 'LR_region_pair_metrics.csv', index=False)
    pairs.to_csv(output / 'physical_contact_pairs.csv', index=False)
    excluded.to_csv(output / 'excluded_region_pairs.csv', index=False)
    contacts.to_csv(output / 'anatomical_contact_counts.csv', index=False)
    merged.to_csv(output / 'contact_filter_audit.csv', index=False)
    pd.DataFrame(checks).to_csv(output / 'validation.csv', index=False)
    stage_specific = pairs.loc[pairs.stage_specific_contact, 'region_pair'].tolist()
    info = {'created_utc': datetime.now(timezone.utc).isoformat(), 'entry_point': str(Path(__file__).resolve()),
            'input_root': str(source.resolve()), 'geometry_root': str(geometry_root.resolve()), 'sources': provenance['sources'],
            'region_order': REGION_ORDER, 'region_pair_order': pairs.region_pair.tolist(),
            'stage_specific_pairs': stage_specific, 'analyzed_LR_count': len(metrics), 'n_retained_pairs': len(pairs),
            'n_excluded_pairs': len(excluded), 'direction_handling': 'M_ab = W_a_to_b + W_b_to_a for each unordered region pair; each direction counted once.',
            'physical_contact': 'At least one shared native 20-um voxel face. Applied separately in each atlas stage, independently of inferred LR scores.',
            'structural_zeros': 'If contact is absent in a stage, its retained mass/count is zero in that stage. A pair is dropped entirely only if contact is absent in both stages.',
            'normalization': 'P_ab = M_ab_contact / sum_unique_contacting_pairs M_contact, separately per LR and stage. Stage profiles sum to 1; differences are 100*(P_late-P_early) pp.',
            'JSD': 'Base-2 Jensen-Shannon divergence between merged, contact-filtered normalized profiles, scipy jensenshannon squared and independently checked by relative entropy.',
            'ranking': 'Recomputed after summing directions and contact filtering. Main/supplementary case choices retain their previously selected original directed-screen ranks.',
            'new_inference': False, 'limitations': provenance['limitations'] + [
                'Strict voxel-face contact is narrower than the model distance criterion; it intentionally excludes no-face aggregate weights even if positions were within 35 um.',
                'Mass shares describe relative allocation among contacting regions, not contact-area or eligible-neighbor-normalized interaction intensity.',
                'Summing directions removes directional asymmetry; original directed source tables and weights remain unchanged.']}
    (output / 'provenance.json').write_text(json.dumps(info, indent=2) + '\n')
    print(metrics[['JSD_rank', 'LR', 'JSD_base2', 'original_directed_JSD_rank']].head(15).to_string(index=False))
    print(f'Merged physical-contact comparison complete: {len(pairs)} retained pairs, {len(excluded)} pairs excluded, {len(rows)} LR/pair rows.', flush=True)
    print('Stage-specific contacts:', stage_specific, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DATA / 'region_pairs')
    parser.add_argument('--geometry-root', type=Path, default=DATA / 'top5_spatial')
    parser.add_argument('--output-dir', type=Path, default=DATA / 'region_pairs_merged_contact')
    args = parser.parse_args()
    prepare(args.input_root, args.geometry_root, args.output_dir)
