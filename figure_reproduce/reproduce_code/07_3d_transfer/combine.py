"""Create lossless age-specific AnnData handoff files from validated slices."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import workflow

ROOT = None
WORK = None
DEST = None
EXPECTED = {'E15.5': (158, 2330927), 'E18.5': (202, 5213461)}


def equal_metadata(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            equal_metadata(left[key], right[key])
    elif isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_categorical=False, check_dtype=False)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        np.testing.assert_equal(left, right)


def combine_age(age):
    started = time.perf_counter()
    folder = ROOT / age
    manifest = pd.read_csv(folder / 'manifest.csv').sort_values('z_index').reset_index(drop=True)
    n_levels, n_positions = EXPECTED[age]
    assert manifest.z_index.tolist() == list(range(n_levels))
    assert int(manifest.n_spots.sum()) == n_positions
    final = DEST / f'{age}_FEAST_3D.h5ad'
    temporary = DEST / f'{age}_FEAST_3D.tmp.h5ad'
    if final.exists() or temporary.exists():
        raise FileExistsError(f'refusing to overwrite an existing handoff: {final}')
    parts = []
    slice_metadata = {}
    genes = None
    source_columns = None
    ids = []
    starts = []
    stops = []
    offset = 0
    for i, row in enumerate(manifest.itertuples(index=False)):
        source = ad.read_h5ad(folder / row.filename)
        if genes is None:
            genes = source.var_names.copy()
            source_columns = list(source.obs.columns)
        assert source.var_names.equals(genes) and source.n_vars == 550
        assert source.n_obs == int(row.n_spots)
        assert list(source.obs.columns) == source_columns
        assert set(source.layers) == {'counts'}
        assert set(source.obsm) == {'spatial', 'spatial_3d'}
        assert not source.obsp and not source.varm and not source.varp and source.raw is None
        assert source.X.dtype == np.int32 and source.layers['counts'].dtype == np.int32
        np.testing.assert_array_equal(source.X, source.layers['counts'])
        key = f'z{int(row.z_index):03d}'
        ids.append(key); starts.append(offset); offset += source.n_obs; stops.append(offset)
        # Preserve slice-specific tables and complete original uns instead of
        # presenting the first slice's parameters as volume-wide metadata.
        slice_metadata[key] = {'source_filename': str(row.filename),
                               'var': source.var.copy(), 'uns': dict(source.uns)}
        source.uns = {}
        source.X = sparse.csr_matrix(source.X)
        source.layers['counts'] = source.X
        parts.append(source)
        if (i + 1) % 25 == 0 or i + 1 == n_levels:
            print(f'{age}: loaded {i + 1}/{n_levels} slices', flush=True)
    combined = ad.concat(parts, axis=0, join='inner', merge='same', uns_merge=None,
                         label='slice_id', keys=ids, index_unique=None)
    combined.obs.index.name = parts[0].obs.index.name
    combined.var.index.name = genes.name
    del parts, source
    gc.collect()
    assert combined.shape == (n_positions, 550)
    assert combined.var_names.equals(genes)
    assert combined.obs_names.is_unique
    assert combined.X.dtype == combined.layers['counts'].dtype == np.int32
    assert (combined.X != combined.layers['counts']).nnz == 0
    # X and counts have identical values; share their in-memory representation.
    combined.layers['counts'] = combined.X
    combined.obs['age'] = pd.Categorical([age] * n_positions)
    manifest['slice_id'] = ids
    manifest['row_start'] = starts
    manifest['row_stop'] = stops
    combined.uns['age'] = age
    combined.uns['coordinate_system'] = 'devccf_world'
    combined.uns['slice_manifest'] = manifest
    combined.uns['slice_metadata'] = slice_metadata
    combined.uns['source_generation_provenance'] = json.loads((folder / 'provenance.json').read_text())
    validation = json.loads((ROOT / 'validation.json').read_text())
    assert validation['status'] == 'validated'
    combined.uns['source_validation'] = next(row for row in validation['ages'] if row['age'] == age)
    combined.uns['handoff'] = {
        'configuration_id': combined.uns['source_generation_provenance']['configuration_id'],
        'data_kind': 'synthetic_FEAST_counts', 'n_references_per_modeling_region': 2,
        'transport_dtype': 'float32', 'count_dtype': 'int32', 'storage': 'CSR sparse',
        'transformation': 'none; lossless dense-to-sparse storage and row concatenation',
        'row_order': 'manifest z_index ascending, original within-slice order',
        'source_directory': str(folder), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'slice_specific_metadata': 'uns[slice_metadata][slice_id]: original var table and uns',
        'common_var_metadata': 'only columns identical across every slice are retained in var',
        'n_slices': n_levels, 'n_positions': n_positions, 'n_genes': 550,
    }
    print(f'{age}: writing {combined.shape}, {combined.X.nnz:,} nonzero counts', flush=True)
    combined.write_h5ad(temporary, compression='gzip')
    del combined, slice_metadata
    gc.collect()

    verify_written(age, temporary)


def verify_written(age, temporary):
    started = time.perf_counter()
    folder = ROOT / age
    n_levels, n_positions = EXPECTED[age]
    final = DEST / f'{age}_FEAST_3D.h5ad'
    if final.exists():
        raise FileExistsError(final)
    # Verify actual values after disk round-trip against every original slice.
    # No hashes or source/build checks are used.
    restored = ad.read_h5ad(temporary)
    manifest = restored.uns['slice_manifest']
    reference = ad.read_h5ad(folder / manifest.iloc[0]['filename'])
    source_columns = list(reference.obs.columns)
    genes = reference.var_names.copy()
    del reference
    assert manifest.z_index.tolist() == list(range(n_levels))
    assert int(manifest.n_spots.sum()) == n_positions
    try:
        assert restored.shape == (n_positions, 550) and restored.obs_names.is_unique
        assert restored.var_names.equals(genes)
        assert set(restored.obs['age']) == {age}
        for i, row in enumerate(manifest.itertuples(index=False)):
            source = ad.read_h5ad(folder / row.filename)
            start, stop = int(row.row_start), int(row.row_stop)
            assert restored.obs_names[start:stop].equals(source.obs_names)
            pd.testing.assert_frame_equal(restored.obs.iloc[start:stop][source_columns], source.obs,
                                          check_dtype=False, check_categorical=False)
            assert set(restored.obs['slice_id'].iloc[start:stop]) == {row.slice_id}
            for key in source.obsm:
                np.testing.assert_array_equal(restored.obsm[key][start:stop], source.obsm[key])
            for lower in range(0, source.n_obs, 4096):
                upper = min(lower + 4096, source.n_obs)
                np.testing.assert_array_equal(restored.X[start+lower:start+upper].toarray(), source.X[lower:upper])
                np.testing.assert_array_equal(restored.layers['counts'][start+lower:start+upper].toarray(), source.layers['counts'][lower:upper])
            meta = restored.uns['slice_metadata'][row.slice_id]
            pd.testing.assert_frame_equal(meta['var'], source.var)
            equal_metadata(meta['uns'], source.uns)
            if (i + 1) % 25 == 0 or i + 1 == n_levels:
                print(f'{age}: verified {i + 1}/{n_levels} slices', flush=True)
    finally:
        restored.file.close()
    del restored, source
    gc.collect()
    os.replace(temporary, final)
    result = {'age': age, 'file': str(final), 'n_positions': n_positions, 'n_genes': 550,
              'n_slices': n_levels, 'size_bytes': final.stat().st_size,
              'verification_seconds': time.perf_counter() - started,
              'counts_verified_exactly': True, 'coordinates_verified_exactly': True,
              'spot_ids_and_order_verified': True, 'original_obs_preserved': True,
              'all_original_slice_uns_and_var_preserved': True}
    (WORK / f'{age}_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    global ROOT, WORK, DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config_male_feast106.yaml'))
    parser.add_argument('--age', choices=(*EXPECTED, 'both'), default='both')
    args = parser.parse_args()
    config = workflow.load_config(args.config)
    ROOT = config['_final_dir']
    WORK = config['_work_dir'] / 'unified'
    DEST = ROOT / 'unified'
    validation = json.loads((ROOT / 'validation.json').read_text())
    assert validation['status'] == 'validated'
    assert validation['configuration_id'] == config['configuration_id']
    DEST.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    ages = EXPECTED if args.age == 'both' else (args.age,)
    for age in ages:
        combine_age(age)
    results = [json.loads((WORK / f'{age}_verification.json').read_text()) for age in ages]
    completion = {'status': 'complete', 'configuration_id': config['configuration_id'],
                  'completed_utc': datetime.now(timezone.utc).isoformat(), 'files': results,
                  'source_slices_preserved': True,
                  'verification': 'every count entry, both coordinate matrices, spot IDs/order, original observation values, and all original slice uns/var compared after disk round-trip',
                  'reader_used': 'anndata.read_h5ad, standard in-memory sparse mode'}
    (DEST / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')


if __name__ == '__main__':
    main()
