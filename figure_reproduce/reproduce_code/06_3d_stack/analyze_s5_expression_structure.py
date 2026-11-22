"""Expression-only clustering and matched reference-space embeddings for S5."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '4'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/feast-s5-clustering-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/feast-s5-clustering-numba')
import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from umap import UMAP
ZENODO = Path(__file__).resolve().parents[2] / 'Zenodo'
STUDY = '06_3d_stack'
PARAMETERS = dict(normalization_total=10000, transform='log1p', n_genes=1122, n_pcs=50, pca_solver='arpack', n_neighbors=15, distance='euclidean', leiden_resolution=1.0, leiden_flavor='leidenalg', leiden_directed=False, leiden_use_weights=True, leiden_iterations=-1, seed=2026, cpu_threads=4, umap_min_dist=0.5, umap_spread=1.0, umap_transform_seed=2026, subsampling='saved native-view indices: 1200 cells/section, seed 2026 + section ID', feature_selection='all 1122 genes; no HVG selection or additional cell/gene filtering', clustering_representation='independent centered PCA fitted separately to each condition', display_representation='observed-fitted centered PCA and UMAP; generated data use observed mean and loadings', annotation_inputs='none; no cell classes or spatial coordinates used in clustering or embedding')

def read_counts(path, selected, gene_names=None, cell_ids=None):
    with h5py.File(path, 'r') as handle:
        obs = handle['obs']
        names = pd.Index(read_elem(obs[obs.attrs['_index']])).astype(str)
        var = handle['var']
        genes = pd.Index(read_elem(var[var.attrs['_index']])).astype(str)
        counts = read_elem(handle['layers/counts'])
    assert genes.is_unique and names.is_unique
    if gene_names is not None:
        assert genes.equals(gene_names), str(path)
    names = names[selected]
    if cell_ids is not None:
        assert names.equals(cell_ids), str(path)
    counts = sparse.csr_matrix(counts[selected], dtype=np.float64)
    assert counts.shape == (len(selected), 1122)
    assert np.isfinite(counts.data).all() and (counts.data >= 0).all()
    return (counts, genes, names)

def load_condition(condition, gap, generation, data_dir, arms, sample, genes, output):
    matrices, rows, sources = ([], [], [])
    for number, (sid, chosen) in enumerate(sample.groupby('target_slice', sort=True), start=1):
        artifact = f'Zhuang-ABCA-1.{sid:03d}'
        path = data_dir / f'{artifact}.h5ad' if gap == 0 else generation / 'targets' / arms[gap] / artifact / 'generated.h5ad'
        indices = chosen.spot_index.to_numpy(int)
        identities = pd.Index(chosen.cell_id)
        matrix, genes, names = read_counts(path, indices, genes, identities)
        frame = chosen[['target_slice', 'spot_index']].copy()
        frame['cell_id'] = names.to_numpy()
        matrices.append(matrix)
        rows.append(frame)
        sources.append(str(path))
        if number % 19 == 0:
            print(f'{condition}: loaded {number}/76 sections', flush=True)
    cells = pd.concat(rows, ignore_index=True)
    assert not cells.duplicated(['target_slice', 'cell_id']).any()
    data = ad.AnnData(sparse.vstack(matrices, format='csr'), obs=pd.DataFrame(index=cells.target_slice.astype(str) + ':' + cells.cell_id), var=pd.DataFrame(index=genes))
    zero_libraries = int(np.count_nonzero(np.asarray(data.X.sum(axis=1)).ravel() == 0))
    sc.pp.normalize_total(data, target_sum=PARAMETERS['normalization_total'])
    sc.pp.log1p(data)
    print(f'{condition}: PCA, neighbor graph and independent Leiden', flush=True)
    sc.pp.pca(data, n_comps=50, zero_center=True, svd_solver='arpack', random_state=2026, dtype='float64')
    sc.pp.neighbors(data, n_neighbors=15, n_pcs=50, metric='euclidean', method='umap', random_state=2026)
    sc.tl.leiden(data, resolution=1.0, random_state=2026, directed=False, use_weights=True, n_iterations=-1, flavor='leidenalg', key_added='leiden')
    cells['condition'] = condition
    cells['gap'] = gap
    cells['leiden'] = data.obs.leiden.to_numpy().astype(str)
    cells.to_csv(output / f'{condition}_clusters.csv.gz', index=False)
    return (data, cells, genes, sources, zero_libraries)

def match_clusters(observed, generated):
    reference_labels = sorted(observed.leiden.unique(), key=int)
    generated_labels = sorted(generated.leiden.unique(), key=int)
    overlap = pd.crosstab(generated.leiden.to_numpy(), observed.leiden.to_numpy()).reindex(index=generated_labels, columns=reference_labels, fill_value=0)
    row, column = linear_sum_assignment(-overlap.to_numpy())
    matches = {generated_labels[r]: reference_labels[c] for r, c in zip(row, column) if overlap.iloc[r, c] > 0}
    rows = []
    for label in generated_labels:
        mapped = matches.get(label)
        rows.append(dict(condition=generated.condition.iloc[0], gap=int(generated.gap.iloc[0]), leiden=label, reference_leiden=mapped, n_cells=int(overlap.loc[label].sum()), matched_cells=int(overlap.loc[label, mapped]) if mapped is not None else 0, plot_label=f'O{int(mapped) + 1:02d}' if mapped is not None else f'G{int(generated.gap.iloc[0])}-U{int(label) + 1:02d}'))
    overlap.index.name = 'generated_leiden'
    return (rows, overlap.reset_index().melt(id_vars='generated_leiden', var_name='observed_leiden', value_name='cell_overlap'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zenodo', type=Path, default=ZENODO)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    results = args.zenodo / 'results' / STUDY
    generation = results / 'reconstruction'
    native = results / 'figure05_sources/native_volume.parquet'
    data_dir = args.zenodo / 'inputs' / STUDY / 'h5ad'
    output = args.output_dir or results / 'expression_structure_replay'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'analysis.json').exists():
        raise FileExistsError('Completed analysis exists; reuse it for plotting or choose a fresh output directory.')
    started = time.time()
    sc.settings.n_jobs = 4
    plan = json.loads((generation / 'plan.json').read_text())
    arms = {int(arm['spec']['gap']): name for name, arm in plan['densities'].items()}
    targets = [set(map(int, arm['target_pool'])) for arm in plan['densities'].values()]
    common = sorted(set.intersection(*targets))
    assert len(common) == 76
    saved = pd.read_parquet(native, columns=['target_slice', 'spot_index', 'spot_id'],
                            filters=[('method', '==', 'Observed'), ('target_slice', 'in', common)])
    sample = saved.rename(columns={'spot_id': 'cell_id'}).sort_values(
        ['target_slice', 'spot_index']).reset_index(drop=True)
    assert sample.groupby('target_slice').size().eq(1200).all() and len(sample) == 91200
    sources = [str(generation / 'plan.json'), str(native)]
    observed, ref_cells, genes, paths, zero = load_condition(
        'Observed', 0, generation, data_dir, arms, sample, None, output)
    sources.extend(paths)
    ref_cells[['target_slice', 'spot_index', 'cell_id']].to_csv(output / 'sampled_cells.csv.gz', index=False)
    pd.DataFrame({'gene': genes}).to_csv(output / 'genes.csv', index=False)
    reference_mean = np.asarray(observed.X.mean(axis=0)).ravel()
    basis = observed.varm['PCs'].copy()
    np.savez_compressed(output / 'observed_pca.npz', mean=reference_mean, components=basis, variance=observed.uns['pca']['variance'])
    print('Observed: fitting the common UMAP display', flush=True)
    embedding = UMAP(n_neighbors=15, n_components=2, min_dist=0.5, spread=1.0, metric='euclidean', random_state=2026, transform_seed=2026, n_jobs=1)
    ref_cells[['umap_1', 'umap_2']] = embedding.fit_transform(observed.obsm['X_pca'])
    reference_labels = sorted(ref_cells.leiden.unique(), key=int)
    mapping = [dict(condition='Observed', gap=0, leiden=label, reference_leiden=label, n_cells=int((ref_cells.leiden == label).sum()), matched_cells=int((ref_cells.leiden == label).sum()), plot_label=f'O{int(label) + 1:02d}') for label in reference_labels]
    frames, overlaps = ([ref_cells], [])
    zero_counts = {'Observed': zero}
    for gap in (3, 5, 10):
        condition = f'FEAST_gap{gap}'
        generated, cells, _, paths, zero = load_condition(
            condition, gap, generation, data_dir, arms, ref_cells, genes, output)
        assert cells[['target_slice', 'spot_index', 'cell_id']].equals(ref_cells[['target_slice', 'spot_index', 'cell_id']])
        sources.extend(paths)
        zero_counts[condition] = zero
        print(f'{condition}: projection into observed PCA/UMAP coordinates', flush=True)
        projected = generated.X @ basis - reference_mean @ basis
        cells[['umap_1', 'umap_2']] = embedding.transform(projected)
        assert np.isfinite(cells[['umap_1', 'umap_2']]).all().all()
        matched, overlap = match_clusters(ref_cells, cells)
        mapping.extend(matched)
        overlap['condition'] = condition
        overlaps.append(overlap)
        frames.append(cells)
        del generated
    mappings = pd.DataFrame(mapping)
    result = pd.concat(frames, ignore_index=True).merge(mappings[['condition', 'leiden', 'reference_leiden', 'plot_label']], on=['condition', 'leiden'], validate='many_to_one')
    result['panel_id'] = 'C'
    result.to_csv(output / 'embedding_plot_data.csv.gz', index=False)
    mappings.to_csv(output / 'cluster_color_mapping.csv', index=False)
    pd.concat(overlaps, ignore_index=True).to_csv(output / 'cluster_overlap.csv.gz', index=False)
    summary = result.groupby(['condition', 'gap'], sort=False).agg(n_cells=('cell_id', 'size'), n_sections=('target_slice', 'nunique'), n_clusters=('leiden', 'nunique')).reset_index()
    summary.to_csv(output / 'condition_summary.csv', index=False)
    metadata = dict(status='complete', script=str(Path(__file__).resolve()), command=[sys.executable, *sys.argv], generation_root=str(generation), parameters=PARAMETERS, common_heldout_sections=common, n_sampled_cells_per_condition=len(sample), source_paths=sources, zero_library_cells=zero_counts, condition_summary=summary.to_dict('records'), cluster_color_rule='maximum-overlap one-to-one assignment to observed clusters; unmatched clusters retain distinct labels/colors; no cluster merging', scientific_limit='descriptive expression structure given supplied classes/geometry; reference projection is not independent evidence of distributional equivalence', figure_promotion_authorized=False, elapsed_seconds=round(time.time() - started, 2), environment={name: version(name) for name in ('scanpy', 'anndata', 'numpy', 'scipy', 'scikit-learn', 'umap-learn', 'pynndescent', 'leidenalg')})
    (output / 'analysis.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    print(f'Completed: {output}', flush=True)
if __name__ == '__main__':
    main()
