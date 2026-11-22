from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from preprocess_common import add_common_arguments, add_counts_layer, add_standard_obs, add_var_metadata, append_manifest_rows, apply_qc, ensure_processed_dirs, filter_requested_samples, manifest_row, preprocessing_uns, print_inventory, set_labels, setup_logging, set_spatial, to_csr_counts, write_processed
DATASET_KEY = 'Allen_Zhuang_ABCA_1'
SCRIPT_PATH = Path(__file__).resolve()
META_COLS = ['cell_label', 'brain_section_label', 'donor_label', 'donor_genotype', 'donor_sex', 'cluster_alias', 'x', 'y', 'z', 'subclass_confidence_score', 'cluster_confidence_score', 'high_quality_transfer', 'neurotransmitter', 'class', 'subclass', 'supertype', 'cluster']

def component_paths(raw_dataset_dir: Path) -> dict[str, Path]:
    return {'raw_h5ad': raw_dataset_dir / 'components' / 'expression_matrices' / 'Zhuang-ABCA-1' / '20230830' / 'Zhuang-ABCA-1-raw.h5ad', 'cluster_metadata': raw_dataset_dir / 'components' / 'metadata' / 'Zhuang-ABCA-1' / '20241115' / 'views' / 'cell_metadata_with_cluster_annotation.csv', 'ccf': raw_dataset_dir / 'components' / 'metadata' / 'Zhuang-ABCA-1-CCF' / '20230830' / 'ccf_coordinates.csv'}

def safe_name(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')

def discover_sections(metadata_path: Path) -> pd.Series:
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(metadata_path, usecols=['brain_section_label'], chunksize=1000000):
        vc = chunk['brain_section_label'].astype(str).value_counts()
        for key, value in vc.items():
            counts[key] = counts.get(key, 0) + int(value)
    return pd.Series(counts).sort_index()

def load_cluster_metadata(metadata_path: Path, sections: list[str]) -> pd.DataFrame:
    selected = set(sections)
    chunks = []
    for chunk in pd.read_csv(metadata_path, usecols=META_COLS, dtype={'cell_label': str}, chunksize=500000):
        chunk['brain_section_label'] = chunk['brain_section_label'].astype(str)
        sub = chunk[chunk['brain_section_label'].isin(selected)]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        raise ValueError(f'No metadata rows found for sections: {sections}')
    meta = pd.concat(chunks, ignore_index=True)
    meta['cell_label'] = meta['cell_label'].astype(str)
    return meta.set_index('cell_label')

def load_ccf(ccf_path: Path, cell_labels: pd.Index) -> pd.DataFrame:
    wanted = set(cell_labels.astype(str))
    chunks = []
    for chunk in pd.read_csv(ccf_path, dtype={'cell_label': str}, chunksize=1000000):
        sub = chunk[chunk['cell_label'].astype(str).isin(wanted)]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame(index=cell_labels, columns=['ccf_x', 'ccf_y', 'ccf_z', 'ccf_parcellation_index'])
    ccf = pd.concat(chunks, ignore_index=True)
    ccf['cell_label'] = ccf['cell_label'].astype(str)
    ccf = ccf.rename(columns={'x': 'ccf_x', 'y': 'ccf_y', 'z': 'ccf_z', 'parcellation_index': 'ccf_parcellation_index'}).set_index('cell_label')
    return ccf.reindex(cell_labels)

def process_section(section: str, raw_backed: ad.AnnData, meta_all: pd.DataFrame, ccf_all: pd.DataFrame, args, paths: dict[str, Path], input_files: list[Path]) -> dict:
    logging.info('Processing section %s', section)
    section_meta = meta_all[meta_all['brain_section_label'].astype(str) == section]
    if section_meta.empty:
        raise ValueError(f'No metadata for section {section}')
    raw_section_mask = raw_backed.obs['brain_section_label'].astype(str).to_numpy() == section
    raw_cell_labels = pd.Index(raw_backed.obs_names.astype(str))
    raw_mask = raw_section_mask & raw_cell_labels.isin(section_meta.index.astype(str))
    dropped_raw_cells = int(raw_section_mask.sum() - raw_mask.sum())
    adata = raw_backed[raw_mask, :].to_memory()
    adata.obs_names = adata.obs_names.astype(str)
    missing_from_raw = section_meta.index.astype(str).difference(adata.obs_names)
    if len(missing_from_raw):
        raise ValueError(f'{section}: {len(missing_from_raw)} metadata cells missing from raw h5ad')
    section_meta = section_meta.reindex(adata.obs_names)
    if section_meta.isna().all(axis=1).any():
        missing = int(section_meta.isna().all(axis=1).sum())
        raise ValueError(f'{section}: {missing} raw cells missing cluster metadata')
    for col in section_meta.columns:
        adata.obs[col] = section_meta[col].to_numpy()
    section_ccf = ccf_all.reindex(adata.obs_names)
    for col in section_ccf.columns:
        if col in {'ccf_x', 'ccf_y', 'ccf_z'}:
            adata.obs[col] = pd.to_numeric(section_ccf[col], errors='coerce').to_numpy(dtype=float)
        else:
            adata.obs[col] = section_ccf[col].astype('string').fillna('unknown').astype(str).to_numpy()
    counts = to_csr_counts(adata.X)
    adata.X = counts
    add_counts_layer(adata, counts)
    gene_ids = adata.var.index.astype(str).to_numpy()
    gene_symbols = adata.var['gene_symbol'].astype(str).to_numpy() if 'gene_symbol' in adata.var else gene_ids
    add_var_metadata(adata, gene_ids=gene_ids, gene_symbols=gene_symbols, species='mouse')
    donor = str(section_meta['donor_label'].iloc[0]) if 'donor_label' in section_meta else 'unknown'
    add_standard_obs(adata, dataset_key=args.dataset_key, sample_id=section, platform='MERFISH', species='mouse', tissue='whole adult brain', condition=str(section_meta['donor_genotype'].iloc[0]) if 'donor_genotype' in section_meta else 'unknown', donor_id=donor, replicate_id=section, section_id=section)
    xyz = section_meta[['x', 'y', 'z']].to_numpy(dtype=float)
    set_spatial(adata, xyz[:, :2], xyz=xyz)
    if {'ccf_x', 'ccf_y', 'ccf_z'}.issubset(adata.obs.columns):
        ccf_xyz = adata.obs[['ccf_x', 'ccf_y', 'ccf_z']].to_numpy(dtype=float)
        adata.obsm['ccf'] = ccf_xyz
        adata.obs['ccf_available'] = np.isfinite(ccf_xyz).all(axis=1)
    labels = section_meta['cluster'].fillna('unknown').astype(str)
    parcellation = adata.obs['ccf_parcellation_index'].astype(str)
    region = 'parcellation_' + parcellation
    set_labels(adata, region=region, cell_type=labels, annotation=labels, ground_truth=labels, label_source='cell_metadata_with_cluster_annotation.cluster')
    for src, dest in [('class', 'cell_class'), ('subclass', 'cell_subclass'), ('supertype', 'cell_supertype')]:
        if src in adata.obs:
            adata.obs[dest] = adata.obs[src].astype('category')
    adata, stats = apply_qc(adata, min_counts=args.min_counts, min_genes=args.min_genes, min_cells_per_gene=args.min_cells_per_gene, max_pct_mt=args.max_pct_mt)
    adata.uns['preprocessing'] = preprocessing_uns(dataset_key=args.dataset_key, sample_id=section, input_files=input_files, counts_source='Zhuang-ABCA-1-raw.h5ad:X', coordinate_source='cell_metadata_with_cluster_annotation.csv:x,y,z;ccf_coordinates.csv', label_source='cell_metadata_with_cluster_annotation.cluster', seed=args.seed, script_path=SCRIPT_PATH, extra={'allen_raw_section_cells_without_metadata': dropped_raw_cells})
    output_path = paths['h5ad'] / f'{safe_name(section)}.h5ad'
    write_processed(adata, output_path, overwrite=args.overwrite)
    return manifest_row(dataset_key=args.dataset_key, sample_id=section, input_files=input_files, output_h5ad=output_path, stats=stats, label_source='cell_metadata_with_cluster_annotation.cluster', coordinate_source='cell_metadata_with_cluster_annotation.csv:x,y,z;ccf_coordinates.csv', counts_source='Zhuang-ABCA-1-raw.h5ad:X', seed=args.seed, script_path=SCRIPT_PATH)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, DATASET_KEY)
    parser.add_argument('--sections', nargs='*', default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_dataset_dir = args.raw_root / args.dataset_key
    paths = ensure_processed_dirs(args.processed_root, args.dataset_key)
    setup_logging(paths['logs'], args.dataset_key)
    comp = component_paths(raw_dataset_dir)
    section_counts = discover_sections(comp['cluster_metadata'])
    selected = args.sections if args.sections else args.samples
    sections = filter_requested_samples(section_counts.index.astype(str).tolist(), selected)
    if args.dry_run:
        print_inventory([{'section': section, 'n_cells': int(section_counts.loc[section]), 'output': str(paths['h5ad'] / f'{safe_name(section)}.h5ad')} for section in sections])
        return
    meta = load_cluster_metadata(comp['cluster_metadata'], sections)
    ccf = load_ccf(comp['ccf'], meta.index)
    raw_backed = ad.read_h5ad(comp['raw_h5ad'], backed='r')
    input_files = [comp['raw_h5ad'], comp['cluster_metadata'], comp['ccf']]
    try:
        rows = [process_section(section, raw_backed, meta, ccf, args, paths, input_files) for section in sections]
    finally:
        raw_backed.file.close()
    manifest_path = append_manifest_rows(paths['manifest'], rows)
    logging.info('Wrote manifest %s', manifest_path)
if __name__ == '__main__':
    main()
