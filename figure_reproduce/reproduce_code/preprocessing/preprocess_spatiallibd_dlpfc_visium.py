from __future__ import annotations
import argparse
import gzip
import json
import logging
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
import numpy as np
import pandas as pd
import scanpy as sc
from preprocess_common import add_common_arguments, add_counts_layer, add_standard_obs, add_var_metadata, append_manifest_rows, apply_qc, ensure_processed_dirs, filter_requested_samples, manifest_row, preprocessing_uns, print_inventory, read_raw_manifest, set_labels, setup_logging, set_spatial, to_csr_counts, write_processed
DATASET_KEY = 'spatialLIBD_DLPFC_Visium'
SCRIPT_PATH = Path(__file__).resolve()
DONOR_REPLICATE = {'151507': ('Br5292', 'rep1'), '151508': ('Br5292', 'rep2'), '151509': ('Br5292', 'rep3'), '151510': ('Br5292', 'rep4'), '151669': ('Br5595', 'rep1'), '151670': ('Br5595', 'rep2'), '151671': ('Br5595', 'rep3'), '151672': ('Br5595', 'rep4'), '151673': ('Br8100', 'rep1'), '151674': ('Br8100', 'rep2'), '151675': ('Br8100', 'rep3'), '151676': ('Br8100', 'rep4')}

def discover_samples(raw_dataset_dir: Path) -> list[str]:
    manifest = read_raw_manifest(raw_dataset_dir)
    return sorted(manifest['SampleID'].astype(str).unique())

def find_coordinate_file(sample_dir: Path, sample_id: str) -> Path | None:
    candidates = [sample_dir / 'spatial' / 'tissue_positions.csv', sample_dir / 'spatial' / 'tissue_positions_list.csv', sample_dir / f'{sample_id}_tissue_positions.csv', sample_dir / f'{sample_id}_tissue_positions_list.csv']
    for path in candidates:
        if path.exists():
            return path
    return None

def find_cloupe_file(sample_dir: Path, sample_id: str) -> Path | None:
    candidates = [sample_dir / f'{sample_id}.cloupe']
    for path in candidates:
        if path.exists():
            return path
    return None

def find_annotation_file(raw_dataset_dir: Path, sample_id: str) -> Path | None:
    candidates = [raw_dataset_dir / 'metadata' / 'DLPFC_annotations' / f'{sample_id}_truth.txt', raw_dataset_dir / 'samples' / sample_id / f'{sample_id}_truth.txt', raw_dataset_dir / sample_id / f'{sample_id}_truth.txt']
    for path in candidates:
        if path.exists():
            return path
    return None

def read_positions(path: Path) -> pd.DataFrame:
    first = path.open().readline().strip().split(',')
    has_header = 'barcode' in first or 'pxl_col_in_fullres' in first or 'imagecol' in first
    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres'])
    col_map = {'pxl_col_in_fullres': 'x', 'imagecol': 'x', 'pxl_row_in_fullres': 'y', 'imagerow': 'y'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    required = {'barcode', 'x', 'y'}
    if not required.issubset(df.columns):
        raise ValueError(f'Coordinate file {path} lacks required columns {required}')
    return df.set_index('barcode')

def _read_cloupe_block(handle, desc: dict) -> bytes:
    handle.seek(int(desc['Start']))
    payload = handle.read(int(desc['End']) - int(desc['Start']))
    compression = int(desc.get('CompressionType', 0))
    if compression in {1, 2}:
        return gzip.decompress(payload)
    if compression == 0:
        return payload
    raise ValueError(f'Unsupported cloupe compression type {compression}')

def read_cloupe_spatial(path: Path) -> pd.DataFrame:
    with path.open('rb') as handle:
        header_bytes = handle.read(4096)
        header = json.loads(header_bytes.split(b'\x00', 1)[0])
        index = json.loads(_read_cloupe_block(handle, header['indexBlock']))
        if not index.get('Matrices'):
            raise ValueError(f'{path} has no matrix index')
        matrix = index['Matrices'][0]
        barcode_desc = matrix['Barcodes']
        barcode_bytes = _read_cloupe_block(handle, barcode_desc)
        width = int(barcode_desc['ArrayWidth'])
        if width <= 0:
            raise ValueError(f'{path} has invalid cloupe barcode width {width}')
        barcodes = [barcode_bytes[i:i + width].decode('utf-8').rstrip('\x00') for i in range(0, len(barcode_bytes), width)]
        spatial = next((proj for proj in index.get('Projections', []) if proj.get('Name') == 'Spatial'), None)
        if spatial is None:
            raise ValueError(f'{path} has no Spatial projection')
        dims = [int(x) for x in spatial['Dims']]
        if len(dims) != 2 or dims[0] < 2:
            raise ValueError(f'{path} Spatial projection has unexpected dims {dims}')
        values = np.frombuffer(_read_cloupe_block(handle, spatial['Matrix']), dtype='<f8')
    expected = int(np.prod(dims))
    if values.size != expected:
        raise ValueError(f'{path} Spatial projection has {values.size} values, expected {expected}')
    matrix_values = values.reshape(dims)
    if matrix_values.shape[1] != len(barcodes):
        raise ValueError(f'{path} Spatial projection and barcode count mismatch')
    positions = pd.DataFrame({'x': matrix_values[0, :], 'y': matrix_values[1, :], 'spot_diameter_fullres': matrix_values[2, :] if matrix_values.shape[0] > 2 else np.nan}, index=pd.Index(barcodes, name='barcode'))
    if positions.index.has_duplicates:
        duplicated = positions.index[positions.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f'{path} has duplicated cloupe barcodes, for example: {duplicated}')
    return positions

def read_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path, sep='\t', header=None, names=['barcode', 'layer'], dtype=str, keep_default_na=False)
    required = {'barcode', 'layer'}
    if not required.issubset(truth.columns):
        raise ValueError(f'Annotation file {path} lacks required columns {required}')
    if truth['barcode'].isna().any():
        raise ValueError(f'Annotation file {path} contains missing barcodes')
    truth['barcode'] = truth['barcode'].astype(str)
    truth['layer'] = truth['layer'].astype(str).str.strip()
    if truth['barcode'].duplicated().any():
        duplicated = truth.loc[truth['barcode'].duplicated(), 'barcode'].head().tolist()
        raise ValueError(f'Annotation file {path} has duplicated barcodes, for example: {duplicated}')
    missing_label = (truth['layer'].str.len() == 0) | truth['layer'].str.upper().isin({'NA', 'NAN'})
    truth.loc[missing_label, 'layer'] = 'unknown'
    return truth.set_index('barcode')

def process_one(sample_id: str, args, paths: dict[str, Path]) -> dict:
    raw_dataset_dir = args.raw_root / args.dataset_key
    sample_dir = raw_dataset_dir / 'samples' / sample_id
    matrix_h5 = sample_dir / f'{sample_id}_filtered_feature_bc_matrix.h5'
    coord_file = find_coordinate_file(sample_dir, sample_id)
    cloupe_file = find_cloupe_file(sample_dir, sample_id)
    annotation_file = find_annotation_file(raw_dataset_dir, sample_id)
    if coord_file is None and cloupe_file is None:
        raise FileNotFoundError(f'{sample_id} has no barcode-keyed tissue_positions file or cloupe Spatial projection. The downloaded raw tree needs one observed barcode-to-coordinate source.')
    output_path = paths['h5ad'] / f'{sample_id}.h5ad'
    adata = sc.read_10x_h5(matrix_h5, gex_only=False)
    adata.X = to_csr_counts(adata.X)
    add_counts_layer(adata, adata.X)
    gene_ids = adata.var['gene_ids'].astype(str).to_numpy() if 'gene_ids' in adata.var else adata.var_names.astype(str)
    add_var_metadata(adata, gene_ids=gene_ids, gene_symbols=adata.var_names.astype(str), species='human')
    if coord_file is not None:
        positions = read_positions(coord_file)
        coordinate_source = str(coord_file)
        coordinate_method = 'tissue_positions'
        missing = adata.obs_names.difference(positions.index)
        n_missing_coordinates = int(len(missing))
        extra_coordinates = positions.index.difference(adata.obs_names)
        if len(missing):
            raise ValueError(f'{sample_id}: {len(missing)} barcodes missing coordinates')
    else:
        positions = read_cloupe_spatial(cloupe_file)
        coordinate_source = str(cloupe_file)
        coordinate_method = 'cloupe_spatial_projection'
        missing = adata.obs_names.difference(positions.index)
        n_missing_coordinates = int(len(missing))
        extra_coordinates = positions.index.difference(adata.obs_names)
        if len(missing):
            logging.warning('%s: dropping %d filtered-matrix barcodes without cloupe spatial coordinates', sample_id, len(missing))
            keep = adata.obs_names.intersection(positions.index)
            if keep.empty:
                raise ValueError(f'{sample_id}: no filtered-matrix barcodes have cloupe spatial coordinates')
            adata = adata[keep, :].copy()
    if len(extra_coordinates):
        logging.warning('%s: ignoring %d coordinate barcodes not present in filtered matrix', sample_id, len(extra_coordinates))
    missing_after_coordinate_filter = adata.obs_names.difference(positions.index)
    if len(missing_after_coordinate_filter):
        raise ValueError(f'{sample_id}: {len(missing_after_coordinate_filter)} barcodes missing coordinates')
    positions = positions.loc[adata.obs_names]
    donor_id, replicate_id = DONOR_REPLICATE.get(sample_id, ('unknown', 'unknown'))
    add_standard_obs(adata, dataset_key=args.dataset_key, sample_id=sample_id, platform='10x Visium', species='human', tissue='dorsolateral prefrontal cortex', condition='DLPFC', donor_id=donor_id, replicate_id=replicate_id, section_id=sample_id)
    for col in ['in_tissue', 'array_row', 'array_col']:
        if col in positions:
            adata.obs[col] = positions[col].to_numpy()
    if 'spot_diameter_fullres' in positions:
        adata.obs['spot_diameter_fullres'] = positions['spot_diameter_fullres'].to_numpy()
    set_spatial(adata, positions[['x', 'y']].to_numpy(dtype=float))
    if annotation_file is None and (not args.allow_missing_labels):
        raise ValueError(f'{sample_id}: no observed layer/cell-type label found in the current raw tree. Add barcode-keyed layer annotations or pass --allow-missing-labels only for unknown-label QC output.')
    if annotation_file is None:
        layer_labels = pd.Series('unknown', index=adata.obs_names)
        label_source = 'not_available_in_raw_tree'
        label_status = 'missing_biological_region_cell_type_label'
        n_unknown_layer_labels = int(adata.n_obs)
    else:
        truth = read_truth(annotation_file)
        missing_labels = adata.obs_names.difference(truth.index)
        extra_labels = truth.index.difference(adata.obs_names)
        if len(missing_labels):
            raise ValueError(f'{sample_id}: {len(missing_labels)} filtered barcodes missing DLPFC layer labels')
        if len(extra_labels):
            logging.warning('%s: annotation file has %d extra barcodes not in filtered matrix', sample_id, len(extra_labels))
        layer_labels = truth.loc[adata.obs_names, 'layer'].astype(str)
        adata.obs['dlpfc_layer'] = layer_labels.to_numpy()
        n_unknown_layer_labels = int((layer_labels == 'unknown').sum())
        try:
            label_source = str(annotation_file.relative_to(raw_dataset_dir))
        except ValueError:
            label_source = str(annotation_file)
        label_status = 'observed_dlpfc_layer_annotation'
    cell_type = pd.Series('unknown', index=adata.obs_names)
    set_labels(adata, region=layer_labels, cell_type=cell_type, annotation=layer_labels, ground_truth=layer_labels, label_source=label_source)
    if annotation_file is not None and n_unknown_layer_labels:
        keep_labeled = adata.obs['region'].astype(str) != 'unknown'
        adata = adata[keep_labeled, :].copy()
    adata, stats = apply_qc(adata, min_counts=args.min_counts, min_genes=args.min_genes, min_cells_per_gene=args.min_cells_per_gene, max_pct_mt=args.max_pct_mt)
    input_files = [matrix_h5, coord_file if coord_file is not None else cloupe_file]
    if annotation_file is not None:
        input_files.append(annotation_file)
    adata.uns['preprocessing'] = preprocessing_uns(dataset_key=args.dataset_key, sample_id=sample_id, input_files=input_files, counts_source=matrix_h5.name, coordinate_source=coordinate_source, label_source=label_source, seed=args.seed, script_path=SCRIPT_PATH, extra={'label_status': label_status, 'coordinate_method': coordinate_method, 'n_barcodes_dropped_missing_coordinates': n_missing_coordinates, 'n_filtered_matrix_barcodes_missing_coordinates': n_missing_coordinates, 'n_coordinate_barcodes_ignored_not_in_matrix': int(len(extra_coordinates)), 'n_unknown_layer_labels': n_unknown_layer_labels, 'n_spots_removed_unknown_layer': n_unknown_layer_labels if annotation_file is not None else 0})
    write_processed(adata, output_path, overwrite=args.overwrite)
    row = manifest_row(dataset_key=args.dataset_key, sample_id=sample_id, input_files=input_files, output_h5ad=output_path, stats=stats, label_source=label_source, coordinate_source=coordinate_source, counts_source=matrix_h5.name, seed=args.seed, script_path=SCRIPT_PATH)
    row['coordinate_method'] = coordinate_method
    row['n_filtered_matrix_barcodes_missing_coordinates'] = n_missing_coordinates
    row['n_coordinate_barcodes_ignored_not_in_matrix'] = int(len(extra_coordinates))
    row['n_unknown_layer_labels'] = n_unknown_layer_labels
    row['n_spots_removed_unknown_layer'] = n_unknown_layer_labels if annotation_file is not None else 0
    return row

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, DATASET_KEY)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_dataset_dir = args.raw_root / args.dataset_key
    paths = ensure_processed_dirs(args.processed_root, args.dataset_key)
    setup_logging(paths['logs'], args.dataset_key)
    samples = filter_requested_samples(discover_samples(raw_dataset_dir), args.samples)
    inventory = []
    for sid in samples:
        sample_dir = raw_dataset_dir / 'samples' / sid
        coord_file = find_coordinate_file(sample_dir, sid)
        cloupe_file = find_cloupe_file(sample_dir, sid)
        annotation_file = find_annotation_file(raw_dataset_dir, sid)
        inventory.append({'sample_id': sid, 'matrix': str(sample_dir / f'{sid}_filtered_feature_bc_matrix.h5'), 'coordinate_file': str(coord_file), 'coordinate_file_exists': coord_file is not None, 'cloupe_file': str(cloupe_file), 'cloupe_file_exists': cloupe_file is not None, 'annotation_file': str(annotation_file), 'annotation_file_exists': annotation_file is not None})
    if args.dry_run:
        print_inventory(inventory)
        return
    rows = [process_one(sid, args, paths) for sid in samples]
    manifest_path = append_manifest_rows(paths['manifest'], rows)
    logging.info('Wrote manifest %s', manifest_path)
if __name__ == '__main__':
    main()
