from __future__ import annotations
import argparse
import gzip
import logging
import re
import tarfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from preprocess_common import add_common_arguments, add_counts_layer, add_standard_obs, add_var_metadata, append_manifest_rows, apply_qc, ensure_processed_dirs, filter_requested_samples, manifest_row, preprocessing_uns, print_inventory, set_labels, setup_logging, set_spatial, to_csr_counts, write_processed
DATASET_KEY = 'GSE269617'
SCRIPT_PATH = Path(__file__).resolve()

class SkippedSample(RuntimeError):

    def __init__(self, sample_id: str, reason: str) -> None:
        super().__init__(reason)
        self.sample_id = sample_id
        self.reason = reason

def tar_path(raw_dataset_dir: Path) -> Path:
    return raw_dataset_dir / 'components' / 'GSE269617_RAW.tar'

def discover_samples(tar_file: Path) -> list[str]:
    samples = set()
    with tarfile.open(tar_file, 'r') as tf:
        for member in tf.getnames():
            match = re.match('^(GSM\\d+_.+)_cell_by_gene\\.csv\\.gz$', member)
            if match:
                samples.add(match.group(1))
    return sorted(samples)

def read_csv_member(tf: tarfile.TarFile, member: str) -> pd.DataFrame:
    fh = tf.extractfile(member)
    if fh is None:
        raise FileNotFoundError(member)
    with gzip.GzipFile(fileobj=fh) as gz:
        return pd.read_csv(gz, dtype={'cell': str, 'EntityID': str})

def validate_expression_table(sample_id: str, expr: pd.DataFrame) -> None:
    if 'cell' not in expr.columns:
        first_col = str(expr.columns[0]) if len(expr.columns) else 'none'
        raise SkippedSample(sample_id, f"cell_by_gene table uses '{first_col}' instead of 'cell'; observed values are not raw count backed")
    genes = [c for c in expr.columns if c != 'cell']
    values = expr[genes].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and (not np.allclose(finite, np.round(finite))):
        raise SkippedSample(sample_id, 'cell_by_gene values are not integer-compatible raw counts')

def parse_sample_id(sample_id: str) -> tuple[str, str, str, str]:
    label = sample_id.split('_', 1)[1] if '_' in sample_id else sample_id
    match = re.match('^(E\\d+|MIA|Abx)([MF])_(\\d+)$', label)
    if not match:
        return ('unknown', 'unknown', 'unknown', 'unknown')
    token, sex, replicate = match.groups()
    if token.startswith('E'):
        return (token, 'PBS', sex, replicate)
    return ('E14', token, sex, replicate)

def process_one(sample_id: str, args, paths: dict[str, Path]) -> dict:
    raw_dataset_dir = args.raw_root / args.dataset_key
    tpath = tar_path(raw_dataset_dir)
    output_path = paths['h5ad'] / f'{sample_id}.h5ad'
    logging.info('Processing %s', sample_id)
    expr_member = f'{sample_id}_cell_by_gene.csv.gz'
    meta_member = f'{sample_id}_cell_metadata.csv.gz'
    transcript_member = f'{sample_id}_detected_transcripts.csv.gz'
    with tarfile.open(tpath, 'r') as tf:
        expr = read_csv_member(tf, expr_member)
        meta = read_csv_member(tf, meta_member)
    validate_expression_table(sample_id, expr)
    expr['cell'] = expr['cell'].astype(str)
    meta['EntityID'] = meta['EntityID'].astype(str)
    genes = [c for c in expr.columns if c != 'cell']
    obs_names = expr['cell'].astype(str)
    X = to_csr_counts(expr[genes].to_numpy())
    obs = pd.DataFrame(index=obs_names)
    var = pd.DataFrame(index=pd.Index(genes, dtype='object'))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    add_counts_layer(adata, X)
    add_var_metadata(adata, species='mouse')
    meta = meta.set_index('EntityID').reindex(adata.obs_names)
    if meta.isna().all(axis=1).any():
        missing = int(meta.isna().all(axis=1).sum())
        raise ValueError(f'{missing} expression cells missing from metadata for {sample_id}')
    for col in meta.columns:
        adata.obs[col] = meta[col].to_numpy()
    developmental_stage, treatment, sex, replicate = parse_sample_id(sample_id)
    add_standard_obs(adata, dataset_key=args.dataset_key, sample_id=sample_id, platform='targeted spatial transcriptomics', species='mouse', tissue='brain', condition=treatment, donor_id=f'{developmental_stage}{treatment}{sex}', replicate_id=replicate, section_id=sample_id)
    adata.obs['sex'] = sex
    adata.obs['developmental_stage'] = developmental_stage
    adata.obs['treatment'] = treatment
    adata.obs['condition_label'] = treatment
    set_spatial(adata, meta[['center_x', 'center_y']].to_numpy(dtype=float))
    adata, stats = apply_qc(adata, min_counts=args.min_counts, min_genes=args.min_genes, min_cells_per_gene=args.min_cells_per_gene, max_pct_mt=args.max_pct_mt)
    if adata.n_obs == 0:
        raise SkippedSample(sample_id, 'no cells pass shared QC defaults')
    if not args.allow_missing_labels:
        raise SkippedSample(sample_id, 'no observed biological region/cell-type label found')
    labels = pd.Series('unknown', index=adata.obs_names)
    label_source = 'not_available_in_raw_tree'
    set_labels(adata, region=labels, cell_type=pd.Series('unknown', index=adata.obs_names), annotation=labels, ground_truth=labels, label_source=label_source)
    adata.uns['preprocessing'] = preprocessing_uns(dataset_key=args.dataset_key, sample_id=sample_id, input_files=[tpath], counts_source=expr_member, coordinate_source=f'{meta_member}:center_x,center_y', label_source=label_source, seed=args.seed, script_path=SCRIPT_PATH, extra={'label_status': 'missing_biological_region_cell_type_label', 'raw_references': {'detected_transcripts': transcript_member}})
    write_processed(adata, output_path, overwrite=args.overwrite)
    return manifest_row(dataset_key=args.dataset_key, sample_id=sample_id, input_files=[tpath], output_h5ad=output_path, stats=stats, label_source=label_source, coordinate_source=f'{meta_member}:center_x,center_y', counts_source=expr_member, seed=args.seed, script_path=SCRIPT_PATH)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, DATASET_KEY)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_dataset_dir = args.raw_root / args.dataset_key
    paths = ensure_processed_dirs(args.processed_root, args.dataset_key)
    setup_logging(paths['logs'], args.dataset_key)
    samples = discover_samples(tar_path(raw_dataset_dir))
    samples = filter_requested_samples(samples, args.samples)
    if args.dry_run:
        print_inventory([{'sample_id': sid, 'expression_member': f'{sid}_cell_by_gene.csv.gz', 'metadata_member': f'{sid}_cell_metadata.csv.gz', 'transcript_member': f'{sid}_detected_transcripts.csv.gz'} for sid in samples])
        return
    rows = []
    skipped = []
    for sid in samples:
        try:
            rows.append(process_one(sid, args, paths))
        except SkippedSample as exc:
            logging.warning('Skipping %s: %s', exc.sample_id, exc.reason)
            skipped.append({'sample_id': exc.sample_id, 'reason': exc.reason})
    if skipped:
        skipped_path = paths['manifest'] / 'skipped_samples.tsv'
        pd.DataFrame(skipped).to_csv(skipped_path, sep='\t', index=False)
        logging.warning('Wrote skipped sample manifest %s', skipped_path)
    if not rows:
        raise RuntimeError('No GSE269617 samples produced downstream-ready h5ad outputs')
    manifest_path = append_manifest_rows(paths['manifest'], rows)
    logging.info('Wrote manifest %s', manifest_path)
if __name__ == '__main__':
    main()
