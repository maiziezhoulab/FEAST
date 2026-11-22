"""Recover marker-rule telencephalon region annotations for GSE269617.

This is a production recovery of the deleted lab6
``annotate_telencephalon_regions.py`` workflow. The recovered evidence shows
that the old annotation used a compact marker panel, robust marker scaling,
spatial marker smoothing, optional author-label centroid calibration from two
AbxF samples, and kNN majority smoothing of final labels.
"""
from __future__ import annotations
import argparse
import gzip
import itertools
import json
import logging
import os
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
ACCESSION = 'GSE269617'
PROCESSED_ROOT = Path(__file__).resolve().parents[2] / 'Datasets' / 'Processed'
RAW_TAR = Path(__file__).resolve().parents[2] / 'Datasets' / 'Raw' / 'GSE269617' / 'components' / 'GSE269617_RAW.tar'
SCRIPT_PATH = Path(__file__).resolve()
MARKER_GENES = ('Pax6', 'Cux2', 'Lhx5', 'Tle4', 'Fezf2', 'Ascl1', 'Gad1', 'Pnoc', 'Sst')
REGION_LABELS = ('DorsalVZ', 'IZ', 'CP', 'GE', 'SeptalVZ', 'VentralVZ', 'Septum', 'BT')
OUTPUT_LABELS = REGION_LABELS + ('Other',)
AUTHOR_SAMPLES = ('GSM8323060_AbxF_1', 'GSM8323063_AbxF_4')
REGION_ID = {'Background': 0, 'DorsalVZ': 1, 'IZ': 2, 'CP': 3, 'VentralVZ': 4, 'GE': 5, 'SeptalVZ': 6, 'Septum': 7, 'BT': 8, 'Other': 9}

@dataclass
class AnchorData:
    sample_id: str
    coords: np.ndarray
    markers: np.ndarray
    truth: np.ndarray

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-root', type=Path, default=PROCESSED_ROOT)
    parser.add_argument('--dataset-key', default=ACCESSION)
    parser.add_argument('--input-subdir', default='h5ad')
    parser.add_argument('--output-subdir', default='h5ad_region_annotated')
    parser.add_argument('--annotation-subdir', default='region_annotations')
    parser.add_argument('--schema-path', type=Path, default=None)
    parser.add_argument('--raw-tar', type=Path, default=RAW_TAR)
    parser.add_argument('--author-samples', nargs='*', default=list(AUTHOR_SAMPLES))
    parser.add_argument('--samples', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--smooth-k-grid', type=int, nargs='+', default=[15, 31, 51])
    parser.add_argument('--percentile-grid', type=float, nargs='+', default=[95.0, 99.0])
    parser.add_argument('--label-smooth-k', type=int, default=25)
    parser.add_argument('--label-majority', type=float, default=0.55)
    parser.add_argument('--other-quantile', type=float, default=0.01)
    parser.add_argument('--fallback-other-threshold', type=float, default=0.05)
    parser.add_argument('--min-qc-accuracy', type=float, default=0.6)
    parser.add_argument('--no-author-centroids', action='store_true')
    parser.add_argument('--write-marker-scores', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def environment_summary() -> str:
    return json.dumps({'python': sys.version.split()[0], 'executable': sys.executable, 'platform': sys.platform, 'conda_prefix': os.environ.get('CONDA_PREFIX', '')}, sort_keys=True)

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'annotate_telencephalon_regions_{stamp}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    return log_path

def discover_inputs(input_dir: Path, requested: list[str] | None, limit: int | None) -> list[Path]:
    paths = sorted(input_dir.glob('GSM*.h5ad'))
    if requested:
        by_sample = {p.stem: p for p in paths}
        missing = sorted(set(requested).difference(by_sample))
        if missing:
            raise FileNotFoundError(f'Requested sample h5ads missing under {input_dir}: {missing}')
        paths = [by_sample[s] for s in requested]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f'No sample h5ads found under {input_dir}')
    return paths

def load_schema(path: Path) -> pd.DataFrame:
    schema = pd.read_csv(path, sep='\t')
    if not {'region_id', 'region_label'}.issubset(schema.columns):
        raise ValueError(f'Schema lacks region_id/region_label columns: {path}')
    return schema

def get_spatial(adata: ad.AnnData, sample_id: str) -> np.ndarray:
    if 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
    elif {'center_x', 'center_y'}.issubset(adata.obs.columns):
        coords = adata.obs[['center_x', 'center_y']].to_numpy(dtype=np.float32)
    elif {'x', 'y'}.issubset(adata.obs.columns):
        coords = adata.obs[['x', 'y']].to_numpy(dtype=np.float32)
    else:
        raise KeyError(f"{sample_id} has no obsm['spatial'] or x/y coordinate columns")
    if coords.shape != (adata.n_obs, 2):
        raise ValueError(f'{sample_id} spatial shape {coords.shape}, expected {(adata.n_obs, 2)}')
    return coords

def extract_lognorm_markers_from_matrix(X: Any, var_names: pd.Index, *, total_counts: np.ndarray | None=None) -> np.ndarray:
    marker_idx = [int(np.where(var_names == gene)[0][0]) for gene in MARKER_GENES]
    marker_counts = X[:, marker_idx]
    if sparse.issparse(marker_counts):
        marker_counts = marker_counts.toarray()
    marker_counts = np.asarray(marker_counts, dtype=np.float32)
    if total_counts is None:
        total_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    total_counts = np.asarray(total_counts, dtype=np.float32)
    total_counts[total_counts <= 0] = 1.0
    scale = float(np.median(total_counts[total_counts > 0])) if np.any(total_counts > 0) else 1.0
    marker_norm = marker_counts / total_counts[:, None] * scale
    return np.log1p(marker_norm).astype(np.float32)

def extract_lognorm_markers_from_frame(expr: pd.DataFrame) -> np.ndarray:
    missing = [gene for gene in MARKER_GENES if gene not in expr.columns]
    if missing:
        raise KeyError(f'Raw expression table missing marker genes: {missing}')
    values = expr.to_numpy(dtype=np.float32, copy=False)
    totals = values.sum(axis=1).astype(np.float32)
    markers = expr.loc[:, MARKER_GENES].to_numpy(dtype=np.float32, copy=False)
    totals[totals <= 0] = 1.0
    scale = float(np.median(totals[totals > 0])) if np.any(totals > 0) else 1.0
    return np.log1p(markers / totals[:, None] * scale).astype(np.float32)

def robust_scale(values: np.ndarray, percentile: float) -> np.ndarray:
    denom = np.nanpercentile(values, float(percentile), axis=0).astype(np.float32)
    denom[~np.isfinite(denom) | (denom <= 0)] = 1.0
    return np.clip(values / denom[None, :], 0.0, 1.0).astype(np.float32)

def smooth_values(coords: np.ndarray, values: np.ndarray, k: int, chunk_size: int=50000) -> np.ndarray:
    if k <= 1 or coords.shape[0] <= 1:
        return values.astype(np.float32, copy=True)
    k = min(int(k), coords.shape[0])
    tree = cKDTree(coords)
    out = np.empty_like(values, dtype=np.float32)
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        _, idx = tree.query(coords[start:end], k=k, workers=-1)
        if idx.ndim == 1:
            idx = idx[:, None]
        out[start:end] = values[idx].mean(axis=1, dtype=np.float32)
    return out

def compute_region_scores(smoothed_markers: np.ndarray) -> pd.DataFrame:
    marker = {gene: smoothed_markers[:, idx] for idx, gene in enumerate(MARKER_GENES)}
    cp_score = np.maximum.reduce([marker['Lhx5'], marker['Tle4'], marker['Fezf2']])
    vz_score = np.maximum(marker['Pax6'], marker['Ascl1'])
    septal_vz_score = np.sqrt(np.clip(marker['Pnoc'], 0, None) * np.clip(vz_score, 0, None))
    ge_score = np.clip(marker['Gad1'] - 0.35 * marker['Ascl1'], 0.0, None)
    scores = pd.DataFrame({'DorsalVZ': marker['Pax6'], 'IZ': marker['Cux2'], 'CP': cp_score, 'GE': ge_score, 'SeptalVZ': septal_vz_score, 'VentralVZ': marker['Ascl1'], 'Septum': marker['Pnoc'], 'BT': marker['Sst']})
    return scores.astype(np.float32)

def fit_marker_centroids(features: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    for label in REGION_LABELS:
        mask = truth == label
        if mask.any():
            rows.append(features[mask].mean(axis=0).astype(np.float32))
        else:
            rows.append(np.zeros(features.shape[1], dtype=np.float32))
    centroids = pd.DataFrame(rows, index=REGION_LABELS, columns=MARKER_GENES)
    return centroids

def centroid_similarity_scores(features: np.ndarray, centroids: pd.DataFrame) -> pd.DataFrame:
    centroid_values = centroids.loc[list(REGION_LABELS), list(MARKER_GENES)].to_numpy(dtype=np.float32)
    feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
    centroid_norm = np.linalg.norm(centroid_values, axis=1, keepdims=True).T
    safe_features = features / np.maximum(feature_norm, 1e-08)
    safe_centroids = centroid_values / np.maximum(centroid_norm.T, 1e-08)
    similarities = safe_features @ safe_centroids.T
    similarities[feature_norm[:, 0] <= 0] = 0.0
    return pd.DataFrame(np.clip(similarities, 0.0, 1.0), columns=REGION_LABELS)

def predict_labels(scores: pd.DataFrame, other_threshold: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_values = scores.loc[:, list(REGION_LABELS)].to_numpy(dtype=np.float32)
    best_idx = np.argmax(score_values, axis=1)
    best_score = score_values[np.arange(score_values.shape[0]), best_idx]
    partitioned = np.partition(score_values, -2, axis=1)
    second_score = partitioned[:, -2]
    labels = np.asarray(REGION_LABELS, dtype=object)[best_idx].astype(object)
    if other_threshold is not None:
        labels[best_score < float(other_threshold)] = 'Other'
    confidence = best_score - second_score
    return (labels.astype(str), best_score.astype(np.float32), confidence.astype(np.float32))

def smooth_label_majority(coords: np.ndarray, labels: np.ndarray, k: int, majority_fraction: float, chunk_size: int=50000) -> tuple[np.ndarray, np.ndarray]:
    if k <= 1 or coords.shape[0] <= 1:
        return (labels.astype(str), np.ones(labels.shape[0], dtype=np.float32))
    label_to_int = {label: idx for idx, label in enumerate(OUTPUT_LABELS)}
    int_labels = np.array([label_to_int[str(label)] for label in labels], dtype=np.int16)
    k = min(int(k), coords.shape[0])
    tree = cKDTree(coords)
    out = int_labels.copy()
    fractions = np.ones(labels.shape[0], dtype=np.float32)
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        _, idx = tree.query(coords[start:end], k=k, workers=-1)
        if idx.ndim == 1:
            idx = idx[:, None]
        neighbor_labels = int_labels[idx]
        counts = np.zeros((neighbor_labels.shape[0], len(OUTPUT_LABELS)), dtype=np.int16)
        for code in range(len(OUTPUT_LABELS)):
            counts[:, code] = np.sum(neighbor_labels == code, axis=1)
        mode = counts.argmax(axis=1).astype(np.int16)
        frac = counts[np.arange(counts.shape[0]), mode].astype(np.float32) / float(k)
        fractions[start:end] = frac
        update = frac >= float(majority_fraction)
        out[start:end][update] = mode[update]
    return (np.asarray(OUTPUT_LABELS, dtype=object)[out].astype(str), fractions)

def collapse_author_region(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return ''
    cleaned = text.replace('__', '_').replace('-', '_').replace(' ', '_')
    parts = [part for part in cleaned.split('_') if part]
    if parts and parts[0].lower() in {'l', 'r', 'left', 'right'}:
        parts = parts[1:]
    compact = '_'.join(parts).lower().replace('_', '')
    if compact in {'dorsalvz', 'dorsvz', 'dorvz', 'dvz'}:
        return 'DorsalVZ'
    if compact in {'ventralvz', 'ventvz', 'venvz', 'vvz'}:
        return 'VentralVZ'
    if compact in {'septalvz', 'septvz', 'svz'}:
        return 'SeptalVZ'
    if compact in {'cp', 'corticalplate'}:
        return 'CP'
    if compact in {'iz', 'intermediatezone'}:
        return 'IZ'
    if compact in {'ge', 'lge', 'mge', 'cge', 'ganglioniceminence'}:
        return 'GE'
    if compact in {'septum', 'septal'}:
        return 'Septum'
    if compact in {'bt', 'basaltelencephalon', 'basallateral', 'basal'}:
        return 'BT'
    return ''

def classification_metrics(truth: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for label in REGION_LABELS:
        tp = int(((truth == label) & (pred == label)).sum())
        fp = int(((truth != label) & (pred == label)).sum())
        fn = int(((truth == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
        rows.append({'region': label, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1})
    return pd.DataFrame(rows)

def read_csv_member(tf: tarfile.TarFile, member: str, **kwargs: Any) -> pd.DataFrame:
    fh = tf.extractfile(member)
    if fh is None:
        raise FileNotFoundError(member)
    with gzip.GzipFile(fileobj=fh) as gz:
        return pd.read_csv(gz, **kwargs)

def load_author_anchors(raw_tar: Path, sample_ids: list[str]) -> list[AnchorData]:
    anchors: list[AnchorData] = []
    if not raw_tar.exists():
        logging.warning('Raw tar unavailable for author anchors: %s', raw_tar)
        return anchors
    with tarfile.open(raw_tar, 'r') as tf:
        for sample_id in sample_ids:
            expr_member = f'{sample_id}_cell_by_gene.csv.gz'
            meta_member = f'{sample_id}_cell_metadata.csv.gz'
            try:
                expr = read_csv_member(tf, expr_member, index_col=0)
                meta = read_csv_member(tf, meta_member, index_col=0)
            except FileNotFoundError:
                logging.warning('Author anchor sample missing in raw tar: %s', sample_id)
                continue
            common = expr.index.astype(str).intersection(meta.index.astype(str))
            expr.index = expr.index.astype(str)
            meta.index = meta.index.astype(str)
            expr = expr.loc[common]
            meta = meta.loc[common]
            label_col = 'Custom cell groups' if 'Custom cell groups' in meta.columns else 'region_label'
            truth = meta[label_col].map(collapse_author_region).astype(str).to_numpy()
            mask = np.isin(truth, REGION_LABELS)
            if not mask.any():
                logging.warning('Author anchor sample has no recognized labels: %s', sample_id)
                continue
            coords = meta.loc[mask, ['center_x', 'center_y']].to_numpy(dtype=np.float32)
            markers = extract_lognorm_markers_from_frame(expr.loc[mask])
            anchors.append(AnchorData(sample_id=sample_id, coords=coords, markers=markers, truth=truth[mask]))
            logging.info('Loaded author anchor %s: %d labeled cells', sample_id, int(mask.sum()))
    return anchors

def evaluate_candidate(anchors: list[AnchorData], smooth_k: int, percentile: float) -> dict[str, Any]:
    smoothed: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    for anchor in anchors:
        scaled = robust_scale(anchor.markers, percentile)
        smoothed[anchor.sample_id] = smooth_values(anchor.coords, scaled, smooth_k)
        truths[anchor.sample_id] = anchor.truth
    if not smoothed:
        return {'smooth_k': smooth_k, 'percentile': percentile, 'n_author_cells': 0, 'accuracy': np.nan, 'macro_f1': np.nan, 'centroids': pd.DataFrame(), 'best_scores': np.array([], dtype=np.float32), 'truth': np.array([], dtype=str), 'pred': np.array([], dtype=str)}
    truth_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    sample_ids = list(smoothed)
    if len(sample_ids) > 1:
        for test_id in sample_ids:
            train_ids = [sid for sid in sample_ids if sid != test_id]
            train_features = np.concatenate([smoothed[sid] for sid in train_ids])
            train_truth = np.concatenate([truths[sid] for sid in train_ids])
            centroids = fit_marker_centroids(train_features, train_truth)
            scores = centroid_similarity_scores(smoothed[test_id], centroids)
            pred, _, _ = predict_labels(scores, other_threshold=None)
            truth_chunks.append(truths[test_id])
            pred_chunks.append(pred)
    else:
        only_id = sample_ids[0]
        centroids = fit_marker_centroids(smoothed[only_id], truths[only_id])
        scores = centroid_similarity_scores(smoothed[only_id], centroids)
        pred, _, _ = predict_labels(scores, other_threshold=None)
        truth_chunks.append(truths[only_id])
        pred_chunks.append(pred)
    truth_all = np.concatenate(truth_chunks)
    pred_all = np.concatenate(pred_chunks)
    accuracy = float((truth_all == pred_all).mean())
    metrics = classification_metrics(truth_all, pred_all)
    macro_f1 = float(metrics['f1'].mean(skipna=True))
    final_features = np.concatenate([smoothed[sid] for sid in sample_ids])
    final_truth = np.concatenate([truths[sid] for sid in sample_ids])
    final_centroids = fit_marker_centroids(final_features, final_truth)
    final_scores = centroid_similarity_scores(final_features, final_centroids)
    _, best_scores, _ = predict_labels(final_scores, other_threshold=None)
    return {'smooth_k': smooth_k, 'percentile': percentile, 'n_author_cells': int(final_truth.size), 'accuracy': accuracy, 'macro_f1': macro_f1, 'centroids': final_centroids, 'best_scores': best_scores, 'truth': truth_all, 'pred': pred_all}

def choose_parameters(args: argparse.Namespace, annotation_dir: Path) -> dict[str, Any]:
    anchors = [] if args.no_author_centroids else load_author_anchors(args.raw_tar, args.author_samples)
    payloads = [evaluate_candidate(anchors, smooth_k, percentile) for smooth_k, percentile in itertools.product(args.smooth_k_grid, args.percentile_grid)]
    metrics = pd.DataFrame([{'smooth_k': item['smooth_k'], 'percentile': item['percentile'], 'n_author_cells': item['n_author_cells'], 'accuracy': item['accuracy'], 'macro_f1': item['macro_f1']} for item in payloads])
    annotation_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(annotation_dir / 'author_anchor_candidate_metrics.tsv', sep='\t', index=False)
    usable = metrics.dropna(subset=['macro_f1', 'accuracy'])
    if usable.empty:
        logging.warning('No author-anchor calibration available; using recovered marker score rules.')
        return {'smooth_k': int(args.smooth_k_grid[0]), 'percentile': float(args.percentile_grid[0]), 'other_threshold': float(args.fallback_other_threshold), 'centroids': None, 'candidate_metrics': metrics, 'classifier': 'marker_score_rules', 'author_anchor_status': 'not_available'}
    best_row = usable.sort_values(['macro_f1', 'accuracy', 'n_author_cells'], ascending=False).iloc[0]
    best = next((item for item in payloads if int(item['smooth_k']) == int(best_row['smooth_k']) and float(item['percentile']) == float(best_row['percentile'])))
    best_scores = best['best_scores']
    other_threshold = float(np.quantile(best_scores, args.other_quantile)) if best_scores.size else float(args.fallback_other_threshold)
    other_threshold = float(np.clip(other_threshold, 0.1, 0.65))
    confusion = pd.crosstab(pd.Series(best['truth'], name='author_region'), pd.Series(best['pred'], name='predicted_region')).reindex(index=REGION_LABELS, columns=REGION_LABELS, fill_value=0)
    confusion.to_csv(annotation_dir / 'author_anchor_confusion_matrix.tsv', sep='\t')
    status = 'passed' if float(best_row['accuracy']) >= float(args.min_qc_accuracy) else 'below_min_qc_accuracy'
    if status != 'passed':
        logging.warning('Best author-anchor accuracy %.3f is below threshold %.3f; continuing with explicit QC status.', float(best_row['accuracy']), float(args.min_qc_accuracy))
    return {'smooth_k': int(best_row['smooth_k']), 'percentile': float(best_row['percentile']), 'other_threshold': other_threshold, 'centroids': best['centroids'], 'candidate_metrics': metrics, 'classifier': 'author_marker_centroid', 'author_anchor_status': status, 'author_anchor_accuracy': float(best_row['accuracy']), 'author_anchor_macro_f1': float(best_row['macro_f1']), 'author_anchor_cells': int(best_row['n_author_cells'])}

def parse_target_stage(sample_id: str) -> tuple[str, str]:
    if '_E14' in sample_id:
        return ('E14', 'E15.5')
    if '_E18' in sample_id:
        return ('E18', 'E18.5')
    return ('unknown', 'unknown')

def annotate_adata(adata: ad.AnnData, sample_id: str, params: dict[str, Any]) -> tuple[ad.AnnData, pd.DataFrame]:
    for gene in MARKER_GENES:
        if gene not in adata.var_names:
            raise KeyError(f'{sample_id} missing marker gene {gene}')
    coords = get_spatial(adata, sample_id)
    markers = extract_lognorm_markers_from_matrix(adata.X, adata.var_names)
    scaled = robust_scale(markers, params['percentile'])
    smoothed = smooth_values(coords, scaled, int(params['smooth_k']))
    centroids = params.get('centroids')
    if isinstance(centroids, pd.DataFrame):
        scores = centroid_similarity_scores(smoothed, centroids)
    else:
        scores = compute_region_scores(smoothed)
    labels, best_score, confidence = predict_labels(scores, other_threshold=float(params['other_threshold']))
    labels, majority_fraction = smooth_label_majority(coords, labels, int(params['label_smooth_k']), float(params['label_majority']))
    raw_stage, target_stage = parse_target_stage(sample_id)
    region_id = np.array([REGION_ID[label] for label in labels], dtype=np.int16)
    label_source = f"recovered_lab6_{params['classifier']}"
    method = f"{params['classifier']};marker_genes={','.join(MARKER_GENES)};marker_smoothing:k={int(params['smooth_k'])};percentile={float(params['percentile']):g};label_k={int(params['label_smooth_k'])};label_majority={float(params['label_majority']):g};other_threshold={float(params['other_threshold']):.5f}"
    blank = np.array([''] * adata.n_obs, dtype=object)
    adata.obs['region_author_raw'] = blank
    adata.obs['region_author_collapsed'] = blank
    adata.obs['region_id'] = region_id
    for col in ('region', 'annotation', 'ground_truth', 'region_pred'):
        adata.obs[col] = pd.Categorical(labels, categories=OUTPUT_LABELS)
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = pd.Categorical(['unknown'] * adata.n_obs)
    adata.obs['region_score'] = best_score
    adata.obs['region_confidence'] = confidence
    adata.obs['region_neighbor_majority'] = majority_fraction
    adata.obs['region_annotation_method'] = method
    adata.obs['label_source'] = label_source
    adata.obs['annotation_recovery_method'] = label_source
    adata.obs['annotation_recovery_confidence'] = confidence
    adata.obs['devccf_target_stage'] = target_stage
    adata.uns['region_annotation'] = {'ontology': list(OUTPUT_LABELS), 'region_id': REGION_ID, 'marker_genes': list(MARKER_GENES), 'classifier': params['classifier'], 'author_anchor_status': params.get('author_anchor_status', 'not_available'), 'region_score_rules': {'DorsalVZ': 'Pax6', 'IZ': 'Cux2', 'CP': 'max(Lhx5, Tle4, Fezf2)', 'GE': 'Gad1 - 0.35 * Ascl1', 'SeptalVZ': 'sqrt(Pnoc * max(Pax6, Ascl1))', 'VentralVZ': 'Ascl1', 'Septum': 'Pnoc', 'BT': 'Sst', 'Other': 'best region score below calibrated threshold'}, 'method': method, 'note': 'Recovered from deleted lab6 marker-rule annotation evidence; original author ROI polygons are unavailable for PBS samples.'}
    adata.uns['annotation_recovery'] = {'dataset_key': ACCESSION, 'method': label_source, 'raw_stage': raw_stage, 'devccf_target_stage': target_stage, 'counts_preserved': True, 'obs_names_preserved': True, 'created_at': now_iso(), 'script': str(SCRIPT_PATH), 'environment': environment_summary()}
    marker_df = pd.DataFrame(smoothed, columns=[f'{gene}_smoothed' for gene in MARKER_GENES], index=adata.obs_names)
    return (adata, marker_df)

def sample_summary(adata: ad.AnnData, sample_id: str, output_path: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
    counts = adata.obs['region_pred'].astype(str).value_counts().reindex(OUTPUT_LABELS, fill_value=0)
    rows = []
    for label, n_cells in counts.items():
        rows.append({'dataset_key': ACCESSION, 'sample_id': sample_id, 'output_h5ad': str(output_path), 'region_id': REGION_ID[label], 'region_pred': label, 'n_cells': int(n_cells), 'sample_total_cells': int(adata.n_obs), 'proportion': float(n_cells) / float(adata.n_obs), 'classifier': params['classifier'], 'author_anchor_status': params.get('author_anchor_status', 'not_available')})
    return rows

def export_per_cell_csv(adata: ad.AnnData, sample_id: str, out_path: Path) -> None:
    columns = ['sample_id', 'developmental_stage', 'sex', 'treatment', 'region_id', 'region_pred', 'region_score', 'region_confidence', 'region_neighbor_majority', 'region_annotation_method']
    available = [col for col in columns if col in adata.obs.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = adata.obs[available].copy()
    table.insert(0, 'cell_barcode', adata.obs_names.astype(str))
    table.to_csv(out_path, index=False)

def write_json(path: Path, payload: dict[str, Any]) -> None:

    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient='list')
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items() if k != 'centroids'}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2), encoding='utf-8')

def main() -> None:
    args = parse_args()
    root = args.processed_root / args.dataset_key
    input_dir = root / args.input_subdir
    output_dir = root / args.output_subdir
    annotation_dir = root / args.annotation_subdir
    manifest_dir = root / 'manifest'
    log_path = setup_logging(root / 'logs')
    if args.schema_path is None:
        args.schema_path = args.processed_root / 'DevCCFv1_figshare_26377171' / 'coordinate_system' / 'GSE269617_region_merged' / 'gse269617_region_schema.tsv'
    inputs = discover_inputs(input_dir, args.samples, args.limit)
    schema = load_schema(args.schema_path)
    params = choose_parameters(args, annotation_dir)
    params['label_smooth_k'] = int(args.label_smooth_k)
    params['label_majority'] = float(args.label_majority)
    params['schema_path'] = str(args.schema_path)
    params['raw_tar'] = str(args.raw_tar)
    params['script'] = str(SCRIPT_PATH)
    params['log_path'] = str(log_path)
    write_json(annotation_dir / 'annotation_parameters.json', params)
    if args.dry_run:
        print(json.dumps({'inputs': [str(p) for p in inputs], 'params': {k: v for k, v in params.items() if k != 'centroids'}}, indent=2, default=str))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    proportion_rows: list[dict[str, Any]] = []
    for input_path in inputs:
        sample_id = input_path.stem
        output_path = output_dir / input_path.name
        if output_path.exists() and (not args.overwrite):
            raise FileExistsError(f'Output exists; pass --overwrite: {output_path}')
        logging.info('Annotating %s', sample_id)
        adata = ad.read_h5ad(input_path)
        before_counts = (adata.layers['counts'] if 'counts' in adata.layers else adata.X).copy()
        adata, marker_df = annotate_adata(adata, sample_id, params)
        after_counts = adata.layers['counts'] if 'counts' in adata.layers else adata.X
        if (before_counts != after_counts).nnz if sparse.issparse(before_counts) else not np.array_equal(before_counts, after_counts):
            raise RuntimeError(f'{sample_id}: expression/count matrix changed during annotation')
        adata.write_h5ad(output_path, compression='gzip')
        export_per_cell_csv(adata, sample_id, annotation_dir / 'per_sample' / f'{sample_id}_regions.csv')
        if args.write_marker_scores:
            marker_df.to_csv(annotation_dir / 'per_sample' / f'{sample_id}_smoothed_marker_scores.csv.gz', index=True)
        raw_stage, target_stage = parse_target_stage(sample_id)
        labels = adata.obs['region_pred'].astype(str)
        non_other = float(labels.isin(REGION_LABELS).mean())
        manifest_rows.append({'dataset_key': ACCESSION, 'sample_id': sample_id, 'input_h5ad': str(input_path), 'output_h5ad': str(output_path), 'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars), 'n_labeled': int(adata.n_obs), 'n_unlabeled': 0, 'raw_stage': raw_stage, 'devccf_target_stage': target_stage, 'label_source': f"recovered_lab6_{params['classifier']}", 'annotation_recovery_method': f"recovered_lab6_{params['classifier']}", 'author_anchor_status': params.get('author_anchor_status', 'not_available'), 'non_other_fraction': non_other, 'schema_path': str(args.schema_path), 'created_at': now_iso(), 'script': str(SCRIPT_PATH), 'environment': environment_summary(), 'status': 'written', 'message': 'marker-rule annotation recovered from deleted lab6 workflow evidence'})
        proportion_rows.extend(sample_summary(adata, sample_id, output_path, params))
    pd.DataFrame(manifest_rows).to_csv(manifest_dir / 'annotation_recovery_manifest.tsv', sep='\t', index=False)
    pd.DataFrame(proportion_rows).to_csv(annotation_dir / 'region_cell_proportions.tsv', sep='\t', index=False)
    schema.to_csv(annotation_dir / 'gse269617_region_schema_used.tsv', sep='\t', index=False)
    logging.info('Wrote %d annotated h5ads to %s', len(inputs), output_dir)
if __name__ == '__main__':
    main()
