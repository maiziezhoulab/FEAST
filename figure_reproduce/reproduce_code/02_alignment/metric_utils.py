"""Metric primitives for fixed-plate alignment with partial transport."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

def mean_row_correlation(left, right) -> float:
    """Return the mean finite per-row Pearson correlation without densifying."""
    left = sparse.csr_matrix(left, dtype=np.float64)
    right = sparse.csr_matrix(right, dtype=np.float64)
    if left.shape != right.shape or left.shape[1] < 2:
        raise RuntimeError('expression matrices do not align for correlation')
    n_features = left.shape[1]
    left_sum = np.asarray(left.sum(axis=1)).ravel()
    right_sum = np.asarray(right.sum(axis=1)).ravel()
    cross = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_sq = np.asarray(left.multiply(left).sum(axis=1)).ravel()
    right_sq = np.asarray(right.multiply(right).sum(axis=1)).ravel()
    numerator = cross - left_sum * right_sum / n_features
    denominator = np.sqrt(np.maximum(left_sq - left_sum ** 2 / n_features, 0) * np.maximum(right_sq - right_sum ** 2 / n_features, 0))
    values = np.divide(numerator, denominator, out=np.full(left.shape[0], np.nan), where=denominator > 0)
    if not np.isfinite(values).any():
        raise RuntimeError('no finite per-spot expression correlations')
    return float(np.nanmean(values))

def recovered_rotation(source: np.ndarray, target: np.ndarray) -> float:
    source = source - source.mean(axis=0)
    target = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(source.T @ target)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))

def angular_error(observed: float, expected: float) -> float:
    return float(abs((observed - expected + 180) % 360 - 180))

def ordered_reference_positions(reference_names, moving_names) -> np.ndarray:
    positions = reference_names.get_indexer(moving_names)
    if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
        raise ValueError('moving spots are not an ordered reference subset')
    return positions

def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return float(2 * precision * recall / denominator) if denominator else 0.0

def partial_transport_metrics(matrix: np.ndarray, reference_positions: np.ndarray, *, support_tolerance: float=1e-12, negative_tolerance: float=1e-12) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Score a reference-by-moving partial coupling without forcing matches.

    Returns the metrics, the active-moving mask, and the moving-to-reference
    prediction. Inactive moving columns have prediction ``-1``.
    """
    if support_tolerance < 0 or negative_tolerance < 0:
        raise ValueError('transport tolerances must be non-negative')
    matrix = np.asarray(matrix, dtype=np.float64)
    reference_positions = np.asarray(reference_positions, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != len(reference_positions) or len(np.unique(reference_positions)) != len(reference_positions) or np.any(reference_positions < 0) or np.any(reference_positions >= matrix.shape[0]):
        raise ValueError('transport matrix and retained reference support do not align')
    if matrix.size == 0 or not np.isfinite(matrix).all():
        raise ValueError('transport matrix must be finite and non-empty')
    minimum = float(matrix.min())
    if minimum < -negative_tolerance:
        raise ValueError('transport matrix contains materially negative mass')
    nonnegative = matrix if minimum >= 0 else np.maximum(matrix, 0.0)
    total_mass = float(nonnegative.sum(dtype=np.float64))
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError('transport matrix must have positive mass')
    reference_mass = nonnegative.sum(axis=1)
    moving_mass = nonnegative.sum(axis=0)
    active_reference = reference_mass > support_tolerance
    active_moving = moving_mass > support_tolerance
    moving_to_reference = np.full(matrix.shape[1], -1, dtype=np.int64)
    if active_moving.any():
        moving_to_reference[active_moving] = np.argmax(nonnegative[:, active_moving], axis=0)
    moving_correct = active_moving & (moving_to_reference == reference_positions)
    n_moving = len(reference_positions)
    n_active_moving = int(active_moving.sum())
    n_moving_correct = int(moving_correct.sum())
    reference_ground_truth = np.full(matrix.shape[0], -1, dtype=np.int64)
    reference_ground_truth[reference_positions] = np.arange(n_moving)
    reference_to_moving = np.full(matrix.shape[0], -1, dtype=np.int64)
    if active_reference.any():
        reference_to_moving[active_reference] = np.argmax(nonnegative[active_reference], axis=1)
    true_overlap = reference_ground_truth >= 0
    active_true_overlap = active_reference & true_overlap
    exact_reference_pairs = active_reference & (reference_to_moving == reference_ground_truth)
    n_active_reference = int(active_reference.sum())
    n_active_true_overlap = int(active_true_overlap.sum())
    n_exact_reference_pairs = int(exact_reference_pairs.sum())
    overlap_precision = n_active_true_overlap / n_active_reference if n_active_reference else 0.0
    overlap_recall = n_active_true_overlap / n_moving if n_moving else 0.0
    exact_precision = n_exact_reference_pairs / n_active_reference if n_active_reference else 0.0
    exact_recall = n_exact_reference_pairs / n_moving if n_moving else 0.0
    mutual = 0.0
    if n_active_reference:
        active_indices = np.flatnonzero(active_reference)
        chosen_columns = reference_to_moving[active_indices]
        mutual = float(np.mean(moving_to_reference[chosen_columns] == active_indices))
    ground_truth_mass = float(nonnegative[reference_positions, np.arange(n_moving, dtype=np.int64)].sum(dtype=np.float64))
    metrics = {'transport_support_tolerance': float(support_tolerance), 'transport_total_mass': total_mass, 'transport_minimum': minimum, 'moving_transport_coverage': float(active_moving.mean()), 'transport_moving_exact_spot_recovery_rate': float(n_moving_correct / n_moving if n_moving else 0.0), 'transport_moving_conditional_identity_accuracy': float(n_moving_correct / n_active_moving if n_active_moving else 0.0), 'reference_active_support_fraction': float(active_reference.mean()), 'reference_overlap_detection_precision': overlap_precision, 'reference_overlap_detection_recall': overlap_recall, 'reference_overlap_detection_f1': _f1(overlap_precision, overlap_recall), 'reference_exact_pair_precision': exact_precision, 'reference_exact_pair_recall': exact_recall, 'reference_exact_pair_f1': _f1(exact_precision, exact_recall), 'active_transport_mutual_consistency': mutual, 'ground_truth_transport_mass_fraction': float(ground_truth_mass / total_mass)}
    return (metrics, active_moving, moving_to_reference)

def mapped_label_metrics(reference_labels: np.ndarray, moving_labels: np.ndarray, active_moving: np.ndarray, moving_to_reference: np.ndarray) -> dict[str, float]:
    """Score semantic label transfer while preserving abstentions."""
    reference_labels = np.asarray(reference_labels)
    moving_labels = np.asarray(moving_labels)
    active_moving = np.asarray(active_moving, dtype=bool)
    moving_to_reference = np.asarray(moving_to_reference, dtype=np.int64)
    if moving_labels.shape != active_moving.shape or moving_labels.shape != moving_to_reference.shape or np.any(moving_to_reference[active_moving] < 0) or np.any(moving_to_reference[active_moving] >= len(reference_labels)):
        raise ValueError('label arrays do not align with moving predictions')
    active_count = int(active_moving.sum())
    if not active_count:
        return {'transport_moving_region_recovery_rate': 0.0, 'transport_moving_conditional_region_accuracy': 0.0, 'transport_label_transfer_ari': np.nan}
    predicted = reference_labels[moving_to_reference[active_moving]]
    observed = moving_labels[active_moving]
    correct = int(np.sum(predicted == observed))
    return {'transport_moving_region_recovery_rate': float(correct / len(moving_labels)), 'transport_moving_conditional_region_accuracy': float(correct / active_count), 'transport_label_transfer_ari': float(adjusted_rand_score(observed, predicted))}
