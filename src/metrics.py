"""Evaluation metrics: AU-PRO and Segmentation F1."""

import cv2
import numpy as np
from sklearn.metrics import f1_score


def calculate_au_pro(
    gt_masks: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """
    Compute the Area Under the Per-Region Overlap (AU-PRO) curve.

    Sweeps thresholds from max to min anomaly score, computes the
    per-region overlap (PRO) and false positive rate (FPR) at each threshold,
    then integrates the PRO-FPR curve up to ``max_fpr`` and normalises.

    Args:
        gt_masks: List of binary ground-truth masks (H x W, uint8).
        anomaly_maps: List of float anomaly score maps (H x W).
        max_fpr: Maximum FPR at which to truncate the integral.
        num_thresholds: Number of linearly-spaced thresholds to evaluate.

    Returns:
        AU-PRO score in [0, 1].
    """
    gt_all = np.concatenate([m.flatten() for m in gt_masks])
    scores_all = np.concatenate([a.flatten() for a in anomaly_maps])
    thresholds = np.linspace(scores_all.max(), scores_all.min(), num_thresholds)

    components_list: list[list] = []
    num_components_total = 0
    for gt in gt_masks:
        _, labels = cv2.connectedComponents(gt.astype(np.uint8))
        num_labels = labels.max()
        comps = [np.where(labels == i) for i in range(1, num_labels + 1)]
        components_list.append(comps)
        num_components_total += len(comps)

    if num_components_total == 0:
        return 1.0

    fprs, pros = [], []
    for th in thresholds:
        pred_all = scores_all >= th
        tn = np.sum(gt_all == 0)
        fpr = np.sum((pred_all == 1) & (gt_all == 0)) / tn if tn > 0 else 0.0

        pro_sum = sum(
            (pred_map >= th)[comp].sum() / len(comp[0])
            for pred_map, comps in zip(anomaly_maps, components_list)
            for comp in comps
        )
        fprs.append(fpr)
        pros.append(pro_sum / num_components_total)

        if fpr >= max_fpr:
            break

    fprs_arr = np.array(fprs)
    pros_arr = np.array(pros)

    if fprs_arr[-1] > max_fpr and len(fprs_arr) > 1:
        pros_arr[-1] = np.interp(max_fpr, [fprs_arr[-2], fprs_arr[-1]], [pros_arr[-2], pros_arr[-1]])
        fprs_arr[-1] = max_fpr

    return float(np.trapezoid(pros_arr, fprs_arr) / max_fpr)


def best_segf1_sweep(
    gt_masks: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
    mu: float,
    sigma: float,
    multipliers: list[int],
) -> tuple[float, int, float]:
    """
    Sweep sigma multipliers and return the best Segmentation F1.

    Args:
        gt_masks: List of binary ground-truth masks.
        anomaly_maps: List of float anomaly score maps upsampled to original resolution.
        mu: Mean of the validation anomaly score distribution.
        sigma: Std-dev of the validation anomaly score distribution.
        multipliers: List of k values to try (threshold = mu + k * sigma).

    Returns:
        Tuple of (best_f1, best_k, best_threshold).
    """
    gt_flat = np.concatenate([m.flatten() for m in gt_masks])
    scores_flat = np.concatenate([a.flatten() for a in anomaly_maps])

    best_f1, best_k, best_thresh = 0.0, multipliers[0], mu + multipliers[0] * sigma
    for k in multipliers:
        thresh = mu + k * sigma
        pred = (scores_flat > thresh).astype(np.uint8)
        f1 = f1_score(gt_flat, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_k, best_thresh = f1, k, thresh

    return best_f1, best_k, best_thresh
