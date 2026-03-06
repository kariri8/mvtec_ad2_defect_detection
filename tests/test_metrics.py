"""Unit tests for src/metrics.py."""

import numpy as np
import pytest

from src.metrics import best_segf1_sweep, calculate_au_pro


# ── calculate_au_pro ──────────────────────────────────────────────────────────

def _make_circle_mask(size: int = 32, r: int = 6) -> np.ndarray:
    """Return a binary mask with a circular anomaly region."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                mask[y, x] = 1
    return mask


def test_au_pro_perfect_detection():
    """AU-PRO should equal 1 when the anomaly map perfectly matches the GT."""
    gt = _make_circle_mask()
    # Score = 1 inside anomaly, 0 outside → perfect
    score = gt.astype(float)
    result = calculate_au_pro([gt], [score])
    assert result == pytest.approx(1.0, abs=0.05)


def test_au_pro_no_anomalies():
    """AU-PRO should return 1.0 when there are no connected components."""
    gt = np.zeros((32, 32), dtype=np.uint8)
    score = np.random.rand(32, 32)
    result = calculate_au_pro([gt], [score])
    assert result == 1.0


def test_au_pro_zero_detection():
    """AU-PRO should be near 0 when scores are inverted (worst case)."""
    gt = _make_circle_mask()
    # Score = 0 inside anomaly, 1 outside → worst possible
    score = (1.0 - gt).astype(float)
    result = calculate_au_pro([gt], [score])
    assert result < 0.1


def test_au_pro_range():
    """AU-PRO should always be in [0, 1]."""
    gt = _make_circle_mask()
    score = np.random.rand(*gt.shape)
    result = calculate_au_pro([gt], [score])
    assert 0.0 <= result <= 1.0


def test_au_pro_multiple_images():
    """AU-PRO should handle a list of multiple images."""
    gts = [_make_circle_mask()] * 3
    scores = [np.random.rand(32, 32)] * 3
    result = calculate_au_pro(gts, scores)
    assert 0.0 <= result <= 1.0


# ── best_segf1_sweep ──────────────────────────────────────────────────────────

def test_segf1_sweep_finds_best_k():
    """The sweep should return a non-zero F1 for a clearly detectable anomaly."""
    gt = _make_circle_mask()
    # Score = 1 inside anomaly, 0 outside, small noise
    score = gt.astype(float) + np.random.rand(*gt.shape) * 0.05

    mu = float(score[gt == 0].mean())
    sigma = float(score[gt == 0].std()) + 1e-6

    best_f1, best_k, best_thresh = best_segf1_sweep(
        [gt], [score], mu, sigma, multipliers=[1, 3, 5, 7]
    )
    assert best_f1 > 0.5
    assert best_k in [1, 3, 5, 7]
    assert best_thresh > mu


def test_segf1_sweep_all_zero_gt():
    """F1 should be 0.0 when ground truth has no positives."""
    gt = np.zeros((32, 32), dtype=np.uint8)
    score = np.random.rand(32, 32)
    mu, sigma = float(score.mean()), float(score.std())

    best_f1, _, _ = best_segf1_sweep([gt], [score], mu, sigma, multipliers=[1, 3])
    assert best_f1 == pytest.approx(0.0, abs=1e-6)


def test_segf1_sweep_returns_best_of_all_k():
    """Returned F1 should be the maximum across all multipliers."""
    gt = _make_circle_mask()
    score = gt.astype(float) * 0.8 + np.random.rand(*gt.shape) * 0.1
    mu, sigma = 0.05, 0.02

    best_f1, best_k, _ = best_segf1_sweep([gt], [score], mu, sigma, multipliers=[1, 3, 5, 7])

    # Manually compute F1 for each k and verify returned is the max
    flat_gt = gt.flatten()
    flat_score = score.flatten()
    from sklearn.metrics import f1_score
    f1s = {k: f1_score(flat_gt, (flat_score > mu + k * sigma).astype(int), zero_division=0)
           for k in [1, 3, 5, 7]}
    assert best_f1 == pytest.approx(max(f1s.values()), abs=1e-6)
    assert best_k == max(f1s, key=f1s.get)
