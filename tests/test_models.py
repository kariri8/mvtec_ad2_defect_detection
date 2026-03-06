"""Unit tests for model architecture (Transformer predictor)."""

import torch
import pytest

from src.experiment3 import LatentTransformerPredictor


@pytest.fixture
def predictor():
    return LatentTransformerPredictor(embed_dim=64, num_layers=2, num_heads=4)


def test_output_shape_matches_input(predictor):
    """Predictor output should have the same shape as the input feature map."""
    x = torch.randn(1, 64, 8, 8)
    out = predictor(x)
    assert out.shape == x.shape


def test_output_shape_with_mask(predictor):
    """Output shape should be unchanged when a mask is provided."""
    x = torch.randn(1, 64, 8, 8)
    mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    out = predictor(x, mask_map=mask)
    assert out.shape == x.shape


def test_non_square_feature_map(predictor):
    """Predictor should handle non-square feature maps via pos-embed interpolation."""
    x = torch.randn(1, 64, 6, 10)
    out = predictor(x)
    assert out.shape == (1, 64, 6, 10)


def test_mask_token_applied(predictor):
    """Masked positions should produce different outputs than unmasked ones."""
    torch.manual_seed(0)
    x = torch.randn(1, 64, 4, 4)

    mask_all_visible = torch.ones(1, 1, 4, 4)
    mask_all_masked = torch.zeros(1, 1, 4, 4)

    out_visible = predictor(x, mask_map=mask_all_visible)
    out_masked = predictor(x, mask_map=mask_all_masked)

    assert not torch.allclose(out_visible, out_masked)


def test_predictor_parameters_exist(predictor):
    """Mask token and positional embedding should be learnable parameters."""
    param_names = [n for n, _ in predictor.named_parameters()]
    assert any("mask_token" in n for n in param_names)
    assert any("pos_embed" in n for n in param_names)
