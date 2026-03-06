"""Experiment 3: Frozen DINOv3 + Transformer encoder (Latent MAE)."""

import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.dataset import MVTecAD2Dataset, PATCH_SIZE
from src.experiment2 import extract_dino_features, grid_inference
from src.metrics import best_segf1_sweep, calculate_au_pro

logger = logging.getLogger(__name__)

EMBED_DIM = 768


class LatentTransformerPredictor(nn.Module):
    """
    Transformer encoder that predicts masked DINOv3 feature tokens.

    Masked positions are replaced with a learnable mask token.
    A learnable 2D positional embedding (base 64×64) is bicubically
    interpolated to the current feature grid size.

    Args:
        embed_dim: Feature dimension (default 768).
        num_layers: Number of Transformer encoder layers (default 4).
        num_heads: Number of attention heads (default 8).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 64, 64) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Feature map (B, C, H, W).
            mask_map: Binary mask (B, 1, H, W); 1 = visible, 0 = masked.

        Returns:
            Predicted feature map (B, C, H, W).
        """
        B, C, H, W = x.shape
        seq_len = H * W
        x_seq = x.view(B, C, seq_len).permute(0, 2, 1)

        if mask_map is not None:
            mask_seq = mask_map.view(B, seq_len, 1)
            x_seq = x_seq * mask_seq + self.mask_token * (1.0 - mask_seq)

        pos = torch.nn.functional.interpolate(
            self.pos_embed, size=(H, W), mode="bicubic", align_corners=False
        )
        x_seq = x_seq + pos.view(1, C, seq_len).permute(0, 2, 1)

        out = self.head(self.transformer(x_seq))
        return out.permute(0, 2, 1).view(B, C, H, W)


def _grid_inference_transformer(
    feature_map: torch.Tensor,
    predictor: "LatentTransformerPredictor",
    device: str,
    stride: int = 2,
) -> torch.Tensor:
    """4-pass grid inference adapted for the Transformer predictor."""
    _, C, H, W = feature_map.shape
    combined = torch.zeros_like(feature_map)
    counts = torch.zeros((1, 1, H, W), device=device)

    predictor.eval()
    with torch.no_grad():
        for ro in range(stride):
            for co in range(stride):
                mask = torch.ones((1, 1, H, W), device=device)
                mask[:, :, ro::stride, co::stride] = 0.0
                preds = predictor(feature_map, mask_map=mask)
                inv = 1.0 - mask
                combined += preds * inv
                counts += inv

    return combined / counts.clamp(min=1)


def train(category: str, cfg: dict) -> tuple[nn.Module, "LatentTransformerPredictor"]:
    """
    Train the Transformer predictor for one category.

    Args:
        category: MVTec AD2 category name.
        cfg: Config dictionary.

    Returns:
        Tuple (backbone, predictor) ready for inference.
    """
    device = cfg["device"]
    ckpt_dir = os.path.join(cfg["checkpoint_dir"], f"exp3_{category}")
    final_path = os.path.join(ckpt_dir, "final_latent_mae.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    backbone = AutoModel.from_pretrained(cfg["backbone"]).to(device).eval()
    predictor = LatentTransformerPredictor().to(device)

    if os.path.exists(final_path):
        logger.info("Found existing predictor at %s — skipping training.", final_path)
        predictor.load_state_dict(torch.load(final_path, map_location=device))
        return backbone, predictor

    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = AdamW(predictor.parameters(), lr=cfg["lr"])

    predictor.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for pixel_values, _, _ in tqdm(loader, desc=f"[{category}] Epoch {epoch + 1}/{cfg['epochs']}"):
            pixel_values = pixel_values.to(device)
            feat = extract_dino_features(backbone, pixel_values)
            B, C, H, W = feat.shape
            mask = (torch.rand((B, 1, H, W), device=device) > cfg["mask_ratio"]).float()
            inv = 1.0 - mask
            preds = predictor(feat, mask_map=mask)
            loss = (nn.functional.mse_loss(preds * inv, feat * inv, reduction="sum")
                    / inv.sum().clamp(min=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info("Epoch %d avg loss: %.4f", epoch + 1, total_loss / len(loader))

    torch.save(predictor.state_dict(), final_path)
    logger.info("Predictor saved to %s", final_path)
    return backbone, predictor


def get_val_stats(
    backbone: nn.Module,
    predictor: "LatentTransformerPredictor",
    category: str,
    cfg: dict,
) -> tuple[float, float]:
    """Compute validation cosine-distance statistics."""
    device = cfg["device"]
    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="validation")
    loader = DataLoader(dataset, batch_size=1)
    predictor.eval()

    errors: list[np.ndarray] = []
    for pixel_values, _, _ in tqdm(loader, desc="Validation stats"):
        pixel_values = pixel_values.to(device)
        feat = extract_dino_features(backbone, pixel_values)
        pred = _grid_inference_transformer(feat, predictor, device)
        dist = 1.0 - nn.functional.cosine_similarity(feat, pred, dim=1)
        errors.append(dist.squeeze().cpu().numpy().flatten())

    all_errors = np.concatenate(errors)
    return float(np.mean(all_errors)), float(np.std(all_errors))


def evaluate(
    backbone: nn.Module,
    predictor: "LatentTransformerPredictor",
    category: str,
    cfg: dict,
) -> dict:
    """
    Evaluate on the test set.

    Args:
        backbone: Frozen DINOv3 backbone.
        predictor: Trained Transformer predictor.
        category: MVTec AD2 category name.
        cfg: Config dictionary.

    Returns:
        Dict with keys 'au_pro', 'seg_f1', 'best_k'.
    """
    device = cfg["device"]
    mu, sigma = get_val_stats(backbone, predictor, category, cfg)
    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="test", status="bad")
    loader = DataLoader(dataset, batch_size=1)
    predictor.eval()

    out_dir = os.path.join(cfg["results_dir"], f"exp3_{category}")
    os.makedirs(out_dir, exist_ok=True)

    gt_masks, score_maps = [], []
    for pixel_values, gt_mask, _ in tqdm(loader, desc="Evaluating"):
        pixel_values = pixel_values.to(device)
        feat = extract_dino_features(backbone, pixel_values)
        pred = _grid_inference_transformer(feat, predictor, device)
        dist = (1.0 - nn.functional.cosine_similarity(feat, pred, dim=1))
        heatmap = cv2.resize(
            dist.squeeze().cpu().numpy(),
            (gt_mask.shape[-1], gt_mask.shape[-2]),
            interpolation=cv2.INTER_CUBIC,
        )
        gt_masks.append(gt_mask.squeeze().numpy())
        score_maps.append(heatmap)

    au_pro = calculate_au_pro(gt_masks, score_maps)
    best_f1, best_k, _ = best_segf1_sweep(gt_masks, score_maps, mu, sigma, cfg["sigma_multipliers"])

    logger.info("[%s] AU-PRO: %.4f | SegF1: %.4f (k=%d)", category, au_pro, best_f1, best_k)
    return {"au_pro": au_pro, "seg_f1": best_f1, "best_k": best_k}
