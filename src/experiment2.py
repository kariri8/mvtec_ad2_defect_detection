"""Experiment 2: Frozen DINOv3 + lightweight CNN feature predictor."""

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
from src.metrics import best_segf1_sweep, calculate_au_pro

logger = logging.getLogger(__name__)

EMBED_DIM = 768


class CNNFeaturePredictor(nn.Module):
    """
    Lightweight CNN predictor for masked DINOv3 feature tokens.

    Two 3×3 Conv2d layers with BatchNorm + ReLU, followed by a 1×1 projection.

    Args:
        embed_dim: Feature dimension of the backbone (default 768).
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_dino_features(
    backbone: nn.Module,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """
    Extract spatial patch features from a frozen DINOv3 backbone.

    Args:
        backbone: Frozen DINOv3 model.
        pixel_values: Input tensor (1, C, H, W).

    Returns:
        Feature map of shape (1, 768, H//16, W//16).
    """
    with torch.no_grad():
        outputs = backbone(pixel_values)
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        grid_h, grid_w = h // PATCH_SIZE, w // PATCH_SIZE
        num_patches = grid_h * grid_w
        tokens = outputs.last_hidden_state[:, -num_patches:, :]
        return tokens.permute(0, 2, 1).reshape(1, EMBED_DIM, grid_h, grid_w)


def grid_inference(
    feature_map: torch.Tensor,
    predictor: nn.Module,
    device: str,
    stride: int = 2,
) -> torch.Tensor:
    """
    4-pass grid inference: mask every other patch at 4 (row, col) offsets,
    predict each set from context, and average the results.

    Args:
        feature_map: Original feature map (1, C, H, W).
        predictor: Trained CNN or Transformer predictor.
        device: Torch device string.
        stride: Grid stride (default 2 → 4 passes).

    Returns:
        Averaged predicted feature map, same shape as ``feature_map``.
    """
    _, C, H, W = feature_map.shape
    combined = torch.zeros_like(feature_map)
    counts = torch.zeros((1, 1, H, W), device=device)

    predictor.eval()
    with torch.no_grad():
        for ro in range(stride):
            for co in range(stride):
                mask = torch.ones((1, 1, H, W), device=device)
                mask[:, :, ro::stride, co::stride] = 0.0
                preds = predictor(feature_map * mask)
                inv = 1.0 - mask
                combined += preds * inv
                counts += inv

    return combined / counts.clamp(min=1)


def train(category: str, cfg: dict) -> tuple[nn.Module, nn.Module]:
    """
    Train the CNN feature predictor for one category.

    Args:
        category: MVTec AD2 category name.
        cfg: Config dictionary.

    Returns:
        Tuple (backbone, predictor) both ready for inference.
    """
    device = cfg["device"]
    ckpt_dir = os.path.join(cfg["checkpoint_dir"], f"exp2_{category}")
    final_path = os.path.join(ckpt_dir, "final_predictor.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    backbone = AutoModel.from_pretrained(cfg["backbone"]).to(device).eval()
    predictor = CNNFeaturePredictor().to(device)

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
            preds = predictor(feat * mask)
            loss = (torch.nn.functional.mse_loss(preds * inv, feat * inv, reduction="sum")
                    / inv.sum().clamp(min=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                mid = os.path.join(ckpt_dir, f"predictor_epoch_{epoch + 1}.pth")
                torch.save(predictor.state_dict(), mid)

        logger.info("Epoch %d avg loss: %.4f", epoch + 1, total_loss / len(loader))

    torch.save(predictor.state_dict(), final_path)
    logger.info("Predictor saved to %s", final_path)
    return backbone, predictor


def get_val_stats(
    backbone: nn.Module,
    predictor: nn.Module,
    category: str,
    cfg: dict,
) -> tuple[float, float]:
    """
    Compute mean and std of cosine distance on the validation set.

    Args:
        backbone: Frozen DINOv3 backbone.
        predictor: Trained CNN predictor.
        category: MVTec AD2 category name.
        cfg: Config dictionary.

    Returns:
        Tuple (mu, sigma).
    """
    device = cfg["device"]
    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="validation")
    loader = DataLoader(dataset, batch_size=1)
    predictor.eval()

    errors: list[np.ndarray] = []
    for pixel_values, _, _ in tqdm(loader, desc="Validation stats"):
        pixel_values = pixel_values.to(device)
        feat = extract_dino_features(backbone, pixel_values)
        pred = grid_inference(feat, predictor, device)
        dist = (1.0 - torch.nn.functional.cosine_similarity(feat, pred, dim=1))
        errors.append(dist.squeeze().cpu().numpy().flatten())

    all_errors = np.concatenate(errors)
    return float(np.mean(all_errors)), float(np.std(all_errors))


def evaluate(
    backbone: nn.Module,
    predictor: nn.Module,
    category: str,
    cfg: dict,
) -> dict:
    """
    Evaluate on the test set and return metrics.

    Args:
        backbone: Frozen DINOv3 backbone.
        predictor: Trained CNN predictor.
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

    out_dir = os.path.join(cfg["results_dir"], f"exp2_{category}")
    os.makedirs(out_dir, exist_ok=True)

    gt_masks, score_maps = [], []
    for pixel_values, gt_mask, _ in tqdm(loader, desc="Evaluating"):
        pixel_values = pixel_values.to(device)
        feat = extract_dino_features(backbone, pixel_values)
        pred = grid_inference(feat, predictor, device)
        dist = (1.0 - torch.nn.functional.cosine_similarity(feat, pred, dim=1))
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
