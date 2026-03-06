"""Experiment 1: ViT-MAE pixel-space reconstruction for anomaly detection."""

import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTMAEForPreTraining

from src.dataset import MVTecAD2Dataset
from src.metrics import best_segf1_sweep, calculate_au_pro

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196


def _get_single_patch_noise(batch_size: int, patch_idx: int, device: str) -> torch.Tensor:
    """
    Build a noise vector that forces exactly one patch to be masked.

    Args:
        batch_size: Number of samples in the batch.
        patch_idx: Index of the patch to mask (0-based).
        device: Torch device string.

    Returns:
        Noise tensor of shape (batch_size, NUM_PATCHES).
    """
    noise = torch.zeros((batch_size, NUM_PATCHES), device=device)
    noise[:, patch_idx] = 1.0
    return noise


def inpaint_sliding_window(
    model: ViTMAEForPreTraining,
    pixel_values: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Reconstruct every 16×16 patch one at a time (196 passes) and stitch results.

    Args:
        model: Fine-tuned ViT-MAE model.
        pixel_values: Input image tensor (B, C, H, W).
        device: Torch device string.

    Returns:
        Stitched reconstruction tensor of the same shape as ``pixel_values``.
    """
    stitched = torch.zeros_like(pixel_values, device=device)
    grid = IMAGE_SIZE // PATCH_SIZE

    for idx in range(NUM_PATCHES):
        noise = _get_single_patch_noise(pixel_values.shape[0], idx, device)
        with torch.no_grad():
            outputs = model(pixel_values, noise=noise)
            recon = model.unpatchify(outputs.logits)
        row, col = idx // grid, idx % grid
        y1, y2 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
        x1, x2 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
        stitched[:, :, y1:y2, x1:x2] = recon[:, :, y1:y2, x1:x2]

    return stitched


def train(category: str, cfg: dict) -> ViTMAEForPreTraining:
    """
    Fine-tune ViT-MAE on normal images for one category.

    Args:
        category: MVTec AD2 category name.
        cfg: Config dictionary (see configs/experiment1.yaml).

    Returns:
        Trained model loaded on the configured device.
    """
    device = cfg["device"]
    ckpt_dir = os.path.join(cfg["checkpoint_dir"], f"exp1_{category}")
    final_dir = os.path.join(ckpt_dir, "final")
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(final_dir):
        logger.info("Found existing model at %s — skipping training.", final_dir)
        return ViTMAEForPreTraining.from_pretrained(final_dir).to(device)

    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="train", resize=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])

    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for pixel_values, _, _ in tqdm(loader, desc=f"[{category}] Epoch {epoch + 1}/{cfg['epochs']}"):
            pixel_values = pixel_values.to(device)
            noise = torch.rand((pixel_values.shape[0], NUM_PATCHES), device=device)
            loss = model(pixel_values, noise=noise).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("Epoch %d avg loss: %.6f", epoch + 1, total_loss / len(loader))

        if (epoch + 1) % 5 == 0:
            mid_dir = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}")
            model.save_pretrained(mid_dir)

    model.save_pretrained(final_dir)
    logger.info("Model saved to %s", final_dir)
    return model


def get_val_stats(
    model: ViTMAEForPreTraining,
    category: str,
    cfg: dict,
) -> tuple[float, float]:
    """
    Compute mean and std of per-pixel reconstruction error on validation set.

    Args:
        model: Trained ViT-MAE model.
        category: MVTec AD2 category name.
        cfg: Config dictionary.

    Returns:
        Tuple (mu, sigma) of validation error statistics.
    """
    device = cfg["device"]
    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="validation", resize=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=1)
    model.eval()

    errors: list[np.ndarray] = []
    for idx, (pixel_values, _, _) in enumerate(tqdm(loader, desc="Validation stats")):
        if idx >= 20:
            break
        pixel_values = pixel_values.to(device)
        recon = inpaint_sliding_window(model, pixel_values, device)
        diff = torch.abs(pixel_values - recon).mean(dim=1).squeeze().cpu().numpy()
        errors.append(cv2.GaussianBlur(diff, (3, 3), 0).flatten())

    all_errors = np.concatenate(errors)
    return float(np.mean(all_errors)), float(np.std(all_errors))


def evaluate(
    model: ViTMAEForPreTraining,
    category: str,
    cfg: dict,
) -> dict:
    """
    Evaluate the model on the test set and save visualisations.

    Args:
        model: Trained ViT-MAE model.
        category: MVTec AD2 category name.
        cfg: Config dictionary.

    Returns:
        Dict with keys 'au_pro', 'seg_f1', 'best_k'.
    """
    device = cfg["device"]
    mu, sigma = get_val_stats(model, category, cfg)

    dataset = MVTecAD2Dataset(cfg["data_root"], category, split="test", status="bad", resize=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=1)
    model.eval()

    out_dir = os.path.join(cfg["results_dir"], f"exp1_{category}")
    os.makedirs(out_dir, exist_ok=True)

    gt_masks, score_maps = [], []
    for pixel_values, gt_mask, _ in tqdm(loader, desc="Evaluating"):
        pixel_values = pixel_values.to(device)
        recon = inpaint_sliding_window(model, pixel_values, device)
        diff = torch.abs(pixel_values - recon).mean(dim=1).squeeze().cpu().numpy()
        heatmap = cv2.GaussianBlur(diff, (3, 3), 0)
        gt_masks.append(gt_mask.squeeze().numpy())
        score_maps.append(heatmap)

    au_pro = calculate_au_pro(gt_masks, score_maps)
    best_f1, best_k, _ = best_segf1_sweep(gt_masks, score_maps, mu, sigma, cfg["sigma_multipliers"])

    logger.info("[%s] AU-PRO: %.4f | SegF1: %.4f (k=%d)", category, au_pro, best_f1, best_k)
    return {"au_pro": au_pro, "seg_f1": best_f1, "best_k": best_k}
