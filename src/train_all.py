"""Full-scale training: Experiment 3 across all 8 MVTec AD2 categories.

Runs categories in parallel across 2 GPUs with early stopping,
resumable checkpointing, and an extended sigma sweep.
"""

import concurrent.futures
import logging
import multiprocessing
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.dataset import MVTecAD2Dataset
from src.experiment2 import extract_dino_features
from src.experiment3 import LatentTransformerPredictor
from src.metrics import best_segf1_sweep, calculate_au_pro

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
RESULTS_FILE = "results/final_summary_results.txt"


def _get_logger(category: str, log_dir: str) -> logging.Logger:
    cat_logger = logging.getLogger(category)
    if not cat_logger.handlers:
        cat_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, f"training_{category}.log"), mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        cat_logger.addHandler(fh)
    return cat_logger


def process_category(category: str, gpu_id: int, worker_idx: int, cfg: dict) -> dict:
    """
    Train and evaluate Experiment 3 for one category on a specific GPU.

    Supports resumable training via a checkpoint saved after each epoch.
    Early stopping halts training when loss does not improve for ``patience``
    consecutive epochs.

    Args:
        category: MVTec AD2 category name.
        gpu_id: CUDA device index to use.
        worker_idx: Worker position used for tqdm progress bar placement.
        cfg: Config dictionary.

    Returns:
        Dict with keys 'category', 'au_pro', 'seg_f1', 'best_k'.
    """
    device = f"cuda:{gpu_id}"
    ckpt_dir = os.path.join(cfg["checkpoint_dir"], category)
    out_dir = os.path.join(cfg["results_dir"], f"{category}_sigma_results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log = _get_logger(category, ckpt_dir)
    log.info("=== Starting %s on GPU %d ===", category, gpu_id)

    best_model_path = os.path.join(ckpt_dir, "best_latent_mae.pth")
    checkpoint_path = os.path.join(ckpt_dir, "training_state.pth")
    complete_flag = os.path.join(ckpt_dir, "training_complete.flag")

    backbone = AutoModel.from_pretrained(cfg["backbone"]).to(device).eval()
    predictor = LatentTransformerPredictor().to(device)
    optimizer = AdamW(predictor.parameters(), lr=cfg["lr"])

    # ── Training ──────────────────────────────────────────────────────────────
    if os.path.exists(complete_flag):
        log.info("Training already complete — loading best model.")
        predictor.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        train_dataset = MVTecAD2Dataset(cfg["data_root"], category, split="train")
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        start_epoch, best_loss, patience_counter = 0, float("inf"), 0

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            predictor.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            patience_counter = ckpt["patience_counter"]
            log.info("Resuming from epoch %d (best loss %.4f)", start_epoch, best_loss)

        predictor.train()
        for epoch in range(start_epoch, cfg["max_epochs"]):
            total_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f"[{category}] Ep {epoch + 1}/{cfg['max_epochs']}",
                position=worker_idx,
                leave=False,
            )
            for pixel_values, _, _ in pbar:
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
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            log.info("Epoch %d loss: %.4f", epoch + 1, avg_loss)

            # Save resumable checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": predictor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "patience_counter": patience_counter,
                },
                checkpoint_path,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(predictor.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= cfg["patience"]:
                log.info("Early stopping at epoch %d.", epoch + 1)
                break

        open(complete_flag, "w").close()
        predictor.load_state_dict(torch.load(best_model_path, map_location=device))

    # ── Validation stats ──────────────────────────────────────────────────────
    predictor.eval()
    val_dataset = MVTecAD2Dataset(cfg["data_root"], category, split="validation")
    val_loader = DataLoader(val_dataset, batch_size=1)

    errors: list[np.ndarray] = []
    for pixel_values, _, _ in tqdm(val_loader, desc=f"[{category}] Val stats", position=worker_idx, leave=False):
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            feat = extract_dino_features(backbone, pixel_values)
            pred = _infer(feat, predictor, device)
            dist = 1.0 - nn.functional.cosine_similarity(feat, pred, dim=1)
        errors.append(dist.squeeze().cpu().numpy().flatten())

    mu = float(np.mean(np.concatenate(errors)))
    sigma = float(np.std(np.concatenate(errors)))
    log.info("Val stats → mu=%.6f  sigma=%.6f", mu, sigma)

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_dataset = MVTecAD2Dataset(cfg["data_root"], category, split="test", status="bad")
    test_loader = DataLoader(test_dataset, batch_size=1)

    gt_masks, score_maps = [], []
    for pixel_values, gt_mask, _ in tqdm(test_loader, desc=f"[{category}] Eval", position=worker_idx, leave=False):
        pixel_values = pixel_values.to(device)
        gt_np = gt_mask.squeeze().numpy()
        with torch.no_grad():
            feat = extract_dino_features(backbone, pixel_values)
            pred = _infer(feat, predictor, device)
            dist = 1.0 - nn.functional.cosine_similarity(feat, pred, dim=1)
        heatmap = cv2.resize(
            dist.squeeze().cpu().numpy(),
            (gt_np.shape[1], gt_np.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        gt_masks.append(gt_np)
        score_maps.append(heatmap)

    au_pro = calculate_au_pro(gt_masks, score_maps)
    best_f1, best_k, _ = best_segf1_sweep(gt_masks, score_maps, mu, sigma, cfg["sigma_multipliers"])

    log.info("AU-PRO: %.4f | SegF1: %.4f (k=%d)", au_pro, best_f1, best_k)
    return {"category": category, "au_pro": au_pro, "seg_f1": best_f1, "best_k": best_k}


def _infer(feature_map: torch.Tensor, predictor: LatentTransformerPredictor, device: str) -> torch.Tensor:
    """4-pass grid inference for the Transformer predictor."""
    _, C, H, W = feature_map.shape
    combined = torch.zeros_like(feature_map)
    counts = torch.zeros((1, 1, H, W), device=device)
    with torch.no_grad():
        for ro in range(2):
            for co in range(2):
                mask = torch.ones((1, 1, H, W), device=device)
                mask[:, :, ro::2, co::2] = 0.0
                preds = predictor(feature_map, mask_map=mask)
                inv = 1.0 - mask
                combined += preds * inv
                counts += inv
    return combined / counts.clamp(min=1)


def main(cfg: dict) -> None:
    """
    Launch parallel training across all categories.

    Args:
        cfg: Config dictionary loaded from configs/final_run.yaml.
    """
    os.makedirs("results", exist_ok=True)
    multiprocessing.set_start_method("spawn", force=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["num_workers"]) as executor:
        futures = {
            executor.submit(process_category, cat, i % cfg["num_gpus"], i, cfg): cat
            for i, cat in enumerate(CATEGORIES)
        }
        for future in concurrent.futures.as_completed(futures):
            cat = futures[future]
            try:
                res = future.result()
                results.append(res)
                tqdm.write(f"✅ {cat} → AU-PRO: {res['au_pro']:.4f} | SegF1: {res['seg_f1']:.4f} (k={res['best_k']})")
            except Exception as exc:
                tqdm.write(f"❌ ERROR {cat}: {exc}")

    if results:
        avg_au_pro = np.mean([r["au_pro"] for r in results])
        avg_f1 = np.mean([r["seg_f1"] for r in results])
        with open(RESULTS_FILE, "w") as f:
            f.write("=" * 52 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("=" * 52 + "\n")
            f.write(f"{'Category':<15} | {'AU-PRO':<10} | {'k':<6} | {'SegF1'}\n")
            f.write("-" * 52 + "\n")
            for r in sorted(results, key=lambda x: x["category"]):
                f.write(f"{r['category']:<15} | {r['au_pro']:<10.4f} | x{r['best_k']:<5} | {r['seg_f1']:.4f}\n")
            f.write("-" * 52 + "\n")
            f.write(f"{'AVERAGE':<15} | {avg_au_pro:<10.4f} | {'—':<6} | {avg_f1:.4f}\n")
        print(f"\nResults saved to {RESULTS_FILE}")
