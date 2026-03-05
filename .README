# Unsupervised Anomaly Detection on MVTec AD2

A research project comparing three unsupervised anomaly detection approaches on the [MVTec Anomaly Detection 2 (AD2)](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) dataset, evaluated on the **vial** and **sheet_metal** categories. The best-performing method is then scaled to all 8 dataset categories.

---

## Experiments

All three experiments follow the same unsupervised paradigm: train only on **good** samples, then detect anomalies at test time by measuring how poorly the model reconstructs or predicts a given region. Thresholds are determined from validation set statistics (mean + k·σ).

### Experiment 1 — MAE Pixel-Space Reconstruction (`mae_experiment.py`)

Fine-tunes a pretrained **ViT-MAE** (`facebook/vit-mae-base`) on normal images. At inference, each of the **196** non-overlapping 16×16 patches is masked one at a time and reconstructed. The per-patch reconstruction errors are stitched into a full-resolution anomaly heatmap.

**Key details:**
- Image size: 224×224, patch size: 16×16 (14×14 grid = 196 patches)
- Training: random 15% patch masking, 15 epochs, AdamW lr=1e-4, batch size 8
- Inference: 196 forward passes per image (one masked patch at a time)
- Anomaly score: per-pixel L1 error between original and reconstruction, smoothed with a Gaussian blur

### Experiment 2 — DINO Feature Predictor (`dinov3_experiment.py`)

Extracts frozen **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`) patch features and trains a lightweight **CNN predictor** to reconstruct masked feature tokens from their unmasked neighbors. Anomaly score is the cosine distance between the original and predicted feature at each patch.

**Key details:**
- Backbone: frozen DINOv3 ViT-B/16, embed_dim=768
- Predictor: 3-layer Conv2d network with BatchNorm + ReLU
- Training: random 15% spatial mask, MSE loss on masked tokens only
- Inference: 4-pass grid inference (2×2 offset stride), cosine distance heatmap upsampled to original resolution

### Experiment 3 — Latent MAE / Transformer Predictor (`latent_mae_experiment.py`)

Replaces the CNN predictor from Experiment 2 with a **Transformer encoder** that operates directly on the flattened DINOv3 feature sequence. Masked tokens are replaced with a learnable mask token before being passed through the Transformer, giving the model global context to fill in missing patches.

**Key details:**
- Backbone: same frozen DINOv3 as Exp. 2
- Predictor: 4-layer Transformer Encoder (8 heads, GELU, Pre-LN) + linear projection head
- Learnable positional embedding interpolated to input grid size
- Training: random 50% mask ratio, MSE loss on masked tokens
- Inference: same 4-pass grid inference as Exp. 2, cosine distance heatmap

---

## Results

Metrics reported: **AU-PRO** (Area Under Per-Region Overlap curve, FPR ≤ 0.3) and **Segmentation F1** at the best σ-multiplier threshold found on validation.

### Vial

| Experiment | Model | AU-PRO ↑ | Seg F1 ↑ | Best Threshold |
|---|---|---|---|---|
| Exp 1 | ViT-MAE (pixel) | 0.7108 | 0.2361 | μ + 3σ |
| Exp 2 | DINOv3 + CNN Predictor | **0.8876** | **0.3153** | μ + 3σ |
| Exp 3 | DINOv3 + Transformer Predictor | 0.8875 | 0.3066 | μ + 3σ |

### Sheet Metal

| Experiment | Model | AU-PRO ↑ | Seg F1 ↑ | Best Threshold |
|---|---|---|---|---|
| Exp 1 | ViT-MAE (pixel) | 0.2402 | 0.1403 | μ + 5σ |
| Exp 2 | DINOv3 + CNN Predictor | **0.4746** | 0.1533 | μ + 5σ |
| Exp 3 | DINOv3 + Transformer Predictor | 0.3633 | **0.3962** | μ + 7σ |

### Average Across Both Categories

| Experiment | Avg AU-PRO | Avg Seg F1 |
|---|---|---|
| Exp 1 | 0.4755 | 0.1882 |
| Exp 2 | **0.6811** | 0.2343 |
| Exp 3 | 0.6254 | **0.3514** |

> **Why Exp 3 was chosen for the final run:** Although Exp 2 edges out Exp 3 on AU-PRO (0.681 vs 0.625), Experiment 3 achieves a substantially higher average Segmentation F1 (0.351 vs 0.234), demonstrating better ability to localise anomalies at the pixel level. The Transformer predictor's global attention allows it to reason about spatial context more effectively than the local CNN, particularly on the structurally complex sheet_metal category where it improves F1 from 0.153 → 0.396.

---

## Qualitative Results

Sample visualisations from Experiment 3 (columns: original image | latent cosine distance heatmap | binary prediction | ground truth mask).

**Sheet Metal:**

![Sheet Metal Result](experiments/src/experiment_3/results/exp3_sheet_metal_sigma_results/000_regular.png)

**Vial:**

![Vial Result](experiments/src/experiment_3/results/exp3_vial_sigma_results/000_regular.png)

---

## Final Run — All Categories (`final_train_all.py`)

Scales Experiment 3 to all 8 MVTec AD2 categories using **parallel training across 2 GPUs**.

**Categories:** `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`

**Additional features over Exp 3:**
- **Early stopping** with patience=3 (max 50 epochs)
- **Resumable training** — full optimizer + epoch state checkpointed each epoch; recovers automatically from interruptions
- **Parallel execution** — 2 concurrent workers, each assigned to a specific GPU (`cuda:0` / `cuda:1`)
- **Sigma sweep** — searches μ + {1,3,5,...,15}σ to find the best F1 threshold per category
- Results written to `final_summary_results.txt`

Results are currently running. ⏳

---

## Setup

```bash
pip install -r requirements.txt
```

**Hardware:** 2× CUDA GPUs recommended for the final parallel run. Experiments 1–3 require a single GPU.

---

## Evaluation Metrics

- **AU-PRO (≤ 0.3 FPR):** Measures how well anomaly scores overlap with individual defect regions (connected components), integrated up to a false positive rate of 0.3. More robust to region size imbalance than pixel-level AUROC.
- **Segmentation F1:** Binary F1 score at pixel level using the best threshold found by sweeping μ + k·σ on validation statistics.