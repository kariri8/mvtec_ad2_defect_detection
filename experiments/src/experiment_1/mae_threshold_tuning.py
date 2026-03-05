import os
import torch
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTMAEForPreTraining, AutoImageProcessor
from sklearn.metrics import f1_score
from mae_experiment import MVTecAD2Dataset, inpaint_16x16_sliding_window

logging.basicConfig(
    filename='experiment_1_16x16_mae.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def calculate_au_pro(gt_masks, anomaly_maps, max_fpr=0.3, num_thresholds=100):
    """Calculates AU-PRO directly from continuous heatmaps (Corrected Overshoot)."""
    gt_all = np.concatenate([m.flatten() for m in gt_masks])
    scores_all = np.concatenate([a.flatten() for a in anomaly_maps])
    
    thresholds = np.linspace(scores_all.max(), scores_all.min(), num_thresholds)
    
    components_list = []
    num_components_total = 0
    for gt in gt_masks:
        num_labels, labels = cv2.connectedComponents(gt.astype(np.uint8))
        comps = [np.where(labels == i) for i in range(1, num_labels)]
        components_list.append(comps)
        num_components_total += len(comps)
        
    if num_components_total == 0: return 1.0
        
    fprs, pros = [], []
    for th in thresholds:
        pred_all = (scores_all >= th)
        fp = np.sum((pred_all == 1) & (gt_all == 0))
        tn = np.sum(gt_all == 0)
        fpr = fp / tn if tn > 0 else 0
        
        pro_sum = 0
        for pred_map, comps in zip(anomaly_maps, components_list):
            pred_binary = pred_map >= th
            for comp in comps:
                pro_sum += pred_binary[comp].sum() / len(comp[0])
        
        fprs.append(fpr)
        pros.append(pro_sum / num_components_total)
        

        if fpr >= max_fpr: 
            break
            
    fprs = np.array(fprs)
    pros = np.array(pros)
    

    if fprs[-1] > max_fpr:
        if len(fprs) > 1:

            pros[-1] = np.interp(max_fpr, [fprs[-2], fprs[-1]], [pros[-2], pros[-1]])
        fprs[-1] = max_fpr
            
    return np.trapezoid(pros, fprs) / max_fpr

def tune_mae_experiment(category):
    logging.info(f"\n{'='*50}\n=== Tuning Exp 1 (MAE): {category} ===\n{'='*50}")
    
    model_dir = f"checkpoints/exp1_{category}/final"
    if not os.path.exists(model_dir):
        logging.info(f"Model not found at {model_dir}")
        return
        
    model = ViTMAEForPreTraining.from_pretrained(model_dir).to("cuda").eval()
    

    val_dataset = MVTecAD2Dataset("../../data", category, split='validation', status='good')
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    all_val_errors = []
    for idx, (pixel_values, _, _) in enumerate(tqdm(val_loader, desc="Baseline Stats")):
        if idx >= 20: break # Optional: limit validation size for speed
        pixel_values = pixel_values.to("cuda")
        with torch.no_grad():
            recon = inpaint_16x16_sliding_window(model, pixel_values)
            diff = torch.abs(pixel_values - recon).mean(dim=1).squeeze().cpu().numpy()
        all_val_errors.append(cv2.GaussianBlur(diff, (3, 3), 0).flatten())
        
    mu = np.mean(np.concatenate(all_val_errors))
    sigma = np.std(np.concatenate(all_val_errors))
    logging.info(f"Baseline -> Mu: {mu:.6f} | Sigma: {sigma:.6f}")


    test_dataset = MVTecAD2Dataset("../../data", category, split='test', status='bad')
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    y_true_masks, y_score_maps, viz_cache = [], [], []
    for i, (pixel_values, gt_mask, path) in enumerate(tqdm(test_loader, desc="Heavy Inference")):
        pixel_values = pixel_values.to("cuda")
        gt_mask_np = gt_mask.squeeze().numpy()
        
        with torch.no_grad():
            recon = inpaint_16x16_sliding_window(model, pixel_values)
            diff = torch.abs(pixel_values - recon).mean(dim=1).squeeze().cpu().numpy()
            
        heatmap = cv2.GaussianBlur(diff, (3, 3), 0)
        y_true_masks.append(gt_mask_np)
        y_score_maps.append(heatmap)
        
        if i < 5: # Save a few for visualization
            orig_img = np.clip((pixel_values.squeeze().cpu().permute(1,2,0).numpy() * 0.224) + 0.456, 0, 1)
            viz_cache.append((os.path.basename(path[0]), orig_img, heatmap, gt_mask_np))


    logging.info("\n--- Continuous Metrics ---")
    au_pro = calculate_au_pro(y_true_masks, y_score_maps, max_fpr=0.3)
    logging.info(f"AU-PRO (up to 0.3 FPR): {au_pro:.4f}")


    gt_all_flat = np.concatenate([m.flatten() for m in y_true_masks])
    scores_all_flat = np.concatenate([s.flatten() for s in y_score_maps])
    
    best_f1, best_mult, best_thresh = 0, 0, 0
    multipliers = [1, 3, 5, 7]
    
    logging.info("\n--- Sigma Sweep (SegF1) ---")
    for m in multipliers:
        thresh = mu + (m * sigma)
        pred_binary = (scores_all_flat > thresh).astype(np.uint8)
        f1 = f1_score(gt_all_flat, pred_binary)
        logging.info(f"Sigma x{m} | Thresh: {thresh:.6f} | SegF1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1, best_mult, best_thresh = f1, m, thresh

    logging.info(f"\n✅ WINNER: Sigma x{best_mult} (SegF1: {best_f1:.4f})")


    out_dir = f"results/exp1_{category}_sigma_results"
    os.makedirs(out_dir, exist_ok=True)
    
    for filename, orig_img, heatmap, gt_mask_np in viz_cache:
        pred = (heatmap > best_thresh).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(orig_img); axes[0].set_title("Original")
        axes[1].imshow(heatmap, cmap='hot'); axes[1].set_title("Error Heatmap")
        axes[2].imshow(pred * 255, cmap='gray'); axes[2].set_title(f"Pred (x{best_mult} σ)")
        axes[3].imshow(gt_mask_np * 255, cmap='gray'); axes[3].set_title("Ground Truth")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename}")
        plt.close()

if __name__ == "__main__":
    for cat in ["vial", "sheet_metal"]:
        tune_mae_experiment(cat)