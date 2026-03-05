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
from torch.optim import AdamW
from sklearn.metrics import f1_score

logging.basicConfig(
    filename='experiment_1_16x16_mae.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DEVICE = "cuda"
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = 224 # 224 / 16 = 14x14 grid (196 patches total)
PATCH_SIZE = 16 
CATEGORIES = ["vial", "sheet_metal"]
DATASET_ROOT = "../../data" 

class MVTecAD2Dataset(Dataset):
    def __init__(self, root_dir, category, split='train', status='good'):
        """
        split: 'train', 'validation', 'test'
        status: 'good', 'bad'
        """
        self.status = status
        self.split = split
        

        if split == 'train':
            self.image_dir = os.path.join(root_dir, category, 'train', 'good')
            self.gt_dir = None
        elif split == 'validation':
            self.image_dir = os.path.join(root_dir, category, 'validation', 'good')
            self.gt_dir = None
        elif split == 'test':
            self.image_dir = os.path.join(root_dir, category, 'test_public', status)
            if status == 'bad':
                self.gt_dir = os.path.join(root_dir, category, 'test_public', 'ground_truth', 'bad')
            else:
                self.gt_dir = None
        else:
            raise ValueError(f"Unknown split: {split}")
            

        self.image_paths = sorted([
            os.path.join(self.image_dir, f) 
            for f in os.listdir(self.image_dir) 
            if f.endswith('.png')
        ])
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        

        if self.split == 'test' and self.status == 'bad' and self.gt_dir:

            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            

            mask_filename = f"{base_name}_mask{ext}"
            gt_path = os.path.join(self.gt_dir, mask_filename)
            
            if os.path.exists(gt_path):

                gt_mask = Image.open(gt_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
                gt_mask = (np.array(gt_mask) > 127).astype(np.uint8) # Convert to strictly 0 or 1
            else:

                logging.warning(f"GT mask not found for {img_path}. Looked for {gt_path}")
                gt_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        else:

            gt_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            
        return pixel_values, gt_mask, img_path

def get_random_patch_noise(batch_size, num_patches=196, mask_ratio=0.15):
    """Randomly masks a percentage of 16x16 patches during training."""
    noise = torch.rand((batch_size, num_patches)).to(DEVICE)
    return noise # MAE automatically masks the patches with highest noise based on its internal ratio

def get_single_patch_noise(batch_size, patch_idx, num_patches=196):
    """Masks EXACTLY ONE 16x16 patch (token) for surgical evaluation."""
    noise = torch.zeros((batch_size, num_patches)).to(DEVICE)
    noise[:, patch_idx] = 1.0 # High noise = Masked
    return noise

def inpaint_16x16_sliding_window(model, pixel_values):
    """
    Iteratively masks and reconstructs every single 16x16 patch (196 passes).
    Stitches the predicted patches together to form a highly detailed hallucination.
    """
    stitched_recon = torch.zeros_like(pixel_values).to(DEVICE)
    grid_side = IMAGE_SIZE // PATCH_SIZE # 14
    
    for patch_idx in range(196):
        noise = get_single_patch_noise(pixel_values.shape[0], patch_idx)
        
        with torch.no_grad():
            outputs = model(pixel_values, noise=noise)
            recon = model.unpatchify(outputs.logits)
            

        row = patch_idx // grid_side
        col = patch_idx % grid_side
        y1, y2 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
        x1, x2 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
        

        stitched_recon[:, :, y1:y2, x1:x2] = recon[:, :, y1:y2, x1:x2]
        
    return stitched_recon

def calculate_au_pro(gt_masks, anomaly_maps, max_fpr=0.3, num_thresholds=100):
    """Calculates the Area Under the Per-Region Overlap Curve"""
    logging.info("Calculating AU-PRO. Finding Connected Components...")
    gt_all = np.concatenate([m.flatten() for m in gt_masks])
    scores_all = np.concatenate([a.flatten() for a in anomaly_maps])
    
    min_score, max_score = scores_all.min(), scores_all.max()
    thresholds = np.linspace(max_score, min_score, num_thresholds)
    

    components_list =[]
    num_components_total = 0
    for gt in gt_masks:
        num_labels, labels = cv2.connectedComponents(gt)
        comps =[np.where(labels == i) for i in range(1, num_labels)]
        components_list.append(comps)
        num_components_total += len(comps)
        
    if num_components_total == 0: return 1.0 # Perfect score if no anomalies exist
        
    fprs, pros = [],[]
    for th in thresholds:

        pred_all = (scores_all >= th)
        fp = np.sum((pred_all == 1) & (gt_all == 0))
        tn = np.sum(gt_all == 0)
        fpr = fp / tn if tn > 0 else 0
        fprs.append(fpr)
        

        pro_sum = 0
        for pred_map, comps in zip(anomaly_maps, components_list):
            pred_binary = pred_map >= th
            for comp in comps:
                overlap = pred_binary[comp].sum() / len(comp[0])
                pro_sum += overlap
        
        pros.append(pro_sum / num_components_total)
        if fpr > max_fpr: break
            

    au_pro = np.trapezoid(pros, fprs) / max_fpr
    return au_pro

def train_model(category):
    logging.info(f"--- Training Start: {category} ---")
    

    base_checkpoint_dir = f"checkpoints/exp1_{category}"
    final_model_dir = os.path.join(base_checkpoint_dir, "final")
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    

    if os.path.exists(final_model_dir):
        logging.info(f"Found fully trained model at {final_model_dir}. Skipping training and loading weights.")
        model = ViTMAEForPreTraining.from_pretrained(final_model_dir).to(DEVICE)
        return model
        

    dataset = MVTecAD2Dataset(DATASET_ROOT, category, split='train', status='good')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for pixel_values, _, _ in pbar:
            pixel_values = pixel_values.to(DEVICE)
            

            noise = get_random_patch_noise(pixel_values.shape[0])
            outputs = model(pixel_values, noise=noise)
            
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss/len(dataloader)
        logging.info(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        

        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(base_checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_dir)
            logging.info(f"Intermediate checkpoint saved to {checkpoint_dir}")
            

    model.save_pretrained(final_model_dir)
    logging.info(f"Final model saved securely to {final_model_dir}")
    
    return model

def find_threshold(model, category):
    logging.info(f"--- Calculating Threshold: {category} ---")
    dataset = MVTecAD2Dataset(DATASET_ROOT, category, split='validation', status='good')
    dataloader = DataLoader(dataset, batch_size=1)
    model.eval()
    
    all_pixel_errors =[]

    for idx, (pixel_values, _, _) in enumerate(tqdm(dataloader, desc="Thresholding")):
        if idx >= 20: break 
        
        pixel_values = pixel_values.to(DEVICE)
        recon = inpaint_16x16_sliding_window(model, pixel_values)
        
        diff = torch.abs(pixel_values - recon).mean(dim=1).squeeze().cpu().numpy()
        diff = cv2.GaussianBlur(diff, (3, 3), 0) # Light smoothing
        all_pixel_errors.append(diff.flatten())
        
    all_pixel_errors = np.concatenate(all_pixel_errors)
    mu, sigma = np.mean(all_pixel_errors), np.std(all_pixel_errors)
    threshold = mu + (3 * sigma)
    
    logging.info(f"Category: {category} | Pixel Mu: {mu:.6f} | Sigma: {sigma:.6f} | Threshold: {threshold:.6f}")
    return threshold

def evaluate_model(model, category, threshold):
    logging.info(f"--- Final Evaluation: {category} ---")
    dataset = MVTecAD2Dataset(DATASET_ROOT, category, split='test', status='bad')
    dataloader = DataLoader(dataset, batch_size=1)
    
    model.eval()
    os.makedirs(f"results/exp1_{category}", exist_ok=True)
    
    y_true_masks, y_score_maps, y_pred_binary = [], [],[]

    for i, (pixel_values, gt_mask, path) in enumerate(tqdm(dataloader, desc="Evaluating 16x16 sliding window")):
        pixel_values = pixel_values.to(DEVICE)
        gt_mask_np = gt_mask.squeeze().numpy()
        

        recon = inpaint_16x16_sliding_window(model, pixel_values)


        diff = torch.abs(pixel_values - recon).mean(dim=1).squeeze().cpu().numpy()
        heatmap = cv2.GaussianBlur(diff, (3, 3), 0)
        

        binary_pred = (heatmap > threshold).astype(np.uint8)
        
        y_true_masks.append(gt_mask_np)
        y_score_maps.append(heatmap)
        y_pred_binary.append(binary_pred.flatten())


        if i < 3:
            orig_img = np.clip((pixel_values.squeeze().cpu().permute(1,2,0).numpy() * 0.224) + 0.456, 0, 1)
            recon_img = np.clip((recon.squeeze().cpu().permute(1,2,0).numpy() * 0.224) + 0.456, 0, 1)
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(orig_img); axes[0].set_title("Original (Defect)")
            axes[1].imshow(recon_img); axes[1].set_title("16x16 Stitched Recon")
            axes[2].imshow(heatmap, cmap='hot'); axes[2].set_title("Error Heatmap")
            axes[3].imshow(binary_pred * 255, cmap='gray'); axes[3].set_title("Binary Prediction")
            axes[4].imshow(gt_mask_np * 255, cmap='gray'); axes[4].set_title("Ground Truth")
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/exp1_{category}/{os.path.basename(path[0])}")
            plt.close()


    logging.info("Computing metrics...")
    
    au_pro = calculate_au_pro(y_true_masks, y_score_maps, max_fpr=0.3)
    

    gt_all_flat = np.concatenate([m.flatten() for m in y_true_masks])
    pred_all_flat = np.concatenate(y_pred_binary)
    seg_f1 = f1_score(gt_all_flat, pred_all_flat)
    
    logging.info(f"RESULTS FOR {category}:")
    logging.info(f"AU-PRO (0.3): {au_pro:.4f}")
    logging.info(f"Segmentation F1: {seg_f1:.4f}")
    print(f"[{category}] AU-PRO: {au_pro:.4f} | SegF1: {seg_f1:.4f}")

if __name__ == "__main__":
    for cat in CATEGORIES:
        trained_model = train_model(cat)
        thresh = find_threshold(trained_model, cat)
        evaluate_model(trained_model, cat, thresh)
    logging.info("Experiment 1 Fully Completed.")