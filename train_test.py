import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel
from torch.optim import AdamW
from sklearn.metrics import f1_score
import logging
import concurrent.futures
import multiprocessing

MAX_EPOCHS = 50 
PATIENCE = 3
LR = 1e-4
PATCH_SIZE = 16

MAX_CONCURRENT_WORKERS = 2 
NUM_GPUS = 2

CATEGORIES = [
    "can", "fabric", "fruit_jelly", "rice", 
    "sheet_metal", "vial", "wallplugs", "walnuts"
]
DATASET_ROOT = "data"
RESULTS_FILE = "final_summary_results.txt"

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_logger(category, log_dir):
    logger = logging.getLogger(category)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh = logging.FileHandler(os.path.join(log_dir, f'training_{category}.log'), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class MVTecAD2DynamicDataset(Dataset):
    def __init__(self, root_dir, category, split='train', status='good'):
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
            
        self.image_paths = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        new_w, new_h = w - (w % PATCH_SIZE), h - (h % PATCH_SIZE)
        image = image.crop((0, 0, new_w, new_h))
        pixel_values = transform_norm(image)
        
        if self.split == 'test' and self.status == 'bad' and self.gt_dir:
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            gt_path = os.path.join(self.gt_dir, f"{base_name}_mask{ext}")
            
            if os.path.exists(gt_path):
                gt_mask = Image.open(gt_path).convert("L")
                gt_mask = gt_mask.crop((0, 0, new_w, new_h))
                gt_mask = (np.array(gt_mask) > 127).astype(np.uint8)
            else:
                gt_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        else:
            gt_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            
        return pixel_values, gt_mask, img_path

class LatentTransformerPredictor(torch.nn.Module):
    def __init__(self, embed_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, embed_dim, 64, 64) * 0.02)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False 
        )
        self.head = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask_map=None):
        B, C, H, W = x.shape
        seq_len = H * W
        
        x_seq = x.view(B, C, seq_len).permute(0, 2, 1) 
        if mask_map is not None:
            mask_seq = mask_map.view(B, seq_len, 1)
            x_seq = (x_seq * mask_seq) + (self.mask_token * (1.0 - mask_seq))
            
        pos_embed_resized = torch.nn.functional.interpolate(
            self.pos_embed, size=(H, W), mode='bicubic', align_corners=False
        )
        pos_embed_seq = pos_embed_resized.view(1, C, seq_len).permute(0, 2, 1)
        x_seq = x_seq + pos_embed_seq
        
        out_seq = self.transformer(x_seq)
        out_seq = self.head(out_seq)
        return out_seq.permute(0, 2, 1).view(B, C, H, W)

def extract_dino_features(model, pixel_values):
    with torch.no_grad():
        outputs = model(pixel_values)
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        grid_h, grid_w = h // PATCH_SIZE, w // PATCH_SIZE
        num_patches = grid_h * grid_w
        patch_tokens = outputs.last_hidden_state[:, -num_patches:, :]
        feature_map = patch_tokens.permute(0, 2, 1).reshape(1, 768, grid_h, grid_w)
    return feature_map

def fast_grid_inference(feature_map, predictor, device, stride=2):
    _, C, H, W = feature_map.shape
    combined_prediction = torch.zeros_like(feature_map)
    count_map = torch.zeros((1, 1, H, W)).to(device)

    predictor.eval()
    with torch.no_grad():
        for row_offset in range(stride):
            for col_offset in range(stride):
                mask = torch.ones((1, 1, H, W)).to(device)
                mask[:, :, row_offset::stride, col_offset::stride] = 0.0
                preds = predictor(feature_map, mask_map=mask)
                inv_mask = 1.0 - mask
                combined_prediction += (preds * inv_mask)
                count_map += inv_mask
    return combined_prediction / count_map.clamp(min=1)

def calculate_au_pro(gt_masks, anomaly_maps, max_fpr=0.3, num_thresholds=100):
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
        if fpr >= max_fpr: break
            
    fprs, pros = np.array(fprs), np.array(pros)
    if fprs[-1] > max_fpr:
        if len(fprs) > 1: pros[-1] = np.interp(max_fpr, [fprs[-2], fprs[-1]], [pros[-2], pros[-1]])
        fprs[-1] = max_fpr
    return np.trapezoid(pros, fprs) / max_fpr

def process_category(category, gpu_id, worker_idx):
    device = f"cuda:{gpu_id}"
    
    base_checkpoint_dir = f"checkpoints/{category}"
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    out_dir = f"results/{category}_sigma_results"
    os.makedirs(out_dir, exist_ok=True)
    
    log = get_logger(category, base_checkpoint_dir)
    log.info(f"=== Starting {category} on GPU {gpu_id} ===")
    
    best_model_path = os.path.join(base_checkpoint_dir, "best_latent_mae.pth")
    
    dino_model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(device).eval()
    predictor = LatentTransformerPredictor(embed_dim=768).to(device)
    optimizer = AdamW(predictor.parameters(), lr=LR)
    

    best_model_path = os.path.join(base_checkpoint_dir, "best_latent_mae.pth")
    checkpoint_path = os.path.join(base_checkpoint_dir, "training_state.pth")
    complete_flag = os.path.join(base_checkpoint_dir, "training_complete.flag")
    
    if os.path.exists(complete_flag):
        log.info("Training already completed. Loading best model for evaluation...")
        predictor.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        train_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='train', status='good')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        
        start_epoch = 0
        best_loss = float('inf')
        patience_counter = 0
        

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            predictor.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            patience_counter = checkpoint['patience_counter']
            log.info(f"Resuming from epoch {start_epoch} (Best Loss: {best_loss:.4f})")
            

        elif os.path.exists(best_model_path):
            predictor.load_state_dict(torch.load(best_model_path, map_location=device))
            log.info("Recovered weights from interrupted run. Resuming training...")

        predictor.train()
        for epoch in range(start_epoch, MAX_EPOCHS):
            epoch_loss = 0
            
            pbar = tqdm(train_loader, desc=f"[{category}] Ep {epoch+1}/{MAX_EPOCHS}", position=worker_idx, leave=False)
            for pixel_values, _, _ in pbar:
                pixel_values = pixel_values.to(device)
                feature_map = extract_dino_features(dino_model, pixel_values)
                
                B, C, H, W = feature_map.shape
                mask = (torch.rand((B, 1, H, W)) > 0.50).float().to(device)
                inverse_mask = 1.0 - mask
                
                preds = predictor(feature_map, mask_map=mask)
                loss = torch.nn.functional.mse_loss(
                    preds * inverse_mask, feature_map * inverse_mask, reduction='sum'
                ) / inverse_mask.sum().clamp(min=1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            avg_loss = epoch_loss / len(train_loader)
            log.info(f"Epoch {epoch+1}/{MAX_EPOCHS} - Loss: {avg_loss:.4f}")
            

            torch.save({
                'epoch': epoch,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }, checkpoint_path)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(predictor.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                log.info(f"Early stopping triggered at epoch {epoch+1}!")
                break
                

        with open(complete_flag, 'w') as f:
            f.write("done")
            
        predictor.load_state_dict(torch.load(best_model_path, map_location=device))


    predictor.eval()
    val_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='validation', status='good')
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    all_val_errors = []
    

    for pixel_values, _, _ in tqdm(val_loader, desc=f"[{category}] Baseline", position=worker_idx, leave=False):
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            feature_map = extract_dino_features(dino_model, pixel_values)
            predicted_map = fast_grid_inference(feature_map, predictor, device)
            cos_sim = torch.nn.functional.cosine_similarity(feature_map, predicted_map, dim=1)
        all_val_errors.append((1.0 - cos_sim).squeeze().cpu().numpy().flatten())
        
    mu = np.mean(np.concatenate(all_val_errors))
    sigma = np.std(np.concatenate(all_val_errors))
    log.info(f"Baseline -> Mu: {mu:.6f} | Sigma: {sigma:.6f}")


    test_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='test', status='bad')
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    y_true_masks, y_score_maps, viz_cache = [], [], []
    

    for i, (pixel_values, gt_mask, path) in enumerate(tqdm(test_loader, desc=f"[{category}] Eval", position=worker_idx, leave=False)):
        pixel_values = pixel_values.to(device)
        gt_mask_np = gt_mask.squeeze().numpy()
        orig_h, orig_w = gt_mask_np.shape
        
        with torch.no_grad():
            feature_map = extract_dino_features(dino_model, pixel_values)
            predicted_map = fast_grid_inference(feature_map, predictor, device)
            cos_sim = torch.nn.functional.cosine_similarity(feature_map, predicted_map, dim=1)
            
        dist_map = (1.0 - cos_sim).squeeze().cpu().numpy()
        heatmap_hd = cv2.resize(dist_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        y_true_masks.append(gt_mask_np)
        y_score_maps.append(heatmap_hd)
        
        if i < 3: 
            orig_img = pixel_values.squeeze().cpu().permute(1,2,0).numpy()
            orig_img = np.clip((orig_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
            viz_cache.append((os.path.basename(path[0]), orig_img, heatmap_hd, gt_mask_np))

    au_pro = calculate_au_pro(y_true_masks, y_score_maps, max_fpr=0.3)
    
    gt_all_flat = np.concatenate([m.flatten() for m in y_true_masks])
    scores_all_flat = np.concatenate([s.flatten() for s in y_score_maps])
    
    best_f1, best_mult, best_thresh = 0, 0, 0
    multipliers = list(range(1, 16, 2)) 
    
    for m in multipliers:
        thresh = mu + (m * sigma)
        pred_binary = (scores_all_flat > thresh).astype(np.uint8)
        f1 = f1_score(gt_all_flat, pred_binary)
        if f1 > best_f1:
            best_f1, best_mult, best_thresh = f1, m, thresh

    log.info(f"AU-PRO (0.3): {au_pro:.4f} | Best SegF1: {best_f1:.4f} (Sigma x{best_mult})")
    
    for filename, orig_img, heatmap, gt_mask_np in viz_cache:
        pred = (heatmap > best_thresh).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(orig_img); axes[0].set_title("Original")
        axes[1].imshow(heatmap, cmap='hot'); axes[1].set_title("Latent Distance")
        axes[2].imshow(pred * 255, cmap='gray'); axes[2].set_title(f"Pred (x{best_mult} σ)")
        axes[3].imshow(gt_mask_np * 255, cmap='gray'); axes[3].set_title("Ground Truth")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename}")
        plt.close()

    return {
        "category": category,
        "au_pro": au_pro,
        "best_sigma": best_mult,
        "best_segf1": best_f1
    }

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) 
    
    print(f"Starting parallel processing for {len(CATEGORIES)} categories...")
    print(f"Check the respective 'checkpoints/<category>' folders for detailed training logs.\n")
    

    print("\n" * MAX_CONCURRENT_WORKERS)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        futures = {}
        for i, cat in enumerate(CATEGORIES):
            gpu_id = i % NUM_GPUS 
            worker_idx = i  # Used to assign a specific terminal line for the progress bar
            futures[executor.submit(process_category, cat, gpu_id, worker_idx)] = cat
            
        for future in concurrent.futures.as_completed(futures):
            cat = futures[future]
            try:
                res = future.result()
                results.append(res)

                tqdm.write(f"✅ Finished {cat} -> AU-PRO: {res['au_pro']:.4f} | SegF1: {res['best_segf1']:.4f} (Sigma x{res['best_sigma']})")
            except Exception as e:
                tqdm.write(f"❌ ERROR processing category {cat}: {e}")
                
    if results:
        avg_au_pro = np.mean([r['au_pro'] for r in results])
        avg_segf1 = np.mean([r['best_segf1'] for r in results])
        
        with open(RESULTS_FILE, 'w') as f:
            f.write("="*50 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"{'Category':<15} | {'AU-PRO':<10} | {'Sigma':<6} | {'SegF1':<10}\n")
            f.write("-" * 50 + "\n")
            
            for r in sorted(results, key=lambda x: x['category']):
                f.write(f"{r['category']:<15} | {r['au_pro']:<10.4f} | x{r['best_sigma']:<5} | {r['best_segf1']:<10.4f}\n")
                
            f.write("-" * 50 + "\n")
            f.write(f"{'AVERAGE':<15} | {avg_au_pro:<10.4f} | {'-':<6} | {avg_segf1:<10.4f}\n")
            
        print(f"\nAll operations complete! Aggregated results saved to {RESULTS_FILE}")