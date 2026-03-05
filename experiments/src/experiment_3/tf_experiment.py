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

logging.basicConfig(
    filename='experiment_3_latent_mae.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DEVICE = "cuda"
EPOCHS = 15
LR = 1e-4
PATCH_SIZE = 16
CATEGORIES = ["vial", "sheet_metal"]
DATASET_ROOT = "../../data"

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False 
        )
        self.head = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask_map=None):
        B, C, H, W = x.shape
        seq_len = H * W
        
        x_seq = x.view(B, C, seq_len).permute(0, 2, 1) # (B, Seq, Dim)
        
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
        
        out_grid = out_seq.permute(0, 2, 1).view(B, C, H, W)
        return out_grid

def extract_dino_features(model, pixel_values):
    with torch.no_grad():
        outputs = model(pixel_values)
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        grid_h, grid_w = h // PATCH_SIZE, w // PATCH_SIZE
        num_patches = grid_h * grid_w
        
        patch_tokens = outputs.last_hidden_state[:, -num_patches:, :]
        feature_map = patch_tokens.permute(0, 2, 1).reshape(1, 768, grid_h, grid_w)
    return feature_map

def fast_grid_inference(feature_map, predictor, stride=2):
    _, C, H, W = feature_map.shape
    combined_prediction = torch.zeros_like(feature_map)
    count_map = torch.zeros((1, 1, H, W)).to(DEVICE)

    predictor.eval()
    with torch.no_grad():
        for row_offset in range(stride):
            for col_offset in range(stride):

                mask = torch.ones((1, 1, H, W)).to(DEVICE)
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
        
        if fpr >= max_fpr: 
            break
            
    fprs = np.array(fprs)
    pros = np.array(pros)
    

    if fprs[-1] > max_fpr:
        if len(fprs) > 1:
            pros[-1] = np.interp(max_fpr, [fprs[-2], fprs[-1]], [pros[-2], pros[-1]])
        fprs[-1] = max_fpr
            
    return np.trapezoid(pros, fprs) / max_fpr

def run_experiment_3(category):
    logging.info(f"\n{'='*50}\n=== Exp 3 (Latent MAE): {category} ===\n{'='*50}")
    
    base_checkpoint_dir = f"checkpoints/exp3_{category}"
    final_model_path = os.path.join(base_checkpoint_dir, "final_latent_mae.pth")
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    dino_model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(DEVICE).eval()
    predictor = LatentTransformerPredictor(embed_dim=768).to(DEVICE)
    optimizer = AdamW(predictor.parameters(), lr=LR)
    

    if os.path.exists(final_model_path):
        logging.info(f"Loaded existing model from {final_model_path}")
        predictor.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
    else:
        logging.info("Training Latent Transformer...")
        train_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='train', status='good')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        
        predictor.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for pixel_values, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                pixel_values = pixel_values.to(DEVICE)
                feature_map = extract_dino_features(dino_model, pixel_values)
                

                B, C, H, W = feature_map.shape
                mask = (torch.rand((B, 1, H, W)) > 0.50).float().to(DEVICE)
                inverse_mask = 1.0 - mask
                
                preds = predictor(feature_map, mask_map=mask)
                

                loss = torch.nn.functional.mse_loss(
                    preds * inverse_mask, 
                    feature_map * inverse_mask, 
                    reduction='sum'
                ) / inverse_mask.sum().clamp(min=1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            logging.info(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
        
        torch.save(predictor.state_dict(), final_model_path)
        logging.info("Training Complete & Saved.")


    predictor.eval()
    val_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='validation', status='good')
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    all_val_errors = []
    for pixel_values, _, _ in tqdm(val_loader, desc="Baseline Stats"):
        pixel_values = pixel_values.to(DEVICE)
        with torch.no_grad():
            feature_map = extract_dino_features(dino_model, pixel_values)
            predicted_map = fast_grid_inference(feature_map, predictor)
            cos_sim = torch.nn.functional.cosine_similarity(feature_map, predicted_map, dim=1)
            
        all_val_errors.append((1.0 - cos_sim).squeeze().cpu().numpy().flatten())
        
    mu = np.mean(np.concatenate(all_val_errors))
    sigma = np.std(np.concatenate(all_val_errors))
    logging.info(f"Baseline -> Mu: {mu:.6f} | Sigma: {sigma:.6f}")


    test_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='test', status='bad')
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    y_true_masks, y_score_maps, viz_cache = [], [], []
    for i, (pixel_values, gt_mask, path) in enumerate(tqdm(test_loader, desc="Heavy Inference")):
        pixel_values = pixel_values.to(DEVICE)
        gt_mask_np = gt_mask.squeeze().numpy()
        orig_h, orig_w = gt_mask_np.shape
        
        with torch.no_grad():
            feature_map = extract_dino_features(dino_model, pixel_values)
            predicted_map = fast_grid_inference(feature_map, predictor)
            cos_sim = torch.nn.functional.cosine_similarity(feature_map, predicted_map, dim=1)
            
        dist_map = (1.0 - cos_sim).squeeze().cpu().numpy()
        heatmap_hd = cv2.resize(dist_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        y_true_masks.append(gt_mask_np)
        y_score_maps.append(heatmap_hd)
        
        if i < 5:
            orig_img = pixel_values.squeeze().cpu().permute(1,2,0).numpy()
            orig_img = np.clip((orig_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
            viz_cache.append((os.path.basename(path[0]), orig_img, heatmap_hd, gt_mask_np))


    logging.info("\n--- Continuous Metrics ---")
    au_pro = calculate_au_pro(y_true_masks, y_score_maps, max_fpr=0.3)
    logging.info(f"AU-PRO (0.3): {au_pro:.4f}")


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
    print(f"[{category}] AU-PRO: {au_pro:.4f} | Best SegF1: {best_f1:.4f} (at Sigma x{best_mult})")


    out_dir = f"results/exp3_{category}_sigma_results"
    os.makedirs(out_dir, exist_ok=True)
    
    for filename, orig_img, heatmap, gt_mask_np in viz_cache:
        pred = (heatmap > best_thresh).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(orig_img); axes[0].set_title("Original")
        axes[1].imshow(heatmap, cmap='hot'); axes[1].set_title("Latent Cosine Distance")
        axes[2].imshow(pred * 255, cmap='gray'); axes[2].set_title(f"Pred (x{best_mult} σ)")
        axes[3].imshow(gt_mask_np * 255, cmap='gray'); axes[3].set_title("Ground Truth")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename}")
        plt.close()

if __name__ == "__main__":
    for cat in CATEGORIES:
        run_experiment_3(cat)