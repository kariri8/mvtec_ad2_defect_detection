import os
import torch
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from torch.optim import AdamW
from torchvision import transforms
from sklearn.metrics import f1_score

logging.basicConfig(
    filename='experiment_2_feature_predictor.log',
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
                gt_mask = gt_mask.crop((0, 0, new_w, new_h)) # Crop GT to match
                gt_mask = (np.array(gt_mask) > 127).astype(np.uint8)
            else:
                gt_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        else:
            gt_mask = np.zeros((new_h, new_w), dtype=np.uint8)
            
        return pixel_values, gt_mask, img_path

class FeaturePredictor(torch.nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(embed_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(embed_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1) # Final projection
        )

    def forward(self, x):
        return self.net(x)

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
    """
    Predicts patches by masking a grid. 
    A stride of 2 means we do 4 passes (2x2 grid offsets).
    """
    _, C, H, W = feature_map.shape
    combined_prediction = torch.zeros_like(feature_map)
    count_map = torch.zeros((1, 1, H, W)).to(DEVICE)

    predictor.eval()
    with torch.no_grad():
        for row_offset in range(stride):
            for col_offset in range(stride):

                mask = torch.ones((1, 1, H, W)).to(DEVICE)
                mask[:, :, row_offset::stride, col_offset::stride] = 0.0
                
                masked_input = feature_map * mask
                preds = predictor(masked_input)
                

                inv_mask = 1.0 - mask
                combined_prediction += (preds * inv_mask)
                count_map += inv_mask

    return combined_prediction / count_map.clamp(min=1)

def calculate_au_pro(gt_masks, anomaly_maps, max_fpr=0.3):
    gt_all = np.concatenate([m.flatten() for m in gt_masks])
    scores_all = np.concatenate([a.flatten() for a in anomaly_maps])
    thresholds = np.linspace(scores_all.max(), scores_all.min(), 100)
    
    components_list =[]
    num_components_total = 0
    for gt in gt_masks:
        num_labels, labels = cv2.connectedComponents(gt)
        comps = [np.where(labels == i) for i in range(1, num_labels)]
        components_list.append(comps)
        num_components_total += len(comps)
        
    if num_components_total == 0: return 1.0
        
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
                pro_sum += pred_binary[comp].sum() / len(comp[0])
        
        pros.append(pro_sum / num_components_total)
        if fpr > max_fpr: break
            
    return np.trapezoid(pros, fprs) / max_fpr

def run_experiment(category):
    logging.info(f"=== Starting Experiment 2: Feature Predictor for {category} ===")
    

    base_checkpoint_dir = f"checkpoints/exp2_{category}"
    final_model_path = os.path.join(base_checkpoint_dir, "final_predictor.pth")
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    

    dino_model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(DEVICE)
    dino_model.eval()
    

    predictor = FeaturePredictor(embed_dim=768).to(DEVICE)
    optimizer = AdamW(predictor.parameters(), lr=LR)
    

    if os.path.exists(final_model_path):
        logging.info(f"Found fully trained predictor at {final_model_path}. Loading weights...")
        predictor.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
    else:
        logging.info("Training Predictor...")
        train_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='train', status='good')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        
        predictor.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for pixel_values, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                pixel_values = pixel_values.to(DEVICE)
                feature_map = extract_dino_features(dino_model, pixel_values)
                

                mask = (torch.rand(feature_map.shape[-2:]) > 0.15).float().to(DEVICE)
                mask = mask.view(1, 1, feature_map.shape[2], feature_map.shape[3])

                inverse_mask = 1.0 - mask
                masked_features = feature_map * mask
                
                preds = predictor(masked_features)
                
                loss = torch.nn.functional.mse_loss(
                    preds * inverse_mask, 
                    feature_map * inverse_mask, 
                    reduction='sum' 
                ) / inverse_mask.sum()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            logging.info(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
            

            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(base_checkpoint_dir, f"predictor_epoch_{epoch+1}.pth")
                torch.save(predictor.state_dict(), checkpoint_path)
                logging.info(f"Intermediate checkpoint saved to {checkpoint_path}")
                

        torch.save(predictor.state_dict(), final_model_path)
        logging.info(f"Final predictor saved securely to {final_model_path}")
        

    logging.info("Calculating Threshold...")
    val_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='validation', status='good')
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    predictor.eval()
    all_errors =[]
    
    for pixel_values, _, _ in tqdm(val_loader, desc="Thresholding"):
        pixel_values = pixel_values.to(DEVICE)
        feature_map = extract_dino_features(dino_model, pixel_values)
        predicted_map = fast_grid_inference(feature_map, predictor)
        

        cos_sim = torch.nn.functional.cosine_similarity(feature_map, predicted_map, dim=1)
        dist_map = (1.0 - cos_sim).squeeze().cpu().numpy()
        all_errors.append(dist_map.flatten())
        
    all_errors = np.concatenate(all_errors)
    mu, sigma = np.mean(all_errors), np.std(all_errors)
    threshold = mu + 3 * sigma
    logging.info(f"Threshold: {threshold:.6f}")
    

    logging.info("Evaluating on Test Set...")
    test_dataset = MVTecAD2DynamicDataset(DATASET_ROOT, category, split='test', status='bad')
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    os.makedirs(f"results/exp2_{category}", exist_ok=True)
    y_true_masks, y_score_maps, y_pred_binary = [], [],[]
    
    for i, (pixel_values, gt_mask, path) in enumerate(tqdm(test_loader, desc="Testing")):
        pixel_values = pixel_values.to(DEVICE)
        gt_mask_np = gt_mask.squeeze().numpy()
        orig_h, orig_w = gt_mask_np.shape
        
        feature_map = extract_dino_features(dino_model, pixel_values)
        predicted_map = fast_grid_inference(feature_map, predictor)
        
        cos_sim = torch.nn.functional.cosine_similarity(feature_map, predicted_map, dim=1)
        dist_map = (1.0 - cos_sim).squeeze().cpu().numpy()
        

        heatmap_hd = cv2.resize(dist_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        binary_pred = (heatmap_hd > threshold).astype(np.uint8)
        
        y_true_masks.append(gt_mask_np)
        y_score_maps.append(heatmap_hd)
        y_pred_binary.append(binary_pred.flatten())
        

        if i < 5:
            orig_img = pixel_values.squeeze().cpu().permute(1,2,0).numpy()
            orig_img = np.clip((orig_img *[0.229, 0.224, 0.225]) +[0.485, 0.456, 0.406], 0, 1)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(orig_img); axes[0].set_title("Original")
            axes[1].imshow(heatmap_hd, cmap='hot'); axes[1].set_title("Semantic Distance Map")
            axes[2].imshow(binary_pred * 255, cmap='gray'); axes[2].set_title(f"Prediction (Thresh={threshold:.2f})")
            axes[3].imshow(gt_mask_np * 255, cmap='gray'); axes[3].set_title("Ground Truth")
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/exp2_{category}/{os.path.basename(path[0])}")
            plt.close()

    logging.info("Computing metrics...")
    au_pro = calculate_au_pro(y_true_masks, y_score_maps, max_fpr=0.3)
    gt_all_flat = np.concatenate([m.flatten() for m in y_true_masks])
    pred_all_flat = np.concatenate(y_pred_binary)
    seg_f1 = f1_score(gt_all_flat, pred_all_flat)
    
    logging.info(f"RESULTS FOR {category} | AU-PRO: {au_pro:.4f} | SegF1: {seg_f1:.4f}")
    print(f"[{category}] AU-PRO: {au_pro:.4f} | SegF1: {seg_f1:.4f}")

if __name__ == "__main__":
    for cat in CATEGORIES:
        run_experiment(cat)