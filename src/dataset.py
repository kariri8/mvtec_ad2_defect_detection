"""MVTec AD2 dataset loader shared across all experiments."""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PATCH_SIZE = 16

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MVTecAD2Dataset(Dataset):
    """
    MVTec AD2 dataset loader.

    Args:
        root_dir: Path to the dataset root (contains category folders).
        category: Category name, e.g. 'vial' or 'sheet_metal'.
        split: One of 'train', 'validation', 'test'.
        status: 'good' or 'bad'. Only relevant for 'test' split.
        resize: If given, resize images to (resize, resize) before cropping.
                If None, crops to nearest multiple of PATCH_SIZE.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        status: str = "good",
        resize: int | None = None,
    ) -> None:
        self.status = status
        self.split = split
        self.resize = resize

        if split == "train":
            self.image_dir = os.path.join(root_dir, category, "train", "good")
            self.gt_dir = None
        elif split == "validation":
            self.image_dir = os.path.join(root_dir, category, "validation", "good")
            self.gt_dir = None
        elif split == "test":
            self.image_dir = os.path.join(root_dir, category, "test_public", status)
            self.gt_dir = (
                os.path.join(root_dir, category, "test_public", "ground_truth", "bad")
                if status == "bad"
                else None
            )
        else:
            raise ValueError(f"Unknown split: {split}. Choose from 'train', 'validation', 'test'.")

        self.image_paths = sorted(
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith(".png")
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.resize is not None:
            image = image.resize((self.resize, self.resize))
            new_w, new_h = self.resize, self.resize
        else:
            w, h = image.size
            new_w = w - (w % PATCH_SIZE)
            new_h = h - (h % PATCH_SIZE)
            image = image.crop((0, 0, new_w, new_h))

        pixel_values = transform_norm(image)

        if self.split == "test" and self.status == "bad" and self.gt_dir:
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(self.gt_dir, f"{base}_mask.png")
            if os.path.exists(gt_path):
                gt_mask = Image.open(gt_path).convert("L").crop((0, 0, new_w, new_h))
                gt_mask = (np.array(gt_mask) > 127).astype(np.uint8)
            else:
                gt_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        else:
            gt_mask = np.zeros((new_h, new_w), dtype=np.uint8)

        return pixel_values, gt_mask, img_path
