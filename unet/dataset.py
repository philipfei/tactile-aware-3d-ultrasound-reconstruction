import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list=None, augment=False):
        if file_list is None:
            self.filenames = sorted([
                f for f in os.listdir(image_dir) if f.endswith('.npy')
            ])
        else:
            self.filenames = list(file_list)

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = np.load(os.path.join(self.image_dir, fname))  # (H, W) float32 [0,1]
        msk = np.load(os.path.join(self.mask_dir, fname))   # (H, W) float32 {0,1}

        if self.augment:
            img, msk = self._apply_augmentation(img, msk)

        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        msk = torch.from_numpy(msk).unsqueeze(0)  # (1, H, W)
        return img, msk

    def _apply_augmentation(self, img, msk):
        """
        Joint augmentation applied synchronously to image and mask.
        Intensity augmentations applied BEFORE spatial transforms so
        black padded borders from rotation/translation stay at 0.0.
        Image: float32 [0,1], Mask: float32 {0,1}.
        """
        H, W = img.shape

        # --- Intensity augmentations (image only) ---
        # Random brightness/contrast
        img = img * np.random.uniform(0.8, 1.2)
        # Random Gaussian noise
        img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
        # Clip back to [0, 1]
        img = np.clip(img, 0.0, 1.0)

        # --- Spatial augmentations (joint image + mask) ---

        # 1. Random horizontal flip (p=0.5)
        if np.random.random() < 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)

        # 2. Combined affine transform: rotation ±10° + scale ±10% + translation ±5%
        center = (W / 2.0, H / 2.0)
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += np.random.uniform(-0.05 * W, 0.05 * W)
        M[1, 2] += np.random.uniform(-0.05 * H, 0.05 * H)

        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        msk = cv2.warpAffine(msk, M, (W, H), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Re-binarize mask after affine transforms
        msk = (msk > 0.5).astype(np.float32)

        return img, msk
