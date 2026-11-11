"""Train the U-Net bone segmentation branch on ultrasound frames."""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.getcwd())

from unet.model import UNet
from unet.dataset import SegmentationDataset
from unet.loss import CombinedLoss

# ====================== Configuration ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_IMAGE_DIR = os.path.join('data', 'segment_data', 'processed_res4', 'images')
DATA_MASK_DIR = os.path.join('data', 'segment_data', 'processed_res4', 'masks')

TRAIN_RATIO = 0.8  # fraction of image/mask pairs used for training
BATCH_SIZE = 8
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
FOCAL_WEIGHT = 1.0  # focal term weight in CombinedLoss
DICE_WEIGHT = 1.0  # Dice term weight in CombinedLoss
UNET_FEATURES = [16, 32, 64, 128]  # channels per encoder level
VALID_PREVIEW_FREQ = 10  # save a validation preview every N epochs

SAVE_PATH = os.path.join('results', 'UNet')
SAVE_MODEL_DIR = os.path.join(SAVE_PATH, 'saved_model')
SAVE_VALID_DIR = os.path.join(SAVE_PATH, 'valid')
os.makedirs(os.path.join(os.getcwd(), SAVE_MODEL_DIR), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), SAVE_VALID_DIR), exist_ok=True)
# ============================================================


def dice_score(pred_binary, target):
    """Compute Dice coefficient for logging (scalar)."""
    intersection = (pred_binary * target).sum()
    denom = pred_binary.sum() + target.sum()
    if denom == 0:
        return torch.tensor(1.0)  # both empty → perfect match
    return (2.0 * intersection) / denom


def save_validation_preview(model, val_loader, device, epoch, save_dir, num_samples=4):
    """Save a grid of input, GT mask, predicted mask, and overlay."""
    model.eval()
    rows = []
    with torch.no_grad():
        for img, msk in val_loader:
            img = img.to(device)
            logits = model(img)
            pred = torch.sigmoid(logits).cpu().numpy()
            img = img.cpu().numpy()
            msk = msk.cpu().numpy()
            for j in range(min(img.shape[0], num_samples - len(rows))):
                rows.append((img[j, 0], msk[j, 0], pred[j, 0]))
            if len(rows) >= num_samples:
                break

    model.train()
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (inp, gt, pr) in enumerate(rows):
        mask_bin = (pr > 0.5).astype(np.float32)
        # Overlay: input + red predicted mask
        overlay = np.stack([inp, inp, inp], axis=-1)
        overlay[:, :, 0] = np.clip(inp + mask_bin * 0.5, 0, 1)

        axes[i, 0].imshow(inp, cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(mask_bin, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'epoch_{epoch:04d}.png'))
    plt.close(fig)


def main():
    # Fixed-seed train/val split of the image files
    all_files = sorted([
        f for f in os.listdir(DATA_IMAGE_DIR) if f.endswith('.npy')
    ])
    n_total = len(all_files)
    n_train = int(n_total * TRAIN_RATIO)
    rng = np.random.RandomState(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    train_files = sorted(shuffled[:n_train])
    val_files = sorted(shuffled[n_train:])

    print(f"Total pairs: {n_total}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    print(f"Device: {device}")

    ds_train = SegmentationDataset(DATA_IMAGE_DIR, DATA_MASK_DIR,
                                   file_list=train_files, augment=True)
    ds_val = SegmentationDataset(DATA_IMAGE_DIR, DATA_MASK_DIR,
                                 file_list=val_files, augment=False)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1, features=UNET_FEATURES).to(device)
    criterion = CombinedLoss(focal_weight=FOCAL_WEIGHT, dice_weight=DICE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    print(f"UNet features: {UNET_FEATURES}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    train_losses = []
    val_losses = []
    val_dices = []
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training ---
        model.train()
        epoch_train_loss = 0.0
        for img, msk in train_loader:
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss, _, _ = criterion(logits, msk)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * img.size(0)
        epoch_train_loss /= len(ds_train)

        # --- Validation ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_dice = 0.0
        total_pixels = 0
        with torch.no_grad():
            for img, msk in val_loader:
                img, msk = img.to(device), msk.to(device)
                logits = model(img)
                loss, _, _ = criterion(logits, msk)
                epoch_val_loss += loss.item() * img.size(0)

                pred_bin = (torch.sigmoid(logits) > 0.5).float()
                epoch_dice += dice_score(pred_bin, msk).item() * img.size(0)
                total_pixels += 1  # sample counter (unused)
        epoch_val_loss /= len(ds_val)
        epoch_dice /= len(ds_val)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_dices.append(epoch_dice)

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(),
                       os.path.join(SAVE_MODEL_DIR, 'best_model.pth'))

        print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Dice: {epoch_dice:.4f} | "
              f"Best Val Loss: {best_val_loss:.4f}")

        # --- LR scheduler ---
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"-> Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}")

        # --- Validation preview ---
        if epoch % VALID_PREVIEW_FREQ == 0:
            save_validation_preview(model, val_loader, device, epoch, SAVE_VALID_DIR)

    # --- Convergence plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax2.plot(val_dices, label='Val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_PATH, 'convergence.png'))
    plt.close(fig)

    # --- Config log ---
    with open(os.path.join(SAVE_PATH, 'config.txt'), 'w') as f:
        f.write(f"Train files: {len(train_files)}\n")
        f.write(f"Val files: {len(val_files)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Focal weight: {FOCAL_WEIGHT}\n")
        f.write(f"Dice weight: {DICE_WEIGHT}\n")
        f.write(f"UNet features: {UNET_FEATURES}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write(f"Best val dice: {val_dices[val_losses.index(min(val_losses))]:.4f}\n")

    # --- Final evaluation: save individual overlays for all val samples ---
    FINAL_EVAL_DIR = os.path.join(SAVE_PATH, 'valid', 'final_eval')
    os.makedirs(os.path.join(os.getcwd(), FINAL_EVAL_DIR), exist_ok=True)

    best_model = UNet(in_channels=1, out_channels=1, features=UNET_FEATURES).to(device)
    best_model.load_state_dict(torch.load(os.path.join(SAVE_MODEL_DIR, 'best_model.pth'),
                                          map_location=device))
    best_model.eval()

    sample_idx = 0
    with torch.no_grad():
        for img, msk in val_loader:
            img_gpu = img.to(device)
            logits = best_model(img_gpu)
            pred = torch.sigmoid(logits).cpu().numpy()
            img_np = img.cpu().numpy()
            msk_np = msk.cpu().numpy()
            for j in range(img_np.shape[0]):
                sample_idx += 1
                inp = img_np[j, 0]
                gt = msk_np[j, 0]
                pr = pred[j, 0]
                mask_bin = (pr > 0.5).astype(np.float32)
                overlay = np.stack([inp, inp, inp], axis=-1)
                overlay[:, :, 0] = np.clip(inp + mask_bin * 0.5, 0, 1)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(inp, cmap='gray')
                axes[0].set_title('Input')
                axes[0].axis('off')
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                axes[2].imshow(mask_bin, cmap='gray')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')
                axes[3].imshow(overlay)
                axes[3].set_title('Overlay')
                axes[3].axis('off')
                plt.tight_layout()
                fig.savefig(os.path.join(FINAL_EVAL_DIR, f'val_sample_{sample_idx:03d}.png'))
                plt.close(fig)

    print(f"Saved {sample_idx} final eval previews -> {FINAL_EVAL_DIR}")
    print("Training complete.")


if __name__ == '__main__':
    main()
