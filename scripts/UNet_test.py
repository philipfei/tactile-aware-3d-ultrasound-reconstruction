"""Run the trained U-Net on HDF5 ultrasound frames and save masks and overlays."""
import os
import sys
import numpy as np
import torch
import h5py
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.getcwd())

from unet.model import UNet

# ====================== Configuration ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESAMPLE_FACTOR = 4
UNET_FEATURES = [16, 32, 64, 128]

H5_PATH = os.path.join('data', f'frames_res{RESAMPLE_FACTOR}_test.h5')
H5_PATH_FLAT = os.path.join('data', f'frames_res{RESAMPLE_FACTOR}.h5')
MODEL_PATH = os.path.join('results', 'UNet', 'saved_model', 'best_model.pth')

SAVE_PATH = os.path.join('results', 'UNet', 'test')
os.makedirs(os.path.join(os.getcwd(), SAVE_PATH), exist_ok=True)

# None = all scans, or set to a list of (sub, scan) tuples
SELECTED_SCANS = None
# None = all frames, or set to an int to sample up to that many per scan
MAX_FRAMES_PER_SCAN = None
# ============================================================


def resolve_h5_path():
    # Prefer the test split; fall back to the full file
    if os.path.exists(H5_PATH):
        return H5_PATH
    if os.path.exists(H5_PATH_FLAT):
        print(f"Test split not found, falling back to {H5_PATH_FLAT}")
        return H5_PATH_FLAT
    raise FileNotFoundError(f"Neither {H5_PATH} nor {H5_PATH_FLAT} found.")


def save_overlay(frame, mask_bin, save_path):
    """Save input frame with red predicted mask overlay."""
    overlay = np.stack([frame, frame, frame], axis=-1)
    overlay[:, :, 0] = np.clip(frame + mask_bin.astype(np.float32) * 0.5, 0, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(frame, cmap='gray')
    ax1.set_title('Input Frame')
    ax1.axis('off')
    ax2.imshow(overlay)
    ax2.set_title('Predicted Mask Overlay')
    ax2.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    h5_path = resolve_h5_path()
    print(f"HDF5: {h5_path}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")

    # Load model
    model = UNet(in_channels=1, out_channels=1, features=UNET_FEATURES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    f = h5py.File(h5_path, 'r')
    num_frames = f['num_frames'][:]
    n_subjects = num_frames.shape[0]
    MAX_SCANS = num_frames.shape[1]

    total_processed = 0
    for i_sub in range(n_subjects):
        for i_scan in range(MAX_SCANS):
            n_frames = int(num_frames[i_sub, i_scan])
            if n_frames == 0:
                continue

            if SELECTED_SCANS is not None and (i_sub, i_scan) not in SELECTED_SCANS:
                continue

            if MAX_FRAMES_PER_SCAN is None:
                frame_indices = list(range(n_frames))
            else:
                frame_step = max(1, n_frames // MAX_FRAMES_PER_SCAN)
                frame_indices = list(range(0, n_frames, frame_step))[:MAX_FRAMES_PER_SCAN]

            scan_processed = 0
            for ifr in frame_indices:

                key = f'/sub{i_sub:03d}_scan{i_scan:02d}_frame{ifr:04d}'
                frame = f[key][:]  # (90, 60) float32 [0,1]

                x = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    pred = torch.sigmoid(logits).squeeze().cpu().numpy()
                mask_bin = (pred > 0.5).astype(np.uint8) * 255

                # Save overlay
                overlay_path = os.path.join(
                    SAVE_PATH,
                    f'scan_{i_sub:03d}_{i_scan:02d}_frame_{ifr:04d}.png'
                )
                save_overlay(frame, mask_bin, overlay_path)

                # Save raw mask
                mask_path = os.path.join(
                    SAVE_PATH,
                    f'scan_{i_sub:03d}_{i_scan:02d}_frame_{ifr:04d}_mask.npy'
                )
                np.save(mask_path, mask_bin)

                scan_processed += 1
                total_processed += 1

            print(f"Scan sub{i_sub:03d}_scan{i_scan:02d}: "
                  f"{scan_processed}/{n_frames} frames")

    f.close()
    print(f"Done. Saved {total_processed} frames → {SAVE_PATH}")


if __name__ == '__main__':
    main()
