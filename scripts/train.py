"""Train the pose prediction network (BlockSequenceModel)."""
import os
import sys
import datetime
import numpy as np
import torch
import cv2
from torchvision.models import efficientnet_b1
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from freehand.loader import SSFrameDataset
from freehand.network import (build_model, SensorFusionModel,
                              BlockSequenceModel, load_pretrained_image_weights)
from freehand.loss import PointDistance
from data.calib import read_calib_matrices
from freehand.transform import (LabelTransform, PredictionTransform,
                                PredictionTransform6D, sixd_to_matrix)
from freehand.utils import *

# ====================== Configuration ======================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark = True         # auto-tune conv algorithms for fixed-size inputs
# torch.backends.cuda.matmul.allow_tf32 = True   # enable TF32 on Ampere/Ada tensor cores

RESAMPLE_FACTOR = 4  # image downsampling factor of the HDF5 frames
FILENAME_CALIB = "data/calib_matrix.csv"  # calibration matrix T_cal (image to tool)

ROTATE_IMAGE_90 = False
ROTATE_TOOL_90  = False  # apply a 90 deg in-plane rotation to T_cal

# ------ Ablation Flags: Sensor Modalities ------
USE_IMU_ACCEL = True   # IMU acceleration (3 channels)
USE_IMU_QUAT  = False  # IMU quaternion (4 channels)
USE_FORCE     = True   # tactile force (5 channels)
# ----------------------------------------------

# ------ Transfer Learning ------
PRETRAINED_WEIGHTS = None  # path to backbone weights, or None for random init
FREEZE_STAGES = 0   # number of leading EfficientNet stages to freeze (-1 = all 9)
# ------------------------------

# ------ Training ------
LEARNING_RATE = 1e-4
NUM_EPOCHS = int(2500)

# ------ Sequence Mode ------
SEQUENCE_MODE = True
SEQ_LEN = 5            # blocks per window
BLOCK_SIZE = 5         # frames per block (stacked as EfficientNet input channels)
TRAIN_STRIDE = 3        # frame stride between overlapping training windows
TEMPORAL_MODEL = "gru"  # "tcn" or "gru"

BATCH_SIZE = 64                   # windows per batch
GRADIENT_ACCUMULATION_STEPS = 1   # 1 = no accumulation
USE_AMP = True                    # mixed-precision training
GRAD_CLIP_MAX_NORM = 1.0

LOSS_TRAJECTORY_WEIGHT = 0.65  # L_traj weight
LOSS_AUX_WEIGHT = 0.25         # L_aux weight
LOSS_EDGE_WEIGHT = 0.0         # edge-length ratio term: logged only, not in the loss
# ---------------------------

SEN_WINDOW_SIZE = 5   # sensor trend window length (samples)
SENSOR_DROPOUT = 0.0  # probability of zeroing the sensor window of a sample

FREQ_INFO = 1   # print training metrics every N epochs
FREQ_SAVE = 10  # validate and save checkpoints every N epochs

# ------ Results Path ------
sensor_tag = ""
if USE_IMU_ACCEL: sensor_tag += "A"
if USE_IMU_QUAT:  sensor_tag += "Q"
if USE_FORCE:     sensor_tag += "F"
if not sensor_tag: sensor_tag = "ImageOnly"
if SEQUENCE_MODE: sensor_tag += "_seq"

saved_results = 'seq_len' + str(SEQ_LEN if SEQUENCE_MODE else 20) \
    + '_blk' + str(BLOCK_SIZE) \
    + '__lr' + str(LEARNING_RATE) \
    + '__pred_parameter__label_point__sensor_' + sensor_tag
SAVE_PATH = os.path.join('results', saved_results)
os.makedirs(os.path.join(os.getcwd(), SAVE_PATH), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), SAVE_PATH, 'saved_model'), exist_ok=True)
writer = SummaryWriter(os.path.join(os.getcwd(), SAVE_PATH))

# Preview hook: saves the first N model-input frames as PNG
PREVIEW_DIR = os.path.join(os.getcwd(), SAVE_PATH, 'preview')
PREVIEW_MAX_IMAGES = 50
# =========================================================

_preview_saved = 0
_preview_disabled = False


def resolve_split_h5(split):
    data_dir = os.path.join(os.getcwd(), "data")
    split_h5 = os.path.join(data_dir, f'frames_res{RESAMPLE_FACTOR}_{split}.h5')
    flat_h5 = os.path.join(data_dir, f'frames_res{RESAMPLE_FACTOR}.h5')
    if os.path.exists(split_h5):
        return split_h5
    if os.path.exists(flat_h5):
        print(f"[Info] Split HDF5 not found ({split_h5}). Falling back to {flat_h5}.")
        return flat_h5
    raise FileNotFoundError(f"Neither {split_h5} nor {flat_h5} found.")


# --- Channel selection (sensor ablation) ---
def compute_sensor_channels():
    win_indices, inst_indices = [], []
    if USE_IMU_ACCEL:
        win_indices.extend([0, 1, 2]); inst_indices.extend([0, 1, 2])
    if USE_IMU_QUAT:
        win_indices.extend([3, 4, 5, 6]); inst_indices.extend([3, 4, 5, 6])
    if USE_FORCE:
        win_indices.extend([7, 8, 9, 10, 11]); inst_indices.extend([7, 8, 9, 10, 11])
    inst_indices.append(12)   # dt (time step)

    # Time-position channel at index 12 of sensor_wins
    if SEN_WINDOW_SIZE > 0:
        win_indices.append(12)

    cnn_ch = len(win_indices) if SEN_WINDOW_SIZE > 0 else 0
    inst_dim = len(inst_indices)  # instantaneous sensor channels per frame
    use_sensor = (cnn_ch > 0) or (len(inst_indices) > 1)
    return cnn_ch, inst_dim, win_indices, inst_indices, use_sensor


def select_window_channels(sensor_wins, win_indices):
    if not win_indices: return None
    return sensor_wins[..., win_indices]

def select_inst_channels(sensors, inst_indices):
    if not inst_indices: return None
    return sensors[..., inst_indices]


# --- Plot helpers ---
def _plot_convergence(train_epochs, train_loss, train_dist,
                      val_epochs, val_loss, val_dist, val_aux_dist, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_epochs, train_loss, 'b-', label='Train', alpha=0.7)
    if val_epochs:
        ax1.plot(val_epochs, val_loss, 'r-', label='Val', alpha=0.7)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss Convergence')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(train_epochs, train_dist, 'b-', label='Train Frame-to-Frame', alpha=0.7)
    if val_epochs:
        ax2.plot(val_epochs, val_dist, 'r--', label='Val Accumulated', alpha=0.7)
        ax2.plot(val_epochs, val_aux_dist, 'r-', label='Val Frame-to-Frame', alpha=0.7)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Distance (mm)'); ax2.set_title('Distance Convergence')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), save_path, 'convergence.png'), dpi=150)
    plt.close(fig)


def _write_config_log(epoch, train_loss, train_dist, train_traj, train_aux,
                      val_loss, val_dist, val_traj, val_aux,
                      val_loss_min, val_dist_min, sensor_flags, save_path):
    log_path = os.path.join(os.getcwd(), save_path, 'config.txt')
    with open(log_path, 'a') as f:
        f.write(f'--- Best model update at epoch {epoch} ---\n')
        f.write(f'  Train: loss={train_loss:.4f}, dist={train_dist:.4f} mm, traj={train_traj:.4f}, aux={train_aux:.4f}\n')
        f.write(f'  Val:   loss={val_loss:.4f}, dist={val_dist:.4f} mm, traj={val_traj:.4f}, aux={val_aux:.4f}\n')
        f.write(f'  Best:  loss={val_loss_min:.4f}, dist={val_dist_min:.4f}\n')
        f.write(f'  Sensor: ACCEL={sensor_flags[0]}, QUAT={sensor_flags[1]}, FORCE={sensor_flags[2]}\n')
        f.write(f'  Timestamp: {datetime.datetime.now().isoformat()}\n\n')


_EPOCH_LOG_HEADER = ('Epoch\ttrain_L_traj\ttrain_L_aux\ttrain_L_edge\ttrain_L\t'
                     'train_eps_acc_mm\ttrain_eps_block_mm\t'
                     'val_L_traj\tval_L_aux\tval_L_edge\tval_L\t'
                     'val_eps_acc_mm\tval_eps_block_mm')


def _write_epoch_log(epoch, t, v, save_path):
    """Append one tab-separated row to <save_path>/epoch_log.txt.

    L_traj = trajectory accumulation loss, L_aux = auxiliary relative accuracy
    loss, L_edge = edge-length ratio term (logged only), L = composite loss,
    eps_acc = accumulated tracking error (mm), eps_block = per-block prediction
    error (mm). v is None on non-validation epochs (val columns written as NaN).
    """
    log_path = os.path.join(os.getcwd(), save_path, 'epoch_log.txt')
    is_new = not os.path.exists(log_path)
    with open(log_path, 'a') as f:
        if is_new:
            f.write('# Metrics per epoch: '
                    'L_traj = trajectory accumulation loss, '
                    'L_aux = auxiliary relative accuracy loss, '
                    'L_edge = edge-length ratio term (logged only), '
                    'L = 0.65*L_traj + 0.25*L_aux, '
                    'eps_acc = accumulated tracking error (mm), '
                    'eps_block = per-block prediction error (mm)\n')
            f.write(_EPOCH_LOG_HEADER + '\n')
        cells = [str(epoch)]
        for x in (t['L_traj'], t['L_aux'], t['L_edge'], t['L'],
                  t['eps_acc_mm'], t['eps_block_mm']):
            cells.append(f'{x:.6f}')
        if v is None:
            cells += ['NaN'] * 6
        else:
            for x in (v['L_traj'], v['L_aux'], v['L_edge'], v['L'],
                      v['eps_acc_mm'], v['eps_block_mm']):
                cells.append(f'{x:.6f}')
        f.write('\t'.join(cells) + '\n')


def _plot_modality_contribution(history, save_path):
    """Plot and log each modality's contribution (relative output change when ablated) per epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = history['epochs']
    colors = {'image': '#2196F3', 'accel': '#4CAF50', 'quat': '#FF9800', 'force': '#F44336'}
    labels = {'image': 'Image', 'accel': 'IMU Accel', 'quat': 'IMU Quat', 'force': 'Force'}
    for key in ['image', 'accel', 'quat', 'force']:
        if len(history[key]) == len(epochs):
            ax.plot(epochs, history[key], 'o-', color=colors[key], label=labels[key],
                    markersize=4, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Relative output change  ||Δout|| / ||out||')
    ax.set_title('Modality Contribution vs. Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), save_path, 'modality_contribution.png'), dpi=150)
    plt.close(fig)

    log_path = os.path.join(os.getcwd(), save_path, 'modality_contribution.txt')
    with open(log_path, 'w') as f:
        f.write(f"{'Epoch':<10} {'Image':<15} {'Accel':<15} {'Quat':<15} {'Force':<15}\n")
        for i, epoch in enumerate(epochs):
            def gv(k): return f"{history[k][i]:.4f}" if len(history[k]) == len(epochs) else "N/A"
            f.write(f"{epoch:<10} {gv('image'):<15} {gv('accel'):<15} {gv('quat'):<15} {gv('force'):<15}\n")


def _save_preview_images(frames, preview_dir, max_images):
    """
    Convert float32 [0,1] batch frames to uint8 [0,255] and save as PNG.
    Stops after max_images in total across all calls.

    :param frames: torch.Tensor (B, SEQ_LEN, BLOCK_SIZE, H, W), float32 [0,1]
    """
    global _preview_saved, _preview_disabled
    if _preview_disabled:
        return

    os.makedirs(preview_dir, exist_ok=True)
    flat = frames.detach().cpu().numpy().reshape(-1, frames.shape[-2], frames.shape[-1])

    for i in range(flat.shape[0]):
        if _preview_saved >= max_images:
            _preview_disabled = True
            print(f"[Preview] Reached limit of {max_images} images. Preview hook disabled.")
            break
        img_uint8 = np.clip(np.round(flat[i] * 255.0), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(preview_dir, f'preview_{_preview_saved:04d}.png'), img_uint8)
        _preview_saved += 1

    if _preview_saved > 0 and not _preview_disabled:
        print(f"[Preview] Saved {_preview_saved} image(s) -> {preview_dir}")


# ================================================================
#  Sequence-mode helpers
# ================================================================

def build_sequence_samples(dataset, seq_frames, stride):
    """List (scan index, start frame) for all overlapping windows of seq_frames frames."""
    samples = []
    for idx, (i_sub, i_scan) in enumerate(dataset.indices_in_use):
        n_frames = int(dataset.num_frames[i_sub, i_scan])
        if n_frames < seq_frames:
            continue
        for start in range(0, n_frames - seq_frames + 1, stride):
            samples.append((idx, start))
    return samples


class SequenceDatasetWrapper:
    """Wrap SSFrameDataset to return sequence-mode samples."""
    def __init__(self, dataset, samples, seq_frames):
        self.dataset = dataset
        self.samples = samples      # list of (scan_pool_idx, start_frame_idx)
        self.seq_frames = seq_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan_pool_idx, start = self.samples[idx]
        i_sub, i_scan = self.dataset.indices_in_use[scan_pool_idx]
        i_frames = list(range(start, start + self.seq_frames))

        f = self.dataset._get_file()
        frames = [f['/sub{:03d}_scan{:02d}_frame{:04d}'.format(i_sub, i_scan, ii)]
                  for ii in i_frames]
        tforms = [f['/sub{:03d}_scan{:02d}_tform{:04d}'.format(i_sub, i_scan, ii)]
                  for ii in i_frames]
        tforms_inv = [f['/sub{:03d}_scan{:02d}_tform_inv{:04d}'.format(i_sub, i_scan, ii)]
                       for ii in i_frames]

        if not self.dataset.sensor_in_use:
            return (np.stack(frames, axis=0), np.stack(tforms, axis=0),
                    np.stack(tforms_inv, axis=0))

        sensors = np.zeros((len(i_frames), 14), dtype=np.float32)
        for j, ifr in enumerate(i_frames):
            skey = '/sub{:03d}_scan{:02d}_sensor{:04d}'.format(i_sub, i_scan, ifr)
            sensors[j] = f[skey][:]

        ws = self.dataset.sen_window_size
        scan_key = (i_sub, i_scan)
        sensor_seq = self.dataset._sensor_seqs.get(scan_key)
        sensor_map = self.dataset._sensor_maps.get(scan_key)
        raw_ch = sensor_seq.shape[1] if sensor_seq is not None else 12

        sensor_wins = np.zeros((len(i_frames), max(ws, 1), raw_ch + 1), dtype=np.float32)
        win_lens = np.full(len(i_frames), ws, dtype=np.int32)

        # Time-position channel: 0..1 across the sensor trend window
        time_pos = np.linspace(0, 1, ws, dtype=np.float32) if ws > 1 else np.array([0], dtype=np.float32)
        sensor_wins[:, :, -1] = time_pos[np.newaxis, :]

        if ws > 0 and sensor_seq is not None and sensor_map is not None:
            for j, ifr in enumerate(i_frames):
                csv_row = int(sensor_map[ifr])
                st = max(0, csv_row - ws + 1)
                window = sensor_seq[st:csv_row + 1]
                al = window.shape[0]
                if al >= ws:
                    sensor_wins[j, :, :raw_ch] = window[-ws:]
                elif al > 0:  # short window at scan start: pad with its first sample
                    sensor_wins[j, -al:, :raw_ch] = window
                    pad = ws - al
                    sensor_wins[j, :pad, :raw_ch] = np.tile(window[0:1], (pad, 1))

        return (np.stack(frames, axis=0), np.stack(tforms, axis=0),
                np.stack(tforms_inv, axis=0), sensors, sensor_wins, win_lens)


def validate_sequence(model, dset_val, seq_len, tform_calib, pts_img,
                       win_indices, inst_indices, pair_weights, criterion, metrics):
    """Sliding-window validation over full scans.

    Poses are accumulated across windows and eps_acc is measured at the
    block-representative frames. Returns (L, eps_acc, L_traj, L_aux, L_edge, eps_block).
    """
    model.train(False)
    total_loss = 0.0
    total_dist = 0.0
    total_aux = 0.0
    total_aux_dist = 0.0
    total_edge = 0.0
    n_scans = 0
    seq_frames = seq_len * BLOCK_SIZE
    val_sensor = dset_val.sensor_in_use

    stride_blocks = 1
    stride_frames = stride_blocks * BLOCK_SIZE

    with torch.no_grad():
        for idx in range(len(dset_val)):
            i_sub, i_scan = dset_val.indices_in_use[idx]
            n_frames = int(dset_val.num_frames[i_sub, i_scan])
            if n_frames < seq_frames:
                continue
            n_scans += 1

            # Load all frames for this scan
            f = dset_val._get_file()
            all_frames, all_tforms, all_tforms_inv = [], [], []
            all_sensors, all_sw, all_wl = [], [], []
            for i_frame in range(n_frames):
                all_frames.append(f['/sub{:03d}_scan{:02d}_frame{:04d}'.format(i_sub, i_scan, i_frame)][:])
                all_tforms.append(f['/sub{:03d}_scan{:02d}_tform{:04d}'.format(i_sub, i_scan, i_frame)][:])
                all_tforms_inv.append(f['/sub{:03d}_scan{:02d}_tform_inv{:04d}'.format(i_sub, i_scan, i_frame)][:])
                if val_sensor:
                    skey = '/sub{:03d}_scan{:02d}_sensor{:04d}'.format(i_sub, i_scan, i_frame)
                    all_sensors.append(f[skey][:])

            all_frames = np.stack(all_frames)      # (N_scan, H, W)
            all_tforms = np.stack(all_tforms)       # (N_scan, 4, 4)

            # Sensor windows (if needed)
            if val_sensor:
                all_sensors = np.stack(all_sensors)
                ws = dset_val.sen_window_size
                scan_key = (i_sub, i_scan)
                sensor_seq = dset_val._sensor_seqs.get(scan_key)
                sensor_map = dset_val._sensor_maps.get(scan_key)
                raw_ch = sensor_seq.shape[1] if sensor_seq is not None else 12
                all_sw = np.zeros((n_frames, max(ws, 1), raw_ch + 1), dtype=np.float32)
                tp = np.linspace(0, 1, ws, dtype=np.float32) if ws > 1 else np.array([0], dtype=np.float32)
                all_sw[:, :, -1] = tp[np.newaxis, :]
                if ws > 0 and sensor_seq is not None and sensor_map is not None:
                    for j in range(n_frames):
                        csv_row = int(sensor_map[j])
                        st = max(0, csv_row - ws + 1)
                        w = sensor_seq[st:csv_row + 1]
                        al = w.shape[0]
                        if al >= ws: all_sw[j, :, :raw_ch] = w[-ws:]
                        elif al > 0:
                            all_sw[j, -al:, :raw_ch] = w
                            p = ws - al; all_sw[j, :p, :raw_ch] = np.tile(w[0:1], (p, 1))
                all_wl = np.full(n_frames, ws, dtype=np.int32)
            else:
                all_sensors = np.zeros((n_frames, 1)); all_sw = np.zeros((n_frames, 1, 1)); all_wl = None

            # GT corner points for all frames
            gt_corners = []
            for i_frame in range(n_frames):
                tform = torch.tensor(all_tforms[i_frame], device=device).unsqueeze(0)
                gt_c = tform @ tform_calib @ pts_img
                gt_corners.append(gt_c[0, :3, :].cpu())
            gt_corners = torch.stack(gt_corners)  # (N_scan, 3, 4)

            # Sliding window; the pose at the end of each window seeds the next
            all_pred_corners = [gt_corners[0].clone() for _ in range(n_frames)]
            # Seed: the last frame of block 0 starts from the GT pose
            pred_poses = {}
            pred_poses[BLOCK_SIZE - 1] = torch.tensor(all_tforms[BLOCK_SIZE - 1], device=device)
            aux_losses = []
            aux_dists = []
            edge_losses = []
            window_traj_losses = []  # per-window L_traj (accumulated from GT at the window anchor)

            # Window starts; a final window is appended to cover the scan tail
            starts = list(range(0, n_frames - seq_frames + 1, stride_frames))
            if not starts or starts[-1] + seq_frames < n_frames:
                starts.append(max(0, n_frames - seq_frames))

            for start in starts:
                end = start + seq_frames
                fr = torch.tensor(all_frames[start:end], device=device).unsqueeze(0)
                fr = fr.reshape(1, seq_len, BLOCK_SIZE, fr.shape[-2], fr.shape[-1])

                if val_sensor:
                    se = torch.tensor(all_sensors[start:end], device=device).unsqueeze(0)
                    sw = torch.tensor(all_sw[start:end], device=device).unsqueeze(0)
                    wl = torch.tensor(all_wl[start:end], device=device).unsqueeze(0)
                    se = se.reshape(1, seq_len, BLOCK_SIZE, -1)
                    sw = sw.reshape(1, seq_len, BLOCK_SIZE, sw.shape[-2], sw.shape[-1])
                    wl = wl.reshape(1, seq_len, BLOCK_SIZE)
                    se_v = select_inst_channels(se, inst_indices)
                    sw_v = select_window_channels(sw, win_indices)
                else:
                    se_v, sw_v, wl = None, None, None

                out = model(fr, sensors=se_v, sensor_wins=sw_v)
                T_pred = sixd_to_matrix(out[0])

                anchor_idx = start + BLOCK_SIZE - 1  # last frame of block 0 in this window

                # Global accumulation: starts from the previous window's predicted pose (eps_acc)
                T_acc = pred_poses.get(anchor_idx, torch.tensor(all_tforms[anchor_idx], device=device))

                # Local accumulation: starts from the GT pose, matching the training L_traj
                T_init_local = torch.tensor(all_tforms[anchor_idx], device=device).to(torch.float32)
                T_acc_local = T_init_local.clone()
                local_traj_loss = 0.0

                for bi in range(seq_len - 1):
                    T_acc = T_acc @ T_pred[bi]
                    bfi = start + (bi + 2) * BLOCK_SIZE - 1  # last frame of block bi+1
                    if bfi < n_frames:
                        all_pred_corners[bfi] = (T_acc @ tform_calib @ pts_img)[:3, :].cpu()
                        pred_poses[bfi] = T_acc.clone()

                        T_acc_local = T_acc_local @ T_pred[bi]
                        tform_gt = torch.tensor(all_tforms[bfi], device=device).to(torch.float32)
                        corners_gt = tform_gt @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                        corners_pred = T_acc_local @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                        local_traj_loss += torch.nn.functional.mse_loss(corners_gt, corners_pred)

                    # L_aux: relative pose between consecutive block-representative frames
                    pi = start + bi * BLOCK_SIZE + (BLOCK_SIZE - 1)
                    ci = start + (bi + 2) * BLOCK_SIZE - 1
                    if ci < n_frames:
                        tform_prev = torch.tensor(all_tforms[pi], device=device)
                        tform_curr = torch.tensor(all_tforms[ci], device=device)
                        T_gt_rel = torch.linalg.inv(tform_prev.to(torch.float32)) @ tform_curr.to(torch.float32)
                        gt_pair = T_gt_rel @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                        pred_pair = T_pred[bi] @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                        aux_losses.append(torch.nn.functional.mse_loss(pred_pair, gt_pair))
                        aux_dists.append(torch.sqrt(((gt_pair - pred_pair) ** 2).sum(dim=0)).mean())

                        # Edge-length ratio term (logged only)
                        pred_acc = all_pred_corners[bfi]
                        gt_acc = gt_corners[ci]
                        eg = torch.stack([torch.norm(gt_acc[:, (j + 1) % 4] - gt_acc[:, j]) for j in range(4)])
                        ep = torch.stack([torch.norm(pred_acc[:, (j + 1) % 4] - pred_acc[:, j]) for j in range(4)])
                        rg = eg / eg[0].clamp(min=1e-6)
                        rp = ep / ep[0].clamp(min=1e-6)
                        edge_losses.append(torch.nn.functional.l1_loss(rp, rg))
                        
                # Mean L_traj over the blocks of this window
                if seq_len - 1 > 0:
                    window_traj_losses.append(local_traj_loss / (seq_len - 1))

            # eps_acc: accumulated corner error (mm) at block-representative frames
            block_indices = [i for i in range(BLOCK_SIZE - 1, n_frames, BLOCK_SIZE)]
            pred_stack = torch.stack([all_pred_corners[i] for i in block_indices])
            gt_stack = gt_corners[block_indices]

            scan_dist = torch.sqrt(((pred_stack - gt_stack) ** 2).sum(dim=1)).mean()
            total_dist += scan_dist

            # Validation L_traj is the mean of the per-window losses
            if window_traj_losses:
                scan_loss = torch.stack(window_traj_losses).mean()
            else:
                scan_loss = torch.tensor(0.0, device=device)
                
            total_loss += scan_loss
            
            if aux_losses:
                total_aux += torch.stack(aux_losses).mean()
            if aux_dists:
                total_aux_dist += torch.stack(aux_dists).mean()
            if edge_losses:
                total_edge += torch.stack(edge_losses).mean()

    model.train(True)
    if n_scans == 0:
        return 1e10, 1e10, 1e10, 1e10, 1e10, 1e10
    mean_traj = (total_loss / n_scans).item()
    mean_aux = (total_aux / max(n_scans, 1)).item()
    mean_edge = (total_edge / max(n_scans, 1)).item()
    mean_aux_dist = (total_aux_dist / max(n_scans, 1)).item()
    mean_dist = (total_dist / n_scans).item()
    mean_combined = (LOSS_TRAJECTORY_WEIGHT * mean_traj
                     + LOSS_AUX_WEIGHT * mean_aux
                     + LOSS_EDGE_WEIGHT * mean_edge)
    return mean_combined, mean_dist, mean_traj, mean_aux, mean_edge, mean_aux_dist

# ================================================================
#  Main
# ================================================================

def main():
    h5_train = resolve_split_h5('train')
    h5_valid = resolve_split_h5('valid')
    cnn_ch, inst_dim, win_indices, inst_indices, use_sensor = compute_sensor_channels()

    print(f"Device: {device}")
    print(f"Train HDF5: {h5_train}")
    print(f"Valid HDF5: {h5_valid}")
    print(f"Sequence mode: {SEQUENCE_MODE}, SEQ_LEN={SEQ_LEN}, Temporal={TEMPORAL_MODEL}")
    print(f"Sensor: ACCEL={USE_IMU_ACCEL}, QUAT={USE_IMU_QUAT}, FORCE={USE_FORCE}")
    if USE_AMP:
        print(f"AMP enabled. Batch={BATCH_SIZE}, GradAccum={GRADIENT_ACCUMULATION_STEPS}")

    # ---- Datasets ----
    dset_train = SSFrameDataset(h5_train, num_samples=BLOCK_SIZE, sample_range=BLOCK_SIZE,
                                sensor_in_use=use_sensor, sen_window_size=SEN_WINDOW_SIZE)
    dset_val = SSFrameDataset(h5_valid, num_samples=BLOCK_SIZE, sample_range=BLOCK_SIZE,
                              sensor_in_use=use_sensor, sen_window_size=SEN_WINDOW_SIZE)

    # Calibration matrix T_cal and image corner points (homogeneous, 4 x 4)
    tform_calib = torch.tensor(
        read_calib_matrices(FILENAME_CALIB, RESAMPLE_FACTOR), device=device)
    if ROTATE_TOOL_90:
        R_tr = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float64, device=device)
        Ri = torch.eye(4, dtype=torch.float64, device=device); Ri[:3, :3] = R_tr.T
        tform_calib = (Ri @ tform_calib.to(torch.float64)).to(torch.float32)
    pts_img = torch.tensor([[0,0,0,1],[dset_train.frame_size[1],0,0,1],
                            [dset_train.frame_size[1],dset_train.frame_size[0],0,1],
                            [0,dset_train.frame_size[0],0,1]], dtype=torch.float32, device=device).T

    # ---- Build model ----
    if SEQUENCE_MODE:
        base_model = efficientnet_b1(weights=None)
        base_model.features[0][0] = torch.nn.Conv2d(BLOCK_SIZE, base_model.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False)
        model = BlockSequenceModel(
            efficientnet_features=base_model.features, image_feature_dim=1280,
            temporal_model=TEMPORAL_MODEL, feat_dim=512,
            sensor_cnn_channels=cnn_ch, sensor_inst_dim=inst_dim,
        ).to(device)

        # Load pretrained backbone weights
        if PRETRAINED_WEIGHTS is not None:
            pretrained_path = os.path.join(os.getcwd(), PRETRAINED_WEIGHTS)
            if os.path.exists(pretrained_path):
                pretrained_state = torch.load(pretrained_path, map_location=device)
                model_dict = model.state_dict()
                matched = {k: v for k, v in pretrained_state.items()
                           if k in model_dict and k.startswith('features.')}
                model_dict.update(matched)
                model.load_state_dict(model_dict, strict=False)
                print(f"Loaded {len(matched)} backbone tensors from {os.path.basename(pretrained_path)}.")

        # Apply freezing
        weights_loaded = PRETRAINED_WEIGHTS and os.path.exists(
            os.path.join(os.getcwd(), PRETRAINED_WEIGHTS))
        fn = 9 if FREEZE_STAGES == -1 else FREEZE_STAGES
        fn = max(0, min(fn, 9))
        if fn > 0 and weights_loaded:
            for i, child in enumerate(model.features):
                if i < fn:
                    for p in child.parameters(): p.requires_grad = False
            fp = sum(p.numel() for i, c in enumerate(model.features)
                     if i < fn for p in c.parameters())
            print(f"Frozen {fn}/9 EfficientNet stages ({fp:,} params).")

        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"BlockSequenceModel: {tp:,} / {sum(p.numel() for p in model.parameters()):,} trainable params.")

        # Prediction transform (6D rotation + translation -> corner points)
        pred_transform = PredictionTransform6D(
            image_points=pts_img,
            tform_image_to_tool=tform_calib)

        # Sequence samples
        seq_frames = SEQ_LEN * BLOCK_SIZE
        train_samples = build_sequence_samples(dset_train, seq_frames, TRAIN_STRIDE)
        print(f"Training samples: {len(train_samples)} windows from {len(dset_train)} scans")
        dset_train_wrapped = SequenceDatasetWrapper(dset_train, train_samples, seq_frames)

        criterion = torch.nn.MSELoss()
        metrics = PointDistance()
        pair_weights = None  # no long-range weighting in sequence mode
    else:
        # Non-sequence mode is not supported
        model = SensorFusionModel(
            efficientnet_features=None, image_feature_dim=1280, out_dim=270,
            sensor_cnn_channels=cnn_ch, sensor_inst_dim=inst_dim * (20 if SEQUENCE_MODE else 20))
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        pred_transform = PredictionTransform("parameter", "point", num_pairs=45,
            image_points=None, tform_image_to_tool=tform_calib)
        dset_train_wrapped = dset_train
        criterion = torch.nn.MSELoss()
        metrics = PointDistance()
        pair_weights = None
        raise NotImplementedError("Legacy mode removed. Use SEQUENCE_MODE=True.")

    # ---- DataLoaders ----
    train_loader = torch.utils.data.DataLoader(
        dset_train_wrapped, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=12, pin_memory=True, drop_last=False,
        prefetch_factor=2, persistent_workers=True)

    # ---- Training loop ----
    val_loss_min, val_dist_min = 1e10, 1e10
    conv_epochs, conv_tl, conv_td, conv_vl, conv_vd = [], [], [], [], []
    val_epochs = []
    conv_vad = []   # val frame-to-frame (relative) distance in mm
    modality_history = {'epochs': [], 'image': [], 'accel': [], 'quat': [], 'force': []}
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    for epoch in range(NUM_EPOCHS):
        train_loss_sum, train_dist_sum = 0.0, 0.0
        train_traj_sum, train_aux_sum = 0.0, 0.0
        train_edge_sum, train_block_dist_sum = 0.0, 0.0
        optimiser.zero_grad()

        for step, batch in enumerate(train_loader):
            if use_sensor:
                frames, tforms, tforms_inv, sensors, sensor_wins, win_lens = batch
                frames, tforms, tforms_inv = frames.to(device, non_blocking=True), tforms.to(device, non_blocking=True), tforms_inv.to(device, non_blocking=True)
                sensors, sensor_wins, win_lens = sensors.to(device, non_blocking=True), sensor_wins.to(device, non_blocking=True), win_lens.to(device, non_blocking=True)

                # Reshape to (B, SEQ_LEN, BLOCK_SIZE, ...)
                B = frames.shape[0]
                H, W = frames.shape[-2], frames.shape[-1]
                frames = frames.reshape(B, SEQ_LEN, BLOCK_SIZE, H, W)
                sensors = sensors.reshape(B, SEQ_LEN, BLOCK_SIZE, -1)
                sensor_wins = sensor_wins.reshape(B, SEQ_LEN, BLOCK_SIZE, sensor_wins.shape[-2], sensor_wins.shape[-1])
                win_lens = win_lens.reshape(B, SEQ_LEN, BLOCK_SIZE)

                sw = select_window_channels(sensor_wins, win_indices)
                si = select_inst_channels(sensors, inst_indices)

                # Sensor dropout
                if SENSOR_DROPOUT > 0 and sw is not None:
                    mask = torch.rand(B, device=device) < SENSOR_DROPOUT
                    if mask.any():
                        sw = sw.clone(); sw[mask] = 0
            else:
                frames, tforms, tforms_inv = batch
                frames, tforms, tforms_inv = frames.to(device, non_blocking=True), tforms.to(device, non_blocking=True), tforms_inv.to(device, non_blocking=True)
                B = frames.shape[0]
                frames = frames.reshape(B, SEQ_LEN, BLOCK_SIZE, H, W)
                sw, si, win_lens = None, None, None

            # ---- Preview hook: save raw model inputs before forward pass ----
            _save_preview_images(frames, PREVIEW_DIR, PREVIEW_MAX_IMAGES)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(frames, sensors=si, sensor_wins=sw)  # (B, SEQ_LEN-1, 9)
                T_pred = sixd_to_matrix(out)                       # (B, SEQ_LEN-1, 4, 4)

                # ---- Trajectory accumulation loss L_traj ----
                # T_init = GT pose of block 0's last frame (index BLOCK_SIZE - 1)
                T_init = tforms[:, BLOCK_SIZE - 1, :, :].to(torch.float32)  # (B, 4, 4)
                T_acc = T_init.clone()                          # absolute, starts at block 0
                traj_loss = 0.0
                aux_loss = 0.0
                edge_loss = 0.0
                dist_acc = 0.0
                block_dist = 0.0

                for bi in range(SEQ_LEN - 1):
                    T_acc = T_acc @ T_pred[:, bi, :, :]

                    block_frame = (bi + 2) * BLOCK_SIZE - 1
                    tform_gt = tforms[:, block_frame, :, :]

                    corners_gt = tform_gt.to(torch.float32) @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                    corners_pred = T_acc @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                    traj_loss += torch.nn.functional.mse_loss(corners_gt, corners_pred)

                    # eps_acc: accumulated corner error in mm (no gradient)
                    with torch.no_grad():
                        dist_acc += torch.sqrt(((corners_gt - corners_pred) ** 2).sum(dim=1)).mean()

                    # Edge-length ratio term (logged only; LOSS_EDGE_WEIGHT = 0)
                    edges_gt = torch.stack([
                        torch.norm(corners_gt[..., 1] - corners_gt[..., 0], dim=1),
                        torch.norm(corners_gt[..., 2] - corners_gt[..., 1], dim=1),
                        torch.norm(corners_gt[..., 3] - corners_gt[..., 2], dim=1),
                        torch.norm(corners_gt[..., 0] - corners_gt[..., 3], dim=1),
                    ], dim=1)
                    edges_pred = torch.stack([
                        torch.norm(corners_pred[..., 1] - corners_pred[..., 0], dim=1),
                        torch.norm(corners_pred[..., 2] - corners_pred[..., 1], dim=1),
                        torch.norm(corners_pred[..., 3] - corners_pred[..., 2], dim=1),
                        torch.norm(corners_pred[..., 0] - corners_pred[..., 3], dim=1),
                    ], dim=1)
                    ratio_gt = edges_gt / edges_gt[:, 0:1].clamp(min=1e-6)
                    ratio_pred = edges_pred / edges_pred[:, 0:1].clamp(min=1e-6)
                    edge_loss += torch.nn.functional.l1_loss(ratio_pred, ratio_gt)

                    # L_aux: relative pose between consecutive block-representative frames
                    tform_prev = tforms[:, bi * BLOCK_SIZE + (BLOCK_SIZE - 1), :, :]
                    tform_curr = tforms[:, block_frame, :, :]
                    T_gt_rel = torch.linalg.inv(tform_prev.to(torch.float32)) @ tform_curr.to(torch.float32)
                    corners_gt_pair = T_gt_rel @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                    corners_pred_pair = T_pred[:, bi, :, :] @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                    aux_loss += torch.nn.functional.mse_loss(corners_gt_pair, corners_pred_pair)

                    # Per-block prediction error (eps_block) in mm
                    with torch.no_grad():
                        block_dist += torch.sqrt(((corners_gt_pair - corners_pred_pair) ** 2).sum(dim=1)).mean()

                traj_loss /= (SEQ_LEN - 1)
                aux_loss /= (SEQ_LEN - 1)
                edge_loss /= (SEQ_LEN - 1)
                dist_acc /= (SEQ_LEN - 1)
                block_dist /= (SEQ_LEN - 1)
                loss = (LOSS_TRAJECTORY_WEIGHT * traj_loss
                        + LOSS_AUX_WEIGHT * aux_loss
                        + LOSS_EDGE_WEIGHT * edge_loss)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    optimiser.step()
                optimiser.zero_grad()

            train_loss_sum += (traj_loss.item() + aux_loss.item() + edge_loss.item())
            train_traj_sum += traj_loss.item()
            train_aux_sum += aux_loss.item()
            train_edge_sum += edge_loss.item()
            train_dist_sum += dist_acc.item()
            train_block_dist_sum += block_dist.item()

        train_loss_sum /= (step + 1)
        train_dist_sum /= (step + 1)
        train_traj_sum /= (step + 1)
        train_aux_sum /= (step + 1)
        train_edge_sum /= (step + 1)
        train_block_dist_sum /= (step + 1)

        # Record training metrics every epoch
        conv_epochs.append(epoch)
        conv_tl.append(train_loss_sum)
        conv_td.append(train_dist_sum)

        if epoch in range(0, NUM_EPOCHS, FREQ_INFO):
            print('[Epoch %d] train-loss=%.4f, dist=%.4f mm, traj=%.4f, aux=%.4f'
                  % (epoch, train_loss_sum, train_dist_sum, train_traj_sum, train_aux_sum))

        # ---- Validation ----
        if epoch in range(0, NUM_EPOCHS, FREQ_SAVE):
            val_seq_loss, val_seq_dist, val_traj, val_aux, val_edge, val_aux_dist = validate_sequence(
                model, dset_val, SEQ_LEN, tform_calib, pts_img,
                win_indices, inst_indices, None, criterion, metrics)
            print('[Epoch %d] val-loss=%.4f, dist=%.4f mm, f2f=%.4f mm, traj=%.4f, aux=%.4f, edge=%.4f'
                  % (epoch, val_seq_loss, val_seq_dist, val_aux_dist, val_traj, val_aux, val_edge))

            val_epochs.append(epoch)
            conv_vl.append(val_seq_loss); conv_vd.append(val_seq_dist)
            conv_vad.append(val_aux_dist)

            save_model(model, epoch, NUM_EPOCHS, FREQ_SAVE, SAVE_PATH)
            prev_lm, prev_dm = val_loss_min, val_dist_min
            val_loss_min, val_dist_min = save_best_network(
                epoch, model, torch.tensor(val_seq_loss), torch.tensor(val_seq_dist),
                val_loss_min, val_dist_min, SAVE_PATH)
            if val_seq_loss < prev_lm or val_seq_dist < prev_dm:
                _write_config_log(epoch, train_loss_sum, train_dist_sum,
                                  train_traj_sum, train_aux_sum,
                                  val_seq_loss, val_seq_dist, val_traj, val_aux,
                                  val_loss_min, val_dist_min,
                                  (USE_IMU_ACCEL, USE_IMU_QUAT, USE_FORCE), SAVE_PATH)
            _plot_convergence(conv_epochs, conv_tl, conv_td,
                              val_epochs, conv_vl, conv_vd, conv_vad, SAVE_PATH)

            # Modality contribution (single val window for efficiency)
            if dset_val.sensor_in_use and len(dset_val) > 0:
                try:
                    with torch.no_grad():
                        i_s, i_sc = dset_val.indices_in_use[0]
                        nf = int(dset_val.num_frames[i_s, i_sc])
                        if nf >= seq_frames:
                            fv = dset_val._get_file()
                            fr0 = np.stack([fv['/sub{:03d}_scan{:02d}_frame{:04d}'.format(i_s, i_sc, fi)][:] for fi in range(seq_frames)])
                            fr0 = torch.tensor(fr0, device=device).unsqueeze(0)
                            fr0 = fr0.reshape(1, SEQ_LEN, BLOCK_SIZE, fr0.shape[-2], fr0.shape[-1])
                            se0 = torch.zeros(1, SEQ_LEN, BLOCK_SIZE, 14, device=device)
                            sw0 = torch.zeros(1, SEQ_LEN, BLOCK_SIZE, max(SEN_WINDOW_SIZE, 1), 13, device=device)
                            for j in range(seq_frames):
                                sk0 = '/sub{:03d}_scan{:02d}_sensor{:04d}'.format(i_s, i_sc, j)
                                if sk0 in fv: se0[0, j // BLOCK_SIZE, j % BLOCK_SIZE] = torch.tensor(fv[sk0][:])
                            si0 = select_inst_channels(se0, inst_indices)
                            sw0_v = select_window_channels(sw0, win_indices)
                            out_full = model(fr0, sensors=si0, sensor_wins=sw0_v)
                            nf0 = torch.norm(out_full).clamp(min=1e-8)
                            # Image contribution
                            fr_mean = fr0.mean(dim=(0, 2, 3), keepdim=True).expand_as(fr0)
                            out_nimg = model(fr_mean, sensors=si0, sensor_wins=sw0_v)
                            modality_history['image'].append(float(torch.norm(out_full - out_nimg) / nf0))
                            # Sensor modalities
                            if USE_IMU_ACCEL and si0 is not None:
                                si_a = si0.clone(); si_a[..., :3] = 0
                                sw_a = sw0_v.clone() if sw0_v is not None else None
                                if sw_a is not None: sw_a[..., :3] = 0
                                out_a = model(fr0, sensors=si_a, sensor_wins=sw_a)
                                modality_history['accel'].append(float(torch.norm(out_full - out_a) / nf0))
                            else:
                                modality_history['accel'].append(0.0)
                            if USE_IMU_QUAT and si0 is not None:
                                qs = 3 if USE_IMU_ACCEL else 0
                                si_q = si0.clone(); si_q[..., qs:qs+4] = 0
                                sw_q = sw0_v.clone() if sw0_v is not None else None
                                if sw_q is not None: sw_q[..., qs:qs+4] = 0
                                out_q = model(fr0, sensors=si_q, sensor_wins=sw_q)
                                modality_history['quat'].append(float(torch.norm(out_full - out_q) / nf0))
                            else:
                                modality_history['quat'].append(0.0)
                            if USE_FORCE and si0 is not None:
                                fs = (3 if USE_IMU_ACCEL else 0) + (4 if USE_IMU_QUAT else 0)
                                si_f = si0.clone(); si_f[..., fs:fs+5] = 0
                                sw_f = sw0_v.clone() if sw0_v is not None else None
                                if sw_f is not None: sw_f[..., fs:fs+5] = 0
                                out_f = model(fr0, sensors=si_f, sensor_wins=sw_f)
                                modality_history['force'].append(float(torch.norm(out_full - out_f) / nf0))
                            else:
                                modality_history['force'].append(0.0)
                            modality_history['epochs'].append(epoch)
                except Exception:
                    pass  # skip modality contribution if val data unavailable
            if len(modality_history['epochs']) > 1:
                _plot_modality_contribution(modality_history, SAVE_PATH)

        # ---- Per-epoch log ----
        train_metrics = dict(L_traj=train_traj_sum, L_aux=train_aux_sum, L_edge=train_edge_sum,
                             L=(LOSS_TRAJECTORY_WEIGHT * train_traj_sum
                                + LOSS_AUX_WEIGHT * train_aux_sum
                                + LOSS_EDGE_WEIGHT * train_edge_sum),
                             eps_acc_mm=train_dist_sum, eps_block_mm=train_block_dist_sum)
        if epoch in range(0, NUM_EPOCHS, FREQ_SAVE):
            # val L is val_seq_loss, the same value used for checkpoint selection
            val_metrics = dict(L_traj=val_traj, L_aux=val_aux, L_edge=val_edge,
                               L=val_seq_loss,
                               eps_acc_mm=val_seq_dist, eps_block_mm=val_aux_dist)
        else:
            val_metrics = None
        _write_epoch_log(epoch, train_metrics, val_metrics, SAVE_PATH)


if __name__ == '__main__':
    main()
