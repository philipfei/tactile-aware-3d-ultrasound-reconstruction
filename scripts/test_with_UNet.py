"""
Test the pose prediction network (BlockSequenceModel) with the U-Net bone segmentation.
Sliding-window inference, evaluation at block-representative frames, and
volumetric (DSC / Hausdorff) metrics of the reconstructed bone surface.
"""
import ssl

# Tolerate a corrupted Windows certificate store, which otherwise crashes SSL on import
_orig_load_win_certs = ssl.SSLContext._load_windows_store_certs
def _safe_load_win_certs(self, storename, purpose):
    try:
        _orig_load_win_certs(self, storename, purpose)
    except Exception:
        pass
ssl.SSLContext._load_windows_store_certs = _safe_load_win_certs

import os
import sys
import csv
import datetime
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import torch
from torchvision.models import efficientnet_b1
sys.path.append(os.getcwd())

from freehand.loader import SSFrameDataset
from freehand.network import BlockSequenceModel
from data.calib import read_calib_matrices
from freehand.transform import sixd_to_matrix
from freehand.utils import *
from unet.model import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESAMPLE_FACTOR = 4  # image downsampling factor of the HDF5 frames
FILENAME_CALIB = "data/calib_matrix.csv"  # calibration matrix T_cal (image to tool)
ROTATE_IMAGE_90 = False
ROTATE_TOOL_90  = False  # apply a 90 deg in-plane rotation to T_cal

# Must match train.py (used to locate the results folder and build the model)
USE_IMU_ACCEL = True
USE_IMU_QUAT  = False
USE_FORCE     = True
SEN_WINDOW_SIZE = 5   # sensor trend window length (samples)
SEQ_LEN = 5           # blocks per window
BLOCK_SIZE = 5        # frames per block
TEMPORAL_MODEL = "gru"
LEARNING_RATE = 1e-4

sensor_tag = ""
if USE_IMU_ACCEL: sensor_tag += "A"
if USE_IMU_QUAT:  sensor_tag += "Q"
if USE_FORCE:     sensor_tag += "F"
if not sensor_tag: sensor_tag = "ImageOnly"
sensor_tag += "_seq"

saved_results = 'seq_len' + str(SEQ_LEN) + '_blk' + str(BLOCK_SIZE) \
    + '__lr' + str(LEARNING_RATE) \
    + '__pred_parameter__label_point__sensor_' + sensor_tag
SAVE_PATH = os.path.join('results', saved_results)
os.makedirs(os.path.join(os.getcwd(), SAVE_PATH, 'plotting'), exist_ok=True)
FILENAME_WEIGHTS = "best_validation_loss_model"

# Plotting mode: "matplotlib" or "pyvista"
PLOT_MODE = "pyvista"

# U-Net bone segmentation overlay config
UNET_FEATURES = [16, 32, 64, 128]
UNET_MODEL_PATH = os.path.join('results', 'UNet', 'saved_model', 'best_model.pth')
BG_DISPLAY_INTERVAL = 5  # show the non-bone background only on every N-th frame
BG_OPACITY = 0       # opacity of non-bone pixels on those frames (0.0-1.0)

# --- 3D volumetric reconstruction metrics (DSC / Hausdorff) ---
VOXEL_SIZE_MM = 0.5        # isotropic voxel size (mm)
GRID_PADDING_MM = 5.0      # minimum grid margin (mm); grows to fit the predicted sweep
EXCLUDE_ANCHOR_BLOCK = True  # drop the GT-seeded block (frame BLOCK_SIZE-1) from the predicted volume
SAVE_VOLUME_GRIDS = True   # save the boolean GT/pred grids per scan as .npz (with origin, voxel size)
MAX_VOXELS = 400**3       # skip volumetric metrics above this grid size (memory guard)

# --- 3D rigid registration (ICP) before volumetric metrics ---
ENABLE_REGISTRATION = True   # align the predicted volume to GT before computing DSC/HD
ICP_MAX_DISTANCE_MM = 150.0  # correspondence threshold (mm); large enough to cover eps_drift
ICP_MAX_ITERATIONS = 50
ICP_MAX_POINTS = 100000      # max voxel-center points per cloud for ICP

# Scan indices to evaluate; None = all scans
selected_scans = None
# ==========================


def resolve_test_h5():
    data_dir = os.path.join(os.getcwd(), "data")
    test_h5 = os.path.join(data_dir, f'frames_res{RESAMPLE_FACTOR}_test.h5')
    flat_h5 = os.path.join(data_dir, f'frames_res{RESAMPLE_FACTOR}.h5')
    if os.path.exists(test_h5): return test_h5
    if os.path.exists(flat_h5):
        print(f"[Info] Test split HDF5 not found. Falling back to {flat_h5}.")
        return flat_h5
    raise FileNotFoundError(f"Neither {test_h5} nor {flat_h5} found.")


def compute_sensor_channels():
    win_indices, inst_indices = [], []
    if USE_IMU_ACCEL:
        win_indices.extend([0, 1, 2]); inst_indices.extend([0, 1, 2])
    if USE_IMU_QUAT:
        win_indices.extend([3, 4, 5, 6]); inst_indices.extend([3, 4, 5, 6])
    if USE_FORCE:
        win_indices.extend([7, 8, 9, 10, 11]); inst_indices.extend([7, 8, 9, 10, 11])
    inst_indices.append(12)  # dt (time step)
    # Time-position channel at index 12 of sensor_wins
    if SEN_WINDOW_SIZE > 0:
        win_indices.append(12)
    cnn_ch = len(win_indices) if SEN_WINDOW_SIZE > 0 else 0
    inst_dim = len(inst_indices)
    return cnn_ch, inst_dim, win_indices, inst_indices, cnn_ch > 0


def _plot_trajectory_with_lines(gt, pred, save_path):
    """Plot GT vs predicted trajectory: 4 corner lines connect consecutive frames."""
    n_frames = gt.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # GT: green corner points, 4 green lines connecting each corner across frames
    for ci in range(4):
        ax.plot(gt[:, 0, ci], gt[:, 1, ci], gt[:, 2, ci],
                color='green', linewidth=2, alpha=0.8)
        ax.scatter(gt[:, 0, ci], gt[:, 1, ci], gt[:, 2, ci],
                   c='green', s=12, alpha=0.6)

    # Pred: red corner points, 4 red lines connecting each corner across frames
    for ci in range(4):
        ax.plot(pred[:, 0, ci], pred[:, 1, ci], pred[:, 2, ci],
                color='red', linewidth=2, alpha=0.8)
        ax.scatter(pred[:, 0, ci], pred[:, 1, ci], pred[:, 2, ci],
                   c='red', s=12, alpha=0.6)

    # First frame GT rectangle (dark green)
    for e in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ax.plot([gt[0, 0, e[0]], gt[0, 0, e[1]]],
                [gt[0, 1, e[0]], gt[0, 1, e[1]]],
                [gt[0, 2, e[0]], gt[0, 2, e[1]]],
                'darkgreen', linewidth=3)

    # Last frame GT rectangle (dark green, dashed)
    for e in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ax.plot([gt[-1, 0, e[0]], gt[-1, 0, e[1]]],
                [gt[-1, 1, e[0]], gt[-1, 1, e[1]]],
                [gt[-1, 2, e[0]], gt[-1, 2, e[1]]],
                'darkgreen', linewidth=3, linestyle='--')

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(f'Trajectory: {n_frames} block frames (green=GT, red=Pred)')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=150)
    plt.close(fig)


def _plot_trajectory_pyvista(all_gt, all_frame_imgs, block_indices, pred_block, save_path,
                             masks=None, volumes=None):
    """
    PyVista visualization:
      - green textured image planes along the GT trajectory (subsampled to ~200)
      - red image planes and wireframes at the predicted block-representative frames
      - translucent volume meshes (GT green, predicted red, registered predicted blue)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  [SKIP] PyVista not installed. Install with: pip install pyvista")
        return

    n_all = all_gt.shape[0]
    has_masks = masks is not None and len(masks) == n_all

    def _grid_to_mesh(grid, origin, voxel):
        if not grid.any():
            return None
        data = pv.ImageData(dimensions=grid.shape, spacing=(voxel,) * 3,
                            origin=origin + voxel / 2.0)
        data['mask'] = grid.astype(np.uint8).flatten(order='F')
        return data.contour([0.5])

    def _build_rgba_texture(raw_fi, bone_color):
        img = all_frame_imgs[raw_fi]
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.dtype == np.float32 or img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        h, w = img.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img

        mask = masks[raw_fi]
        if mask.dtype != bool:
            mask = mask.astype(bool)

        rgba[mask, 0] = bone_color[0]
        rgba[mask, 1] = bone_color[1]
        rgba[mask, 2] = bone_color[2]
        rgba[mask, 3] = 255

        bg_alpha = int(BG_OPACITY * 255)
        if raw_fi % BG_DISPLAY_INTERVAL == 0:
            rgba[~mask, 3] = bg_alpha
        else:
            rgba[~mask, 3] = 0

        return np.ascontiguousarray(rgba[::-1, :, :])

    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.add_axes()

    gt_stride = max(1, n_all // 200)
    gt_indices = list(range(0, n_all, gt_stride))
    for fi in gt_indices:
        g = all_gt[fi].T
        if fi < len(all_frame_imgs):
            if has_masks:
                tex = pv.numpy_to_texture(_build_rgba_texture(fi, (0, 255, 0)))
            else:
                img = all_frame_imgs[fi]
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.dtype == np.float32 or img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img = np.ascontiguousarray(img)
                tex = pv.numpy_to_texture(img)
            x = np.array([[g[0, 0], g[1, 0]], [g[3, 0], g[2, 0]]])
            y = np.array([[g[0, 1], g[1, 1]], [g[3, 1], g[2, 1]]])
            z = np.array([[g[0, 2], g[1, 2]], [g[3, 2], g[2, 2]]])
            grid = pv.StructuredGrid(x, y, z)
            grid.texture_map_to_plane(inplace=True)
            plotter.add_mesh(grid, texture=tex, show_edges=False, opacity=1)

    gt_pts = all_gt[gt_indices]
    for ci in range(4):
        pts = gt_pts[:, :, ci]
        for fi in range(len(gt_indices) - 1):
            line = pv.Line(pts[fi], pts[fi + 1])
            plotter.add_mesh(line, color='green', line_width=4, opacity=1)

    n_pred = pred_block.shape[0]
    if has_masks:
        for fi in range(n_pred):
            raw_fi = block_indices[fi]
            p = pred_block[fi].T
            if raw_fi < len(all_frame_imgs):
                tex = pv.numpy_to_texture(_build_rgba_texture(raw_fi, (255, 0, 0)))
                x = np.array([[p[0, 0], p[1, 0]], [p[3, 0], p[2, 0]]])
                y = np.array([[p[0, 1], p[1, 1]], [p[3, 1], p[2, 1]]])
                z = np.array([[p[0, 2], p[1, 2]], [p[3, 2], p[2, 2]]])
                grid = pv.StructuredGrid(x, y, z)
                grid.texture_map_to_plane(inplace=True)
                plotter.add_mesh(grid, texture=tex, show_edges=False, opacity=1)

    for fi in range(n_pred):
        p = pred_block[fi].T
        for e in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            line = pv.Line(p[e[0]], p[e[1]])
            plotter.add_mesh(line, color='red', line_width=4)

    for ci in range(4):
        pts = pred_block[:, :, ci]
        for fi in range(n_pred - 1):
            line = pv.Line(pts[fi], pts[fi + 1])
            plotter.add_mesh(line, color='red', line_width=4, opacity=1)

    if volumes and volumes.get('gt_grid') is not None:
        voxel = volumes['voxel']
        origin = volumes['origin']
        mesh_gt = _grid_to_mesh(volumes['gt_grid'], origin, voxel)
        if mesh_gt is not None:
            plotter.add_mesh(mesh_gt, color='green', opacity=0.25, show_edges=False, name='gt_volume')
        mesh_pred = _grid_to_mesh(volumes['pred_grid'], origin, voxel)
        if mesh_pred is not None:
            plotter.add_mesh(mesh_pred, color='red', opacity=0.25, show_edges=False, name='pred_volume')
        pred_reg = volumes.get('pred_grid_reg')
        if pred_reg is not None:
            mesh_reg = _grid_to_mesh(pred_reg, origin, voxel)
            if mesh_reg is not None:
                plotter.add_mesh(mesh_reg, color='blue', opacity=0.25, show_edges=False,
                                 name='pred_volume_reg')

    plotter.export_html(save_path + '_pyvista.html')
    print(f"  PyVista HTML saved to {save_path}_pyvista.html")

    SHOW_GUI = bool(os.environ.get('DISPLAY'))
    if SHOW_GUI:
        plotter.show(title=f"Trajectory: {n_all} GT frames, {len(block_indices)} Pred")
    else:
        print("  [INFO] Headless mode — HTML export saved. Open in browser locally.")


def _pose_np(p):
    """Convert a pose (torch CUDA/CPU or ndarray) to np.float64 (4,4)."""
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    return np.asarray(p, dtype=np.float64)


def _plane_normal(pose_calib):
    """Image-plane (elevation) normal in the world frame: R @ [0,0,1], normalized."""
    n = pose_calib[:3, :3] @ np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def _voxelize_slab(pose_calib, mask, origin, voxel, shape, thickness, grid):
    # Mark the voxels of a mask extruded along the plane normal by `thickness` (mm)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return
    coords = np.vstack([cols, rows, np.zeros_like(cols, dtype=np.float64),
                        np.ones_like(cols, dtype=np.float64)])
    P = (pose_calib @ coords).T[:, :3]
    n = _plane_normal(pose_calib)
    offs = np.arange(-thickness / 2.0, thickness / 2.0 + voxel, voxel)
    if offs.size == 0:
        offs = np.array([0.0])
    pts = (P[None, :, :] + (offs[:, None] * n)[:, None, :]).reshape(-1, 3)
    idx = np.floor((pts - origin) / voxel + 1e-9).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    ix, iy, iz = idx[ok, 0], idx[ok, 1], idx[ok, 2]
    grid[ix, iy, iz] = True


def _voxel_centers(grid, origin, voxel, max_points):
    idx = np.argwhere(grid)
    if idx.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    if idx.shape[0] > max_points:
        step = idx.shape[0] // max_points
        idx = idx[::step]
    return (idx.astype(np.float64) + 0.5) * voxel + origin


def _kabsch(P, Q):
    # Least-squares rigid transform (4x4) mapping points P onto Q
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cq - R @ cp
    return T


def _rigid_icp_scipy(pred_pts, gt_pts, max_dist, max_iter, init=None, tol=1e-6):
    T = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).copy()
    tree = cKDTree(gt_pts)
    prev_err = float('inf')
    src_h = np.concatenate([pred_pts, np.ones((pred_pts.shape[0], 1))], axis=1)
    for _ in range(max_iter):
        p = (T @ src_h.T).T[:, :3]
        d, nn = tree.query(p, k=1)
        ok = d < max_dist
        if ok.sum() < 50:
            break
        T = _kabsch(p[ok], gt_pts[nn[ok]]) @ T
        err = float(d[ok].mean())
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return T


def _rigid_icp_open3d(pred_pts, gt_pts, max_dist, max_iter, init=None):
    import open3d as o3d
    pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pts))
    pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    init = np.eye(4) if init is None else np.asarray(init, dtype=np.float64)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt, max_correspondence_distance=max_dist, init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return reg.transformation


def _find_registration(pred_grid, gt_grid, origin, voxel, init=None):
    pred_pts = _voxel_centers(pred_grid, origin, voxel, ICP_MAX_POINTS)
    gt_pts = _voxel_centers(gt_grid, origin, voxel, ICP_MAX_POINTS)
    if pred_pts.shape[0] < 50 or gt_pts.shape[0] < 50:
        return None, None
    try:
        return _rigid_icp_open3d(pred_pts, gt_pts, ICP_MAX_DISTANCE_MM, ICP_MAX_ITERATIONS, init), 'open3d'
    except Exception:
        return _rigid_icp_scipy(pred_pts, gt_pts, ICP_MAX_DISTANCE_MM, ICP_MAX_ITERATIONS, init), 'scipy'


def _compute_volumetric_metrics(all_tf, all_masks, gt_corners, all_pred_corners, pred_poses,
                                block_indices, n_frames, frame_w, frame_h, tform_calib_np):
    # Voxelize GT (every frame) and predicted (block-representative frames) bone masks,
    # then compute DSC and Hausdorff distance, before and after rigid registration.
    nan_vol = {'gt_grid': None, 'pred_grid': None, 'origin': None, 'voxel': VOXEL_SIZE_MM,
               'dsc': float('nan'), 'hd': float('inf'),
               'dsc_reg': float('nan'), 'hd_reg': float('inf'),
               'T_reg': None, 'method': None, 'pred_grid_reg': None}

    center_h = np.array([frame_w / 2.0, frame_h / 2.0, 0.0, 1.0], dtype=np.float64)
    center_world = np.empty((n_frames, 3), dtype=np.float64)
    for fi in range(n_frames):
        center_world[fi] = (all_tf[fi].astype(np.float64) @ tform_calib_np @ center_h)[:3]

    included_blocks = [bi for bi in block_indices
                       if bi in pred_poses and not (EXCLUDE_ANCHOR_BLOCK and bi == BLOCK_SIZE - 1)]
    if not included_blocks:
        return nan_vol

    first_blk, last_blk = min(included_blocks), max(included_blocks)
    gt_end = min(n_frames, last_blk + BLOCK_SIZE)
    gt_frames = range(first_blk, gt_end)

    corners_gt = gt_corners[first_blk:gt_end].numpy().transpose(0, 2, 1).reshape(-1, 3)
    corners_pred = torch.stack([all_pred_corners[bi] for bi in included_blocks]).numpy() \
        .transpose(0, 2, 1).reshape(-1, 3)
    lo = np.minimum(corners_gt.min(axis=0), corners_pred.min(axis=0))
    hi = np.maximum(corners_gt.max(axis=0), corners_pred.max(axis=0))

    # Grid margin covers half the largest block-to-block gap (slab thickness)
    max_gap = 0.0
    for bi in included_blocks:
        prev_idx = max(0, bi - BLOCK_SIZE)
        max_gap = max(max_gap, np.linalg.norm(center_world[bi] - center_world[prev_idx]))

    pad = max(GRID_PADDING_MM, max_gap / 2.0 + VOXEL_SIZE_MM)
    origin = lo - pad
    shape = np.ceil((hi + pad - origin) / VOXEL_SIZE_MM).astype(np.int64)

    if np.prod(shape) > MAX_VOXELS:
        print(f"  [WARN] Scan: volume grid {tuple(shape)} exceeds MAX_VOXELS={MAX_VOXELS}, "
              f"skipping volumetric metrics.")
        return nan_vol

    gt_grid = np.zeros(shape, dtype=bool)
    pred_grid = np.zeros(shape, dtype=bool)

    for fi in gt_frames:
        if not all_masks[fi].any():
            continue
        if fi == 0:
            th = VOXEL_SIZE_MM
        else:
            th = max(np.linalg.norm(center_world[fi] - center_world[fi - 1]), VOXEL_SIZE_MM)
        pose_calib = all_tf[fi].astype(np.float64) @ tform_calib_np
        _voxelize_slab(pose_calib, all_masks[fi], origin, VOXEL_SIZE_MM, shape, th, gt_grid)

    for bi in included_blocks:
        if not all_masks[bi].any():
            continue
        prev_idx = max(0, bi - BLOCK_SIZE)
        th = max(np.linalg.norm(center_world[bi] - center_world[prev_idx]), VOXEL_SIZE_MM)
        pose_calib = _pose_np(pred_poses[bi]) @ tform_calib_np
        _voxelize_slab(pose_calib, all_masks[bi], origin, VOXEL_SIZE_MM, shape, th, pred_grid)

    v_gt, v_pred = gt_grid.sum(), pred_grid.sum()
    if v_gt == 0 or v_pred == 0:
        dsc, hd = float('nan'), float('inf')
    else:
        dsc = 2.0 * (gt_grid & pred_grid).sum() / (v_gt + v_pred)
        d_to_pred = distance_transform_edt(~pred_grid, sampling=VOXEL_SIZE_MM)
        d_to_gt = distance_transform_edt(~gt_grid, sampling=VOXEL_SIZE_MM)
        hd = max(d_to_pred[gt_grid].max(), d_to_gt[pred_grid].max())

    dsc_reg, hd_reg, T_reg, method, pred_grid_reg = float('nan'), float('inf'), None, None, None
    if ENABLE_REGISTRATION and v_gt > 0 and v_pred > 0:
        # Initialize ICP with a Kabsch fit of the predicted to GT corner points
        pred_traj_pts = np.concatenate(
            [np.asarray(all_pred_corners[bi].cpu().numpy().T, dtype=np.float64)
             for bi in included_blocks], axis=0)
        gt_traj_pts = np.concatenate(
            [np.asarray(gt_corners[bi].cpu().numpy().T, dtype=np.float64)
             for bi in included_blocks], axis=0)
        T_traj = _kabsch(pred_traj_pts, gt_traj_pts) if pred_traj_pts.shape[0] >= 3 else np.eye(4)
        T_reg, method = _find_registration(pred_grid, gt_grid, origin, VOXEL_SIZE_MM, init=T_traj)
        if T_reg is not None:
            pred_grid_reg = np.zeros(shape, dtype=bool)
            for bi in included_blocks:
                if not all_masks[bi].any():
                    continue
                prev_idx = max(0, bi - BLOCK_SIZE)
                th = max(np.linalg.norm(center_world[bi] - center_world[prev_idx]), VOXEL_SIZE_MM)
                pose_calib = T_reg @ _pose_np(pred_poses[bi]) @ tform_calib_np
                _voxelize_slab(pose_calib, all_masks[bi], origin, VOXEL_SIZE_MM, shape, th, pred_grid_reg)
            v_pred_reg = pred_grid_reg.sum()
            if v_pred_reg > 0:
                dsc_reg = 2.0 * (gt_grid & pred_grid_reg).sum() / (v_gt + v_pred_reg)
                d_to_pred = distance_transform_edt(~pred_grid_reg, sampling=VOXEL_SIZE_MM)
                d_to_gt = distance_transform_edt(~gt_grid, sampling=VOXEL_SIZE_MM)
                hd_reg = max(d_to_pred[gt_grid].max(), d_to_gt[pred_grid_reg].max())

    return {'gt_grid': gt_grid, 'pred_grid': pred_grid, 'origin': origin, 'voxel': VOXEL_SIZE_MM,
            'dsc': dsc, 'hd': hd, 'dsc_reg': dsc_reg, 'hd_reg': hd_reg,
            'T_reg': T_reg, 'method': method, 'pred_grid_reg': pred_grid_reg}


def main():
    test_h5 = resolve_test_h5()
    cnn_ch, inst_dim, win_indices, inst_indices, use_sensor = compute_sensor_channels()

    dset_test = SSFrameDataset(test_h5, num_samples=BLOCK_SIZE, sample_range=BLOCK_SIZE,
                               sensor_in_use=use_sensor, sen_window_size=SEN_WINDOW_SIZE)
    print(f"Test scans: {len(dset_test)}, frame size: {dset_test.frame_size}")

    tform_calib = torch.tensor(
        read_calib_matrices(FILENAME_CALIB, RESAMPLE_FACTOR), device=device)
    if ROTATE_TOOL_90:
        R_tr = torch.tensor([[0,1,0],[-1,0,0],[0,0,1]], dtype=torch.float64, device=device)
        Ri = torch.eye(4, dtype=torch.float64, device=device); Ri[:3,:3] = R_tr.T
        tform_calib = (Ri @ tform_calib.to(torch.float64)).to(torch.float32)
    pts_img = torch.tensor([[0,0,0,1],[dset_test.frame_size[1],0,0,1],
                            [dset_test.frame_size[1],dset_test.frame_size[0],0,1],
                            [0,dset_test.frame_size[0],0,1]], dtype=torch.float32, device=device).T
    tform_calib_np = _pose_np(tform_calib)

    base_model = efficientnet_b1(weights=None)
    base_model.features[0][0] = torch.nn.Conv2d(BLOCK_SIZE, base_model.features[0][0].out_channels,
        kernel_size=3, stride=2, padding=1, bias=False)
    model = BlockSequenceModel(
        efficientnet_features=base_model.features, image_feature_dim=1280,
        temporal_model=TEMPORAL_MODEL, feat_dim=512,
        sensor_cnn_channels=cnn_ch, sensor_inst_dim=inst_dim,
    ).to(device)

    weights_path = os.path.join(SAVE_PATH, 'saved_model', FILENAME_WEIGHTS)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    model.train(False)
    print(f"Loaded weights from: {weights_path}")

    unet_model = UNet(in_channels=1, out_channels=1, features=UNET_FEATURES).to(device)
    unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
    unet_model.eval()
    print(f"Loaded UNet from: {UNET_MODEL_PATH}")

    seq_frames = SEQ_LEN * BLOCK_SIZE  # frames per window
    stride_frames = BLOCK_SIZE         # windows advance by one block
    all_scan_errors, all_scan_drifts, all_scan_block_errors = [], [], []
    all_scan_traj_len = []
    all_scan_dsc, all_scan_hd = [], []
    all_scan_dsc_reg, all_scan_hd_reg = [], []
    evaluated_scan_indices = []

    for i_scan in range(len(dset_test)):
        if selected_scans is not None and i_scan not in selected_scans:
            continue
        i_sub, i_scan_idx = dset_test.indices_in_use[i_scan]
        n_frames = int(dset_test.num_frames[i_sub, i_scan_idx])
        if n_frames < seq_frames:
            print(f"Scan {i_scan}: too short ({n_frames} < {seq_frames}), skipping.")
            continue

        scan_block_errors = []

        f = dset_test._get_file()
        all_fr, all_tf = [], []
        all_se, all_sw_data = [], []
        for ifr in range(n_frames):
            all_fr.append(f['/sub{:03d}_scan{:02d}_frame{:04d}'.format(i_sub, i_scan_idx, ifr)][:])
            all_tf.append(f['/sub{:03d}_scan{:02d}_tform{:04d}'.format(i_sub, i_scan_idx, ifr)][:])
            if use_sensor:
                all_se.append(f['/sub{:03d}_scan{:02d}_sensor{:04d}'.format(i_sub, i_scan_idx, ifr)][:])

        all_fr = np.stack(all_fr); all_tf = np.stack(all_tf)

        # Bone masks for all frames from the U-Net (threshold 0.5)
        all_fr_tensor = torch.from_numpy(all_fr).unsqueeze(1).to(torch.float32)
        all_masks = []
        unet_batch_size = 32
        for batch_start in range(0, n_frames, unet_batch_size):
            batch_end = min(batch_start + unet_batch_size, n_frames)
            batch = all_fr_tensor[batch_start:batch_end].to(device)
            with torch.no_grad():
                logits = unet_model(batch)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            for j in range(probs.shape[0]):
                all_masks.append(probs[j] > 0.5)

        if use_sensor:
            all_se_np = np.stack(all_se)
            ws = dset_test.sen_window_size
            sk = (i_sub, i_scan_idx)
            ss = dset_test._sensor_seqs.get(sk)
            sm = dset_test._sensor_maps.get(sk)
            raw_ch = ss.shape[1] if ss is not None else 12
            all_sw_data_np = np.zeros((n_frames, max(ws, 1), raw_ch + 1), dtype=np.float32)
            tp = np.linspace(0, 1, ws, dtype=np.float32) if ws > 1 else np.array([0], dtype=np.float32)
            all_sw_data_np[:, :, -1] = tp[np.newaxis, :]
            if ws > 0 and ss is not None and sm is not None:
                for j in range(n_frames):
                    cr = int(sm[j]); st = max(0, cr - ws + 1)
                    w = ss[st:cr + 1]; al = w.shape[0]
                    if al >= ws: all_sw_data_np[j, :, :raw_ch] = w[-ws:]
                    elif al > 0:
                        all_sw_data_np[j, -al:, :raw_ch] = w
                        p = ws - al; all_sw_data_np[j, :p, :raw_ch] = np.tile(w[0:1], (p, 1))

        gt_corners = []
        for ifr in range(n_frames):
            t = torch.tensor(all_tf[ifr], device=device).unsqueeze(0)
            gt_corners.append((t @ tform_calib @ pts_img)[0, :3, :].cpu())
        gt_corners = torch.stack(gt_corners)

        centers = gt_corners.mean(dim=2)
        traj_len = torch.norm(centers[1:] - centers[:-1], dim=1).sum().item()
        all_scan_traj_len.append(traj_len)
        evaluated_scan_indices.append(i_scan)

        # Seed: the last frame of block 0 starts from the GT pose
        all_pred_corners = [gt_corners[0].clone() for _ in range(n_frames)]
        all_pred_corners[BLOCK_SIZE - 1] = gt_corners[BLOCK_SIZE - 1].clone()
        pred_poses = {}
        pred_poses[BLOCK_SIZE - 1] = torch.tensor(all_tf[BLOCK_SIZE - 1], device=device)

        # Window starts; a final window is appended to cover the scan tail
        starts = list(range(0, n_frames - seq_frames + 1, stride_frames))
        if not starts or starts[-1] + seq_frames < n_frames:
            starts.append(max(0, n_frames - seq_frames))

        for start in starts:
            end = start + seq_frames
            fr = torch.tensor(all_fr[start:end], device=device).unsqueeze(0)
            fr = fr.reshape(1, SEQ_LEN, BLOCK_SIZE, fr.shape[-2], fr.shape[-1])

            if use_sensor:
                se = torch.tensor(all_se_np[start:end], device=device).unsqueeze(0)
                sw = torch.tensor(all_sw_data_np[start:end], device=device).unsqueeze(0)
                se = se.reshape(1, SEQ_LEN, BLOCK_SIZE, -1)
                sw = sw.reshape(1, SEQ_LEN, BLOCK_SIZE, sw.shape[-2], sw.shape[-1])

                wix = [i for i in win_indices if i < sw.shape[-1]]
                iix = [i for i in inst_indices if i < se.shape[-1]]
                sw_v = sw[..., wix] if wix else None
                se_v = se[..., iix] if iix else None
            else:
                sw_v, se_v = None, None

            with torch.no_grad():
                out = model(fr, sensors=se_v, sensor_wins=sw_v)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"  [WARN] NaN/Inf in model output at start={start}, skipping window.")
                continue
            T_pred = sixd_to_matrix(out[0])
            if torch.isnan(T_pred).any() or torch.isinf(T_pred).any():
                print(f"  [WARN] NaN/Inf in T_pred at start={start}, skipping window.")
                continue

            # Accumulate from the pose of this window's anchor (last frame of block 0)
            anchor_idx = start + BLOCK_SIZE - 1
            T_acc = pred_poses.get(anchor_idx,
                torch.tensor(all_tf[anchor_idx], device=device))
            for bi in range(SEQ_LEN - 1):
                T_acc = T_acc @ T_pred[bi]
                bfi = start + (bi + 2) * BLOCK_SIZE - 1
                if bfi < n_frames:
                    all_pred_corners[bfi] = (T_acc @ tform_calib @ pts_img)[:3, :].cpu()
                    pred_poses[bfi] = T_acc.clone()

                # eps_block: relative pose error between consecutive block-representative frames
                with torch.no_grad():
                    pi = start + bi * BLOCK_SIZE + (BLOCK_SIZE - 1)
                    ci = bfi
                    if ci < n_frames:
                        T_gt_rel = (torch.linalg.inv(torch.tensor(all_tf[pi], device=device).to(torch.float32))
                                    @ torch.tensor(all_tf[ci], device=device).to(torch.float32))
                        corners_gt_pair = T_gt_rel @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                        corners_pred_pair = T_pred[bi] @ tform_calib.to(torch.float32) @ pts_img.to(torch.float32)
                        pair_err = torch.sqrt(((corners_gt_pair - corners_pred_pair) ** 2).sum(dim=0)).mean()
                        scan_block_errors.append(pair_err.item())

        # eps_acc per block-representative frame; eps_drift is the error at the last one
        block_indices = [i for i in range(BLOCK_SIZE - 1, n_frames, BLOCK_SIZE)]
        pred_stack = torch.stack([all_pred_corners[i] for i in block_indices])
        gt_stack = gt_corners[block_indices]
        frame_errors = torch.sqrt(((pred_stack - gt_stack) ** 2).sum(dim=1)).mean(dim=1).numpy()
        all_scan_errors.append(frame_errors)
        final_drift = float(frame_errors[-1])
        all_scan_drifts.append(final_drift)
        all_scan_block_errors.append(np.array(scan_block_errors))
        print(f"Scan {i_scan}: {len(block_indices)} block frames, traj_len={traj_len:.1f} mm, "
              f"eps_acc={frame_errors.mean():.3f} mm (std={frame_errors.std():.3f}), "
              f"eps_drift={final_drift:.3f} mm, eps_block={np.mean(scan_block_errors):.3f} mm")

        frame_w = float(dset_test.frame_size[1]); frame_h = float(dset_test.frame_size[0])
        vol = _compute_volumetric_metrics(all_tf, all_masks, gt_corners, all_pred_corners,
                                          pred_poses, block_indices, n_frames,
                                          frame_w, frame_h, tform_calib_np)
        all_scan_dsc.append(vol['dsc'])
        all_scan_hd.append(vol['hd'])
        all_scan_dsc_reg.append(vol['dsc_reg'])
        all_scan_hd_reg.append(vol['hd_reg'])
        v_gt = int(vol['gt_grid'].sum()) if vol['gt_grid'] is not None else 0
        v_pred = int(vol['pred_grid'].sum()) if vol['pred_grid'] is not None else 0
        print(f"Scan {i_scan}: DSC={vol['dsc']:.4f}, HD={vol['hd']:.3f} mm | "
              f"reg DSC={vol['dsc_reg']:.4f}, HD={vol['hd_reg']:.3f} mm "
              f"[{vol['method'] or 'raw-only'}] (GTvox={v_gt}, PREDvox={v_pred})")
        if SAVE_VOLUME_GRIDS and vol['gt_grid'] is not None:
            _vol_out = {'gt': vol['gt_grid'], 'pred': vol['pred_grid'],
                        'origin': vol['origin'], 'voxel': vol['voxel']}
            if vol['pred_grid_reg'] is not None:
                _vol_out['pred_reg'] = vol['pred_grid_reg']
            np.savez(os.path.join(SAVE_PATH, 'plotting', f'volumes_scan_{i_scan}.npz'),
                     **_vol_out)

        gt_np = gt_stack.numpy(); pred_np = pred_stack.numpy()
        plot_path = os.path.join(SAVE_PATH, 'plotting', f'test_scan_{i_scan}')

        if PLOT_MODE == "pyvista":
            _plot_trajectory_pyvista(gt_corners.numpy(), all_fr, block_indices, pred_np, plot_path,
                                     masks=all_masks, volumes=vol)
        else:
            _plot_trajectory_with_lines(gt_np, pred_np, plot_path)
        print(f"Scan {i_scan}: trajectory plot saved.")

        coord_path = os.path.join(SAVE_PATH, 'plotting', f'test_scan_{i_scan}_coords.txt')
        with open(coord_path, 'w') as f:
            f.write(f"Scan {i_scan} — Block-representative frame coordinates\n")
            f.write(f"Frames: {len(block_indices)} (1 per {BLOCK_SIZE} raw frames)\n")
            f.write(f"Format: frame_idx  GT_corner0(xyz) ... GT_corner3(xyz)  Pred_corner0(xyz) ...\n\n")
            for fi, bi in enumerate(block_indices):
                f.write(f"Frame {bi} (block idx {fi}):\n")
                for ci in range(4):
                    f.write(f"  GT_c{ci}: ({gt_np[fi,0,ci]:.3f}, {gt_np[fi,1,ci]:.3f}, {gt_np[fi,2,ci]:.3f})    "
                            f"Pred_c{ci}: ({pred_np[fi,0,ci]:.3f}, {pred_np[fi,1,ci]:.3f}, {pred_np[fi,2,ci]:.3f})\n")
                f.write("\n")
        print(f"Scan {i_scan}: coordinates saved to {coord_path}.")

        motion_dir = os.path.join(SAVE_PATH, 'plotting', 'motion_analysis')
        os.makedirs(motion_dir, exist_ok=True)

        gt_centers = gt_np.mean(axis=2)
        pred_centers = pred_np.mean(axis=2)

        gt_vel = np.diff(gt_centers, axis=0, prepend=gt_centers[:1])
        pred_vel = np.diff(pred_centers, axis=0, prepend=pred_centers[:1])

        csv_path = os.path.join(motion_dir, f'test_scan_{i_scan}_motion_data.csv')
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['block_frame_idx',
                             'gt_x', 'gt_y', 'gt_z', 'pred_x', 'pred_y', 'pred_z',
                             'gt_vx', 'gt_vy', 'gt_vz', 'pred_vx', 'pred_vy', 'pred_vz'])
            for row_i, bf_idx in enumerate(block_indices):
                writer.writerow([bf_idx,
                                 f'{gt_centers[row_i, 0]:.6f}', f'{gt_centers[row_i, 1]:.6f}', f'{gt_centers[row_i, 2]:.6f}',
                                 f'{pred_centers[row_i, 0]:.6f}', f'{pred_centers[row_i, 1]:.6f}', f'{pred_centers[row_i, 2]:.6f}',
                                 f'{gt_vel[row_i, 0]:.6f}', f'{gt_vel[row_i, 1]:.6f}', f'{gt_vel[row_i, 2]:.6f}',
                                 f'{pred_vel[row_i, 0]:.6f}', f'{pred_vel[row_i, 1]:.6f}', f'{pred_vel[row_i, 2]:.6f}'])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        labels = ['X', 'Y', 'Z']
        for col, lbl in enumerate(labels):
            ax_pos = axes[0, col]
            ax_pos.plot(block_indices, gt_centers[:, col], color='green', linewidth=2, label='GT')
            ax_pos.plot(block_indices, pred_centers[:, col], color='red', linewidth=2, label='Pred')
            ax_pos.set_xlabel('Block Frame Index')
            ax_pos.set_ylabel('Position (mm)')
            ax_pos.set_title(f'{lbl} Position')
            ax_pos.legend()
            ax_pos.grid(True, alpha=0.3)

            ax_vel = axes[1, col]
            ax_vel.plot(block_indices, gt_vel[:, col], color='green', linewidth=2, label='GT')
            ax_vel.plot(block_indices, pred_vel[:, col], color='red', linewidth=2, label='Pred')
            ax_vel.set_xlabel('Block Frame Index')
            ax_vel.set_ylabel('Velocity (mm/step)')
            ax_vel.set_title(f'{lbl} Velocity')
            ax_vel.legend()
            ax_vel.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(motion_dir, f'test_scan_{i_scan}_motion_analysis.png')
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Scan {i_scan}: motion analysis saved to {csv_path} and {fig_path}.")

    # ---- Write result.txt ----
    all_errors = np.concatenate(all_scan_errors).astype(np.float64)
    all_drifts = np.array(all_scan_drifts)
    all_blocks = np.concatenate(all_scan_block_errors).astype(np.float64)
    all_traj = np.array(all_scan_traj_len)
    all_dsc = np.array(all_scan_dsc)
    all_hd = np.array(all_scan_hd)
    all_dsc_reg = np.array(all_scan_dsc_reg)
    all_hd_reg = np.array(all_scan_hd_reg)
    valid_vol = np.isfinite(all_dsc) & np.isfinite(all_hd)
    valid_reg = np.isfinite(all_dsc_reg) & np.isfinite(all_hd_reg)
    result_path = os.path.join(os.getcwd(), SAVE_PATH, 'result.txt')
    with open(result_path, 'w') as f:
        f.write(f"Test Results (Sequence Mode) — {datetime.datetime.now().isoformat()}\n")
        f.write(f"========================================\n")
        f.write(f"NOTE: Evaluation based on block representative frames\n")
        f.write(f"      (1 pose per {BLOCK_SIZE} frames); trajectory resolution is {BLOCK_SIZE}x lower than raw frame rate.\n")
        f.write(f"Model: SEQ_LEN={SEQ_LEN}, Temporal={TEMPORAL_MODEL}, BLOCK_SIZE={BLOCK_SIZE}\n")
        f.write(f"Scans evaluated: {len(all_scan_errors)}\n")
        f.write(f"Total eval frames: {len(all_errors)}\n")
        f.write(f"Total trajectory length: {all_traj.sum():.1f} mm "
                f"({all_traj.mean():.1f} mm/scan)\n\n")
        f.write(f"--- Overall Metrics ---\n")
        f.write(f"epsilon_block  (per-pair prediction accuracy):\n")
        f.write(f"  Mean:  {all_blocks.mean():.4f} mm\n")
        f.write(f"  Std:   {all_blocks.std():.4f} mm\n")
        f.write(f"  Median:{np.median(all_blocks):.4f} mm\n")
        f.write(f"  Max:   {all_blocks.max():.4f} mm\n\n")
        f.write(f"epsilon_acc    (accumulated tracking error):\n")
        f.write(f"  Mean:  {all_errors.mean():.4f} mm\n")
        f.write(f"  Std:   {all_errors.std():.4f} mm\n")
        f.write(f"  Median:{np.median(all_errors):.4f} mm\n")
        f.write(f"  Max:   {all_errors.max():.4f} mm\n\n")
        f.write(f"epsilon_drift  (final frame drift):\n")
        f.write(f"  Mean:  {all_drifts.mean():.4f} mm\n")
        f.write(f"  Std:   {all_drifts.std():.4f} mm\n")
        f.write(f"  Max:   {all_drifts.max():.4f} mm\n\n")
        f.write(f"--- Volumetric Metrics (3D bone reconstruction) ---\n")
        f.write(f"Voxel size: {VOXEL_SIZE_MM} mm (isotropic); GT=per-frame, Pred=sparse block extrusion\n")
        f.write(f"Registration: rigid ICP align Pred to GT, max_distance={ICP_MAX_DISTANCE_MM} mm, "
                f"enabled={ENABLE_REGISTRATION}\n")
        if valid_vol.sum() == 0:
            f.write(f"  No valid volumetric metrics (all scans empty or OOM-skipped).\n\n")
        else:
            f.write(f"  Scans with valid volumes: {valid_vol.sum()}/{len(all_dsc)}\n")
            f.write(f"  DSC (Dice Similarity):   Mean={all_dsc[valid_vol].mean():.4f}  "
                    f"Std={all_dsc[valid_vol].std():.4f}\n")
            f.write(f"  HD  (Hausdorff, mm):     Mean={all_hd[valid_vol].mean():.3f}  "
                    f"Std={all_hd[valid_vol].std():.3f}\n")
        if ENABLE_REGISTRATION:
            if valid_reg.sum() == 0:
                f.write(f"  No valid registered metrics (ICP failed or all scans skipped).\n\n")
            else:
                f.write(f"  Registered DSC (Dice):  Mean={all_dsc_reg[valid_reg].mean():.4f}  "
                        f"Std={all_dsc_reg[valid_reg].std():.4f}\n")
                f.write(f"  Registered HD (mm):     Mean={all_hd_reg[valid_reg].mean():.3f}  "
                        f"Std={all_hd_reg[valid_reg].std():.3f}\n\n")
        f.write(f"--- Per-Scan Breakdown ---\n")
        for i, (errs, drift, blk, trl) in enumerate(zip(all_scan_errors, all_drifts, all_scan_block_errors, all_scan_traj_len)):
            f.write(f"  Scan {evaluated_scan_indices[i]}: traj_len={trl:.1f} mm, eps_acc={errs.mean():.4f} +/- {errs.std():.4f} mm, "
                    f"eps_drift={drift:.4f} mm, eps_block={blk.mean():.4f} +/- {blk.std():.4f} mm, "
                    f"DSC={all_dsc[i]:.4f}, HD={all_hd[i]:.3f} mm, "
                    f"DSC_reg={all_dsc_reg[i]:.4f}, HD_reg={all_hd_reg[i]:.3f} mm\n")
        f.write(f"\nSensor: ACCEL={USE_IMU_ACCEL}, QUAT={USE_IMU_QUAT}, FORCE={USE_FORCE}\n")
        f.write(f"Model: {weights_path}\n")
    print(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()