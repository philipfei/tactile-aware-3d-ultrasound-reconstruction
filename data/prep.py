import os
import sys
import json
import random
import cv2
import h5py
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from freehand.sensor import (
    parse_force_values, remove_gravity, normalise_quaternions,
    filter_sensor_sequence, normalise_linear_accel, normalise_force,
    compute_force_direction,
    build_sensor_features, build_per_frame_sensor, validate_gravity_removal,
)

# ================= 1. Configuration =================
DIR_DATA = os.path.join(os.getcwd(), 'data')
DIR_RAW = os.path.join(DIR_DATA, 'raw_data')
PATH_SAVE = DIR_DATA
CALIB_PATH = os.path.join(DIR_DATA, 'calib_matrix.csv')

RESAMPLE_FACTOR = 4
MAX_SCANS = 20

# Image crop parameters (OpenCV coordinates)
X_START, Y_START = 217, 33
WIDTH, HEIGHT = 240, 360

# Optional coordinate alignment (both disabled).
#   ROTATE_TOOL_90:  rotate the tool frame by -90° around Z (old +Y becomes +X).
#   ROTATE_IMAGE_90: rotate frames 90° clockwise.
ROTATE_IMAGE_90 = False
ROTATE_TOOL_90  = False

# Savitzky-Golay filter parameters
FILTER_ENABLE = False
WINDOW_LENGTH = 5
POLYORDER = 3

G_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)
# ====================================================


# Global histogram equalisation LUT
def compute_global_he_lut(splits, x_start, y_start, width, height,
                          resample_factor, rotate_90, max_images=50000):
    """
    Accumulate a single 256-bin histogram over all splits and return a
    uint8 LUT of shape (256,) for cv2.LUT.

    Uses the same crop/resize/rotate pipeline as process_one_split. The LUT is one
    global monotonic mapping (no per-image or per-split variation), which keeps
    intensities consistent across frames.
    """
    # Collect all JPEG paths from all splits
    all_paths = []
    for _split_name, folders in splits.items():
        for folder_path in folders:
            img_dir = os.path.join(folder_path, 'images')
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.endswith('.jpg'):
                    all_paths.append(os.path.join(img_dir, fname))

    # Random sample covering all splits
    n_total = len(all_paths)
    if max_images and n_total > max_images:
        sampled_paths = random.sample(all_paths, max_images)
        print(f"  Sampling {max_images} images from {n_total} total "
              f"({len(splits)} split(s)).")
    else:
        sampled_paths = all_paths
        print(f"  Using all {n_total} images from {len(splits)} split(s).")

    # Accumulate global histogram
    hist = np.zeros(256, dtype=np.float64)
    for img_path in sampled_paths:
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        cropped = raw[y_start:y_start + height, x_start:x_start + width]
        if resample_factor > 1:
            resized = cv2.resize(cropped,
                                 (width // resample_factor, height // resample_factor),
                                 interpolation=cv2.INTER_AREA)
        else:
            resized = cropped
        if rotate_90:
            resized = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
        # range=(0, 256): one bin per integer intensity
        h, _ = np.histogram(resized.ravel(), bins=256, range=(0, 256))
        hist += h.astype(np.float64)

    # Build LUT, preserving the pure-black background.
    # Ultrasound images have large zero-valued background regions; if bin 0 entered the
    # CDF, black would map to gray. So bin 0 is excluded from the CDF and lut[0] is forced to 0.
    hist_0 = hist[0]
    hist[0] = 0
    cdf = hist.cumsum()
    if cdf[-1] == 0:
        cdf[-1] = 1
    cdf_norm = cdf / cdf[-1]
    lut = np.round(cdf_norm * 255).astype(np.uint8)
    lut[0] = 0  # keep pure black as pure black
    print(f"  Global HE LUT computed from {len(sampled_paths)} images. "
          f"Bin 0 had {int(hist_0):,} pixels (→ lut[0] forced to 0).")
    return lut


# Interactive selection of frame spans (scans)
class SpanManager:
    def __init__(self, ax_shared_list):
        """ax_shared_list: list of axes that share span visualisation."""
        self.spans = []
        self.ax_list = ax_shared_list
        self.patches_list = []

    def onselect(self, xmin, xmax):
        start, end = int(np.floor(xmin)), int(np.ceil(xmax))
        if end > start:
            self.spans.append((start, end))
            scan_idx = len(self.spans) - 1
            print(f"  -> [Slice] Scan {scan_idx:02d}: Frame {start} to {end} ({end - start} frames)")
            batch_patches = []
            for ax in self.ax_list:
                p = ax.axvspan(start, end, alpha=0.3, color='gray')
                batch_patches.append(p)
            self.patches_list.append(batch_patches)
            plt.draw()

    def on_key_press(self, event):
        if event.key in ['z', 'Z']:
            if self.spans:
                removed = self.spans.pop()
                batch = self.patches_list.pop()
                for p in batch:
                    p.remove()
                print(f"  -> [Undo] Removed slice: Frame {removed[0]} to {removed[1]}")
                plt.draw()
            else:
                print("  -> [Hint] No slices to undo.")


# Tracking data smoothing (optional Savitzky-Golay)
def smooth_tracking_data(pos, quats):
    num_samples = len(pos)
    if num_samples < WINDOW_LENGTH:
        print("  [Warning] Sequence shorter than filter window, skipping filter.")
        return pos, R.from_quat(quats).as_matrix()

    if FILTER_ENABLE:
        smooth_pos = savgol_filter(pos, window_length=WINDOW_LENGTH, polyorder=POLYORDER, axis=0)
        try:
            rotvecs = R.from_quat(quats).as_rotvec()
            smooth_rotvecs = savgol_filter(rotvecs, window_length=WINDOW_LENGTH, polyorder=POLYORDER, axis=0)
            smooth_rotmats = R.from_rotvec(smooth_rotvecs).as_matrix()
        except Exception as e:
            print(f"  [Error] Rotation filter failed: {e}, falling back to raw quaternions.")
            smooth_rotmats = R.from_quat(quats).as_matrix()
        return smooth_pos, smooth_rotmats
    else:
        return pos, R.from_quat(quats).as_matrix()


# Calibration matrix loading: returns (scale matrix, rigid matrix); T_cal = rigid @ scale
def load_calib_matrix(filepath, resample_factor=1.0):
    try:
        df = pd.read_csv(filepath, header=None)
        if len(df) >= 8:
            calib_scale = df.iloc[:4, :].values.astype(np.float64)
            calib_rigid = df.iloc[4:8, :].values.astype(np.float64)
        else:
            calib_scale = np.eye(4, dtype=np.float64)
            calib_rigid = df.iloc[:4, :].values.astype(np.float64)

        calib_scale[0, 0] *= resample_factor
        calib_scale[1, 1] *= resample_factor
        return calib_scale, calib_rigid
    except Exception as e:
        print(f"[Warning] Failed to read calib_matrix: {e}, using identity.")
        return np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def process_one_split(split_name, subject_folders, h5_path, he_lut=None):
    """
    Process a list of subject folders into one HDF5 file.

    :param split_name: str, 'train' / 'valid' / 'test' / '' (for flat layout)
    :param subject_folders: list of folder paths (full paths)
    :param h5_path: output HDF5 file path
    """
    num_subjects = len(subject_folders)
    if num_subjects == 0:
        print(f"  -> No subjects found for split '{split_name}', skipping.")
        return

    num_frames_matrix = np.zeros((num_subjects, MAX_SCANS), dtype=np.uint16)

    # After resampling the image is (WIDTH//R, HEIGHT//R) (width, height); rotation swaps them.
    if ROTATE_IMAGE_90:
        # Stored size: width = old height, height = old width
        final_frame_h = WIDTH // RESAMPLE_FACTOR
        final_frame_w = HEIGHT // RESAMPLE_FACTOR
    else:
        final_frame_h = HEIGHT // RESAMPLE_FACTOR
        final_frame_w = WIDTH // RESAMPLE_FACTOR

    calib_scale, calib_rigid = load_calib_matrix(CALIB_PATH, RESAMPLE_FACTOR)
    T_image_to_tool = calib_rigid @ calib_scale

    # Tool rotation matrix (applied to both T_cal and the tracker transforms)
    if ROTATE_TOOL_90:
        R_tool_rot = R.from_euler('z', -90, degrees=True).as_matrix()
        print(f"  Tool frame rotation enabled: old +Y -> new +X")
    else:
        R_tool_rot = None

    # Image rotation: T_cal_new = T_cal_old @ R_rot
    if ROTATE_IMAGE_90:
        H_orig = HEIGHT // RESAMPLE_FACTOR  # height of the pre-rotation image
        R_rot = np.array([
            [0,  1, 0, 0],
            [-1, 0, 0, H_orig - 1],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ], dtype=np.float64)
        T_image_to_tool = T_image_to_tool @ R_rot
        print(f"  Image rotation 90° enabled: stored frame size = {final_frame_w}×{final_frame_h}")
    if ROTATE_TOOL_90:
        # Tool rotation: T_cal_new = R_tool^(-1) @ T_cal_old
        R_tool_inv = np.eye(4, dtype=np.float64)
        R_tool_inv[:3, :3] = R_tool_rot.T
        T_image_to_tool = R_tool_inv @ T_image_to_tool

    pts_img = np.array([
        [0, 0, 0, 1],
        [final_frame_w, 0, 0, 1],
        [final_frame_w, final_frame_h, 0, 1],
        [0, final_frame_h, 0, 1]
    ]).T
    pt_tool = np.array([0, 0, 0, 1])

    fh5 = h5py.File(h5_path, 'w')
    folder_names_list = []

    label = f"{split_name}: " if split_name else ""
    print(f"\n{'='*50}")
    print(f" {label}Processing {num_subjects} subject(s)")
    print(f"{'='*50}")

    for i_sub, folder_path in enumerate(subject_folders):
        folder_name = os.path.basename(folder_path)
        folder_names_list.append(folder_name)
        print(f"\n>>> [{label}Subject {i_sub:03d}/{num_subjects - 1}] {folder_name}")

        csv_path = os.path.join(folder_path, "sensor_data.txt")
        if not os.path.exists(csv_path):
            print("  -> Skipping: sensor_data.txt not found.")
            continue

        df = pd.read_csv(csv_path)
        n_rows = len(df)

        # Raw tracker pose
        pos_raw = df[['fusion_px', 'fusion_py', 'fusion_pz']].values.astype(np.float64)
        quats_raw = df[['fusion_qx', 'fusion_qy', 'fusion_qz', 'fusion_qw']].values.astype(np.float64)

        # Raw sensor data
        imu_acc_raw = df[['imu_ax', 'imu_ay', 'imu_az']].values.astype(np.float64)
        imu_quat_raw = df[['imu_qx', 'imu_qy', 'imu_qz', 'imu_qw']].values.astype(np.float64)
        timestamps = df['timestamp_sec'].values.astype(np.float64) + df['timestamp_nsec'].values.astype(np.float64) / 1e9

        force_raw = np.zeros((n_rows, 3), dtype=np.float64)
        for i_row, fstr in enumerate(df['force_values']):
            force_raw[i_row] = parse_force_values(fstr)

        # Remove gravity from IMU accelerometer
        lin_acc, _ = remove_gravity(imu_acc_raw, imu_quat_raw, g_world=G_WORLD)
        validate_gravity_removal(lin_acc)

        # Optional SG filtering of sensor signals
        if FILTER_ENABLE:
            lin_acc_filt = filter_sensor_sequence(lin_acc, WINDOW_LENGTH, POLYORDER)
            imu_quat_filt = filter_sensor_sequence(imu_quat_raw, WINDOW_LENGTH, POLYORDER)
            force_filt = filter_sensor_sequence(force_raw, WINDOW_LENGTH, POLYORDER)
        else:
            lin_acc_filt = lin_acc
            imu_quat_filt = imu_quat_raw
            force_filt = force_raw

        # Normalise quaternions after filtering
        imu_quat_filt = normalise_quaternions(imu_quat_filt)

        # Normalise linear acceleration and force
        lin_acc_norm = normalise_linear_accel(lin_acc_filt)
        force_norm = normalise_force(force_filt)

        # Force direction in XY plane (unit vector)
        force_dir = compute_force_direction(force_filt)

        # Full sensor feature sequence (all CSV rows)
        sensor_seq = build_sensor_features(lin_acc_norm, imu_quat_filt, force_norm, force_dir)
        # sensor_seq shape: (n_rows, 12)

        # Smooth tracker pose
        smooth_pos, smooth_rotmats = smooth_tracking_data(pos_raw, quats_raw)
 

        # Phase 1: image processing, MD5 de-duplication of repeated frames, per-frame sensor vectors
        kept_frames = []
        kept_tforms = []
        kept_tforms_inv = []
        kept_sensor_per_frame = []   # (14,) per-frame sensor vector
        kept_csv_indices = []        # which CSV row each kept frame came from
        last_md5 = None
        prev_kept_csv = None         # CSV row index of the previous kept frame

        for i_row, row in df.iterrows():
            img_path = os.path.join(folder_path, "images", row['image_filename'])
            if not os.path.exists(img_path):
                continue

            raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                continue

            cropped_img = raw_img[Y_START:Y_START + HEIGHT, X_START:X_START + WIDTH]
            if RESAMPLE_FACTOR > 1:
                # Resize dimensions pre-rotation: (WIDTH//R, HEIGHT//R)
                resized_img = cv2.resize(cropped_img, (WIDTH // RESAMPLE_FACTOR, HEIGHT // RESAMPLE_FACTOR), interpolation=cv2.INTER_AREA)
            else:
                resized_img = cropped_img

            if ROTATE_IMAGE_90:
                resized_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)
                # After rotation: shape is (WIDTH//R, HEIGHT//R) — swapped

            curr_md5 = hashlib.md5(resized_img.tobytes()).hexdigest()
            if curr_md5 != last_md5:
                if he_lut is not None:
                    eq_img = cv2.LUT(resized_img, he_lut)      # uint8, full-range
                else:
                    eq_img = resized_img
                norm_img = eq_img.astype(np.float32) / 255.0   # float32 [0,1]
                tform = np.eye(4, dtype=np.float32)
                tform[:3, :3] = smooth_rotmats[i_row]
                tform[:3, 3] = smooth_pos[i_row]
                if ROTATE_TOOL_90:
                    tform[:3, :3] = tform[:3, :3] @ R_tool_rot
                tform_inv = np.linalg.inv(tform)

                kept_frames.append(norm_img)
                kept_tforms.append(tform)
                kept_tforms_inv.append(tform_inv)

                # dt (s) from the previous kept frame; 0 for the first
                if prev_kept_csv is None:
                    dt = 0.0
                else:
                    dt = float(timestamps[i_row] - timestamps[prev_kept_csv])

                sf = build_per_frame_sensor(
                    lin_acc_norm[i_row],
                    imu_quat_filt[i_row],
                    force_norm[i_row],
                    force_dir[i_row],
                    dt,
                    i_row,
                )
                kept_sensor_per_frame.append(sf)
                kept_csv_indices.append(i_row)

                last_md5 = curr_md5
                prev_kept_csv = i_row

        n_frames = len(kept_frames)
        print(f"  -> MD5 dedup: {len(df)} rows -> {n_frames} unique frames")
        if n_frames < 2:
            print("  -> Not enough frames, skipping.")
            continue

        kept_csv_indices = np.array(kept_csv_indices, dtype=np.int32)

        # Phase 2: trajectory and inter-frame motion
        tracker_path, corner_paths, frames_planes = [], [[], [], [], []], []
        for tform in kept_tforms:
            tracker_path.append((tform @ pt_tool)[:3])
            world_corners = tform @ T_image_to_tool @ pts_img
            current_plane_pts = []
            for i_c in range(4):
                corner_pt = world_corners[:3, i_c]
                corner_paths[i_c].append(corner_pt)
                current_plane_pts.append(corner_pt)
            current_plane_pts.append(world_corners[:3, 0])
            frames_planes.append(np.array(current_plane_pts))

        tracker_path = np.array(tracker_path)
        corner_paths = [np.array(p) for p in corner_paths]

        displacements, rotations = [], []
        for i in range(1, n_frames):
            T_rel = kept_tforms_inv[i - 1] @ kept_tforms[i]
            displacements.append([T_rel[0, 3], T_rel[1, 3], T_rel[2, 3]])
            rotations.append(R.from_matrix(T_rel[:3, :3]).as_euler('xyz', degrees=True))

        displacements = np.array(displacements)
        rotations = np.array(rotations)
        t_axis = np.arange(1, n_frames)

        avg_disp = np.mean(np.abs(displacements), axis=0)
        avg_rot = np.mean(np.abs(rotations), axis=0)

        # Sensor data at kept frames (for visualisation)
        sensor_kept = np.array([kept_sensor_per_frame[i][:12] for i in range(n_frames)])
        # columns: lin_acc(3), quat(4), force(3), force_dir(2)
        dt_kept = np.array([kept_sensor_per_frame[i][12] for i in range(n_frames)], dtype=np.float64)
        # dt is at index 12 (after the 12 sensor channels)

        # Raw sensor dt (full sensor sequence, for overlay on the dt subplot)
        raw_sensor_dt = np.diff(timestamps)  # (N-1,) float64, inter-sensor-row deltas

        # IMU Euler angles from quaternion (for visualisation)
        imu_euler = R.from_quat(sensor_kept[:, 3:7]).as_euler('xyz', degrees=True)
        # columns: roll, pitch, yaw (degrees)

        # Phase 3: interactive visualisation (4x3 grid)
        # Layout:
        #   Row 1: 3D trajectory (cols 1-2)  |  Translation (col 3) [Drag to Slice]
        #   Row 2: Rotation (col 1)  |  IMU accel (col 2)  |  IMU Euler (col 3)
        #   Row 3: Force magnitude (col 1)  |  Force angle (col 2)
        #   Row 4: dt: image + sensor (cols 1-3, full width)
        n_rows, n_cols = 4, 3
        fig = plt.figure(figsize=(24, 20))
        title_str = f"{split_name}: {folder_name}" if split_name else folder_name
        fig.suptitle(
            f"Subject: {title_str} | Frames: {n_frames} | Resample: {RESAMPLE_FACTOR}x\n"
            f"(Drag on subplot 2 to select slices. Press 'Z' to undo. Close window to confirm.)",
            fontsize=13, fontweight='bold'
        )

        # Row 1, cols 1-2: 3D trajectory
        ax1 = fig.add_subplot(n_rows, n_cols, (1, 2), projection='3d')
        ax1.plot(tracker_path[:, 0], tracker_path[:, 1], tracker_path[:, 2],
                 label="Tracker Path", color='black', linestyle='--', linewidth=2)
        colors = ['red', 'green', 'blue', 'orange']
        for i_c in range(4):
            ax1.plot(corner_paths[i_c][:, 0], corner_paths[i_c][:, 1], corner_paths[i_c][:, 2],
                     color=colors[i_c], alpha=0.7)
        for plane_pts in frames_planes:
            ax1.plot(plane_pts[:, 0], plane_pts[:, 1], plane_pts[:, 2],
                     color='gray', linewidth=0.5, alpha=0.3)
        ax1.set_title("1. Scaled 3D Trajectory & Image Planes")
        set_axes_equal(ax1)

        # Row 1, col 3: Translation (interactive slice selection target)
        ax2 = fig.add_subplot(n_rows, n_cols, 3)
        ax2.plot(t_axis, displacements[:, 0], label=f'dx (Avg: {avg_disp[0]:.2f} mm)', color='r', alpha=0.8)
        ax2.plot(t_axis, displacements[:, 1], label=f'dy (Avg: {avg_disp[1]:.2f} mm)', color='g', alpha=0.8)
        ax2.plot(t_axis, displacements[:, 2], label=f'dz (Avg: {avg_disp[2]:.2f} mm)', color='b', alpha=0.8)
        ax2.set_title("2. Inter-frame Translation [Drag to Slice]")
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, linestyle=':', alpha=0.7)
        ax2.set_xlabel("Frame index")

        # Row 2, col 1: Rotation
        ax3 = fig.add_subplot(n_rows, n_cols, 4)
        ax3.plot(t_axis, rotations[:, 0], label=f'd_roll (Avg: {avg_rot[0]:.2f}°)', color='r', alpha=0.8)
        ax3.plot(t_axis, rotations[:, 1], label=f'd_pitch (Avg: {avg_rot[1]:.2f}°)', color='g', alpha=0.8)
        ax3.plot(t_axis, rotations[:, 2], label=f'd_yaw (Avg: {avg_rot[2]:.2f}°)', color='b', alpha=0.8)
        ax3.set_title("3. Inter-frame Rotation")
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=':', alpha=0.7)
        ax3.set_xlabel("Frame index")

        # Row 2, col 2: IMU linear acceleration
        ax4 = fig.add_subplot(n_rows, n_cols, 5)
        ax4.plot(range(n_frames), sensor_kept[:, 0], label='lin_acc_x', color='r', alpha=0.8)
        ax4.plot(range(n_frames), sensor_kept[:, 1], label='lin_acc_y', color='g', alpha=0.8)
        ax4.plot(range(n_frames), sensor_kept[:, 2], label='lin_acc_z', color='b', alpha=0.8)
        ax4.set_title("4. IMU Linear Acceleration (norm. by 0.02g)")
        ax4.legend(loc="upper right", fontsize=7)
        ax4.grid(True, linestyle=':', alpha=0.7)
        ax4.set_xlabel("Frame index")

        # Row 2, col 3: IMU Euler angles (from filtered quaternion)
        ax5 = fig.add_subplot(n_rows, n_cols, 6)
        ax5.plot(range(n_frames), imu_euler[:, 0], label='roll', color='r', alpha=0.8)
        ax5.plot(range(n_frames), imu_euler[:, 1], label='pitch', color='g', alpha=0.8)
        ax5.plot(range(n_frames), imu_euler[:, 2], label='yaw', color='b', alpha=0.8)
        ax5.set_title("5. IMU Orientation (Euler, deg)")
        ax5.legend(loc="upper right", fontsize=7)
        ax5.grid(True, linestyle=':', alpha=0.7)
        ax5.set_xlabel("Frame index")

        # Force angle from direction (atan2)
        force_angle = np.arctan2(sensor_kept[:, 11], sensor_kept[:, 10]) * 180.0 / np.pi

        # Row 3, col 1: Tactile force magnitude
        ax6 = fig.add_subplot(n_rows, n_cols, 7)
        ax6.plot(range(n_frames), sensor_kept[:, 7], label='force_x', color='r', alpha=0.8)
        ax6.plot(range(n_frames), sensor_kept[:, 8], label='force_y', color='g', alpha=0.8)
        ax6.plot(range(n_frames), sensor_kept[:, 9], label='force_z', color='b', alpha=0.8)
        ax6.set_title("6. Tactile Force Magnitude")
        ax6.legend(loc="upper right", fontsize=7)
        ax6.grid(True, linestyle=':', alpha=0.7)
        ax6.set_xlabel("Frame index")

        # Row 3, col 2: Force direction angle
        ax6b = fig.add_subplot(n_rows, n_cols, 8)
        ax6b.plot(range(n_frames), force_angle, color='purple', alpha=0.8, linewidth=1.5)
        ax6b.set_title("7. Force Direction Angle (XY)")
        ax6b.set_ylabel("Angle (deg)")
        ax6b.grid(True, linestyle=':', alpha=0.7)
        ax6b.set_xlabel("Frame index")
        ax6b.set_ylim(-185, 185)

        # Row 4, cols 1-3: Frame time deltas (image dt bars + sensor dt overlay)
        ax7 = fig.add_subplot(n_rows, n_cols, (10, 12))
        ax7.bar(range(n_frames), dt_kept, color='steelblue', alpha=0.7, width=0.8,
                label='Image dt (inter-frame)')
        # Overlay raw sensor dt as a scatter/horizontal line
        mean_sensor_dt = np.mean(raw_sensor_dt)
        ax7.axhline(y=mean_sensor_dt, color='red', linestyle='-', linewidth=1.5,
                    label=f'Sensor mean dt = {mean_sensor_dt:.4f}s (~{1.0/mean_sensor_dt:.1f} Hz)')
        ax7.axhline(y=np.mean(dt_kept[1:]), color='orange', linestyle='--', linewidth=1.5,
                    label=f'Image mean dt = {np.mean(dt_kept[1:]):.4f}s (~{1.0/np.mean(dt_kept[1:]):.1f} Hz)')
        ax7.set_title("7. Frame Time Deltas (blue bars = image, red line = sensor native rate)")
        ax7.legend(loc="upper right", fontsize=7)
        ax7.grid(True, linestyle=':', alpha=0.5, axis='y')
        ax7.set_xlabel("Frame index")
        ax7.set_ylabel("dt (seconds)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Phase 4: bind interactive span selection
        span_manager = SpanManager([ax2, ax3, ax4, ax5, ax6, ax6b, ax7])
        span_selector = SpanSelector(
            ax2, span_manager.onselect, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='red'), interactive=True
        )
        fig.canvas.mpl_connect('key_press_event', span_manager.on_key_press)

        # Reuse saved intervals if available, otherwise select interactively
        intervals_dir = os.path.join(DIR_DATA, 'intervals')
        intervals_path = os.path.join(intervals_dir, f'{folder_name}.json')
        loaded_spans = None

        if os.path.exists(intervals_path):
            try:
                with open(intervals_path, 'r') as f:
                    saved = json.load(f)
                # Convert saved CSV-row spans back to kept-frame indices
                loaded_spans = []
                for csv_start, csv_end in saved['csv_spans']:
                    start_idx = int(np.argmin(np.abs(kept_csv_indices - csv_start)))
                    end_idx = int(np.argmin(np.abs(kept_csv_indices - csv_end)))
                    if end_idx > start_idx:
                        loaded_spans.append((start_idx, end_idx))
                if loaded_spans:
                    print(f"  -> Loaded {len(loaded_spans)} saved interval(s) from {intervals_path}")
                else:
                    print(f"  -> Saved intervals file exists but no valid spans found. Re-selecting.")
                    loaded_spans = None
            except Exception as e:
                print(f"  -> Failed to load intervals: {e}. Re-selecting.")
                loaded_spans = None

        if loaded_spans is not None:
            span_manager = SpanManager([])  # no-op, spans pre-loaded
            span_manager.spans = loaded_spans
        else:
            print("  -> Waiting for interactive slice selection... (Z to undo, close window to continue)")
            plt.show()

        # Phase 5: write selected spans to HDF5
        spans = loaded_spans if loaded_spans is not None else span_manager.spans
        if not spans:
            print("  -> [Skip] No slices selected.")
            continue

        # Save intervals for future runs (using CSV row indices, robust to parameter changes)
        if loaded_spans is None and span_manager.spans:
            os.makedirs(intervals_dir, exist_ok=True)
            csv_spans = []
            for start_idx, end_idx in span_manager.spans:
                csv_start = int(kept_csv_indices[start_idx])
                csv_end = int(kept_csv_indices[end_idx - 1])
                csv_spans.append([csv_start, csv_end])
            with open(intervals_path, 'w') as f:
                json.dump({'csv_spans': csv_spans}, f, indent=2)
            print(f"  -> Saved {len(csv_spans)} interval(s) to {intervals_path}")

        for scan_idx, (start_idx, end_idx) in enumerate(spans):
            if scan_idx >= MAX_SCANS:
                print(f"  -> [Warning] Exceeded max scans ({MAX_SCANS}), truncating.")
                break

            span_frames = kept_frames[start_idx:end_idx]
            span_tforms = kept_tforms[start_idx:end_idx]
            span_tforms_inv = kept_tforms_inv[start_idx:end_idx]
            span_sensors = kept_sensor_per_frame[start_idx:end_idx]
            span_len = len(span_frames)

            num_frames_matrix[i_sub, scan_idx] = span_len

            for i_frame in range(span_len):
                base = '/sub%03d_scan%02d' % (i_sub, scan_idx)
                fh5.create_dataset(base + '_frame%04d' % i_frame, data=span_frames[i_frame], dtype=np.float32)
                fh5.create_dataset(base + '_tform%04d' % i_frame, data=span_tforms[i_frame], dtype=np.float32)
                fh5.create_dataset(base + '_tform_inv%04d' % i_frame, data=span_tforms_inv[i_frame], dtype=np.float32)
                fh5.create_dataset(base + '_sensor%04d' % i_frame, data=span_sensors[i_frame], dtype=np.float32)

            # Per-scan raw sensor sequence: the CSV rows that fall within this span
            span_start_csv = kept_csv_indices[start_idx]
            span_end_csv = kept_csv_indices[end_idx - 1]
            # Include all rows from span_start_csv to span_end_csv (inclusive)
            scan_sensor_seq = sensor_seq[span_start_csv:span_end_csv + 1]  # (N_raw_in_scan, 12)
            # Mapping: for each kept frame in the span, its CSV row index (relative to scan_sensor_seq start)
            scan_sensor_map = kept_csv_indices[start_idx:end_idx] - span_start_csv  # (span_len,)

            fh5.create_dataset(base + '_sensor_seq', data=scan_sensor_seq, dtype=np.float32)
            fh5.create_dataset(base + '_sensor_map', data=scan_sensor_map, dtype=np.int32)

            print(f"  -> [Written] Scan {scan_idx:02d} | Frames: {span_len} | Sensor rows: {scan_sensor_seq.shape[0]} | Key: /sub{i_sub:03d}_scan{scan_idx:02d}_...")

    # Global metadata
    fh5.create_dataset('num_frames', data=num_frames_matrix, dtype=np.uint16)
    folder_names_ascii = [f.encode("ascii", "ignore") for f in folder_names_list]
    fh5.create_dataset('sub_folders', data=folder_names_ascii)
    fh5.create_dataset('frame_size', data=np.array([final_frame_h, final_frame_w], dtype=np.uint16))

    fh5.flush()
    fh5.close()
    print(f"\n{split_name} split saved to: {h5_path}")


def discover_splits():
    """
    Check if raw_data contains train/valid/test subdirectories.
    Returns dict: split_name -> list of subject folder paths.
    If no subdirs are found, returns {'': [all subject folders]}.
    """
    if not os.path.exists(DIR_RAW):
        raise FileNotFoundError(f"Data directory not found: {DIR_RAW}")

    split_dirs = {}
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(DIR_RAW, split)
        if os.path.isdir(split_path):
            folders = sorted([
                os.path.join(split_path, f)
                for f in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, f))
                and f != 'temporary_ignore'
            ])
            if folders:
                split_dirs[split] = folders

    if split_dirs:
        print(f"Found folder-based splits: {list(split_dirs.keys())}")
        # Check for subjects also in root raw_data (not inside a split folder)
        root_folders = [
            f for f in os.listdir(DIR_RAW)
            if os.path.isdir(os.path.join(DIR_RAW, f))
            and f not in ['train', 'valid', 'test', 'temporary_ignore']
        ]
        if root_folders:
            print(f"  [Warning] Also found {len(root_folders)} subject(s) directly in raw_data — these will be IGNORED when splits are present.")
        return split_dirs

    # Flat layout (no train/valid/test subdirs)
    all_folders = sorted([
        os.path.join(DIR_RAW, f)
        for f in os.listdir(DIR_RAW)
        if os.path.isdir(os.path.join(DIR_RAW, f))
        and f != 'temporary_ignore'
    ])
    if all_folders:
        print("No train/valid/test subdirectories found. Using flat layout (single HDF5).")
    return {'': all_folders}


def process_dataset():
    splits = discover_splits()

    # Compute or load the global HE LUT (shared by all splits)
    lut_path = os.path.join(PATH_SAVE, f'he_global_lut_res{RESAMPLE_FACTOR}.npy')
    if os.path.exists(lut_path):
        print(f"Loading cached global HE LUT: {lut_path}")
        he_lut = np.load(lut_path)
    else:
        print("Computing global histogram equalization LUT from all splits...")
        he_lut = compute_global_he_lut(
            splits, X_START, Y_START, WIDTH, HEIGHT,
            RESAMPLE_FACTOR, ROTATE_IMAGE_90)
        np.save(lut_path, he_lut)
        print(f"Global HE LUT saved to: {lut_path}")

    for split_name, folders in splits.items():
        if split_name:
            h5_filename = os.path.join(PATH_SAVE, f'frames_res{RESAMPLE_FACTOR}_{split_name}.h5')
        else:
            h5_filename = os.path.join(PATH_SAVE, f'frames_res{RESAMPLE_FACTOR}.h5')

        process_one_split(split_name, folders, h5_filename, he_lut)

    print(f"\n{'='*50}")
    print(" All processing complete!")
    print(f"{'='*50}")


if __name__ == '__main__':
    process_dataset()
