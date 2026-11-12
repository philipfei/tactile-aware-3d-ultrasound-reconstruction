import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ================= Configuration (must match data/prep.py) =================
DIR_DATA = os.path.join(os.getcwd(), 'data')
CALIB_PATH = os.path.join(DIR_DATA, 'calib_matrix.csv')
PLOT_DIR = os.path.join(DIR_DATA, 'plotting')

RESAMPLE_FACTOR = 4

# Must match data/prep.py
ROTATE_IMAGE_90 = False
ROTATE_TOOL_90  = False

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
# =========================================================================


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


def check_one_h5(h5_path, split_label, global_stats):
    """Generate QC plots for one HDF5 file and accumulate frame-rate statistics into global_stats."""
    print(f"\n{'='*50}")
    print(f" Checking: {h5_path}  [{split_label}]")
    print(f"{'='*50}")

    calib_scale, calib_rigid = load_calib_matrix(CALIB_PATH, RESAMPLE_FACTOR)
    T_image_to_tool = calib_rigid @ calib_scale

    if ROTATE_IMAGE_90:
        H_orig = 360 // RESAMPLE_FACTOR
        R_rot = np.array([
            [0,  1, 0, 0],
            [-1, 0, 0, H_orig - 1],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ], dtype=np.float64)
        T_image_to_tool = T_image_to_tool @ R_rot

    if ROTATE_TOOL_90:
        R_tool_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R_tool_inv = np.eye(4, dtype=np.float64)
        R_tool_inv[:3, :3] = R_tool_rot.T
        T_image_to_tool = R_tool_inv @ T_image_to_tool

    with h5py.File(h5_path, 'r') as f:
        num_frames_matrix = f['num_frames'][:]
        sub_folders = f['sub_folders'][:]
        frame_height, frame_width = f['frame_size'][:]

        pts_img = np.array([
            [0, 0, 0, 1],
            [frame_width, 0, 0, 1],
            [frame_width, frame_height, 0, 1],
            [0, frame_height, 0, 1]
        ]).T
        pt_tool = np.array([0, 0, 0, 1])

        # Set when any scan contains sensor data
        has_sensor = False
        num_subjects, num_scans_limit = num_frames_matrix.shape

        for i_sub in range(num_subjects):
            folder_name = sub_folders[i_sub].decode('ascii')
            scans_processed = 0

            for i_scan in range(num_scans_limit):
                n_frames = num_frames_matrix[i_sub, i_scan]
                if n_frames < 2:
                    continue

                scans_processed += 1
                print(f"[Processing] Sub {i_sub:03d} ({folder_name}) | Scan {i_scan:02d} | Frames: {n_frames}")

                # Frame transforms
                tforms = []
                for i_frame in range(n_frames):
                    key = '/sub{:03d}_scan{:02d}_tform{:04d}'.format(i_sub, i_scan, i_frame)
                    tforms.append(f[key][:])
                tforms = np.array(tforms)

                # Sensor data, if available
                sensor_kept = None
                dt_kept = None
                base = '/sub{:03d}_scan{:02d}'.format(i_sub, i_scan)
                if base + '_sensor0000' in f:
                    has_sensor = True
                    sensor_kept = np.zeros((n_frames, 12), dtype=np.float32)
                    dt_kept = np.zeros(n_frames, dtype=np.float64)
                    for i_frame in range(n_frames):
                        skey = base + '_sensor{:04d}'.format(i_frame)
                        sdata = f[skey][:]
                        sensor_kept[i_frame] = sdata[:12]  # lin_acc(3), quat(4), force(3), force_dir(2)
                        dt_kept[i_frame] = sdata[12]        # dt

                # Frequency statistics
                if dt_kept is not None and n_frames > 1:
                    # Image frequency: 1/dt for each inter-frame interval (skip frame 0, dt=0)
                    img_intervals = dt_kept[1:]  # drop the first frame (dt = 0)
                    img_freqs = 1.0 / np.maximum(img_intervals, 0.001)  # Hz
                    total_time = np.sum(img_intervals)
                    # Sensor row count from the raw sensor sequence
                    seq_key = base + '_sensor_seq'
                    n_sensor_rows = f[seq_key].shape[0] if seq_key in f else n_frames
                    sensor_freq = n_sensor_rows / total_time if total_time > 0 else 0
                    img_freq = n_frames / total_time if total_time > 0 else 0

                    global_stats['total_frames'] += n_frames
                    global_stats['total_time'] += total_time
                    global_stats['n_scans'] += 1
                    global_stats['img_freqs'].extend(img_freqs.tolist())
                    global_stats['sensor_freqs'].append(sensor_freq)
                    global_stats['img_freqs_per_scan'].append(img_freq)
                    global_stats['sensor_freqs_per_scan'].append(sensor_freq)

                # Trajectory
                tracker_path, corner_paths, frames_planes = [], [[], [], [], []], []
                for tform in tforms:
                    tracker_path.append((tform @ pt_tool)[:3])
                    world_corners = tform @ T_image_to_tool @ pts_img
                    current_plane = []
                    for i_c in range(4):
                        corner_pt = world_corners[:3, i_c]
                        corner_paths[i_c].append(corner_pt)
                        current_plane.append(corner_pt)
                    current_plane.append(world_corners[:3, 0])
                    frames_planes.append(np.array(current_plane))

                tracker_path = np.array(tracker_path)
                corner_paths = [np.array(p) for p in corner_paths]

                # Inter-frame motion
                displacements, rotations = [], []
                for i in range(1, n_frames):
                    T_rel = np.linalg.inv(tforms[i - 1]) @ tforms[i]
                    displacements.append([T_rel[0, 3], T_rel[1, 3], T_rel[2, 3]])
                    rotations.append(R.from_matrix(T_rel[:3, :3]).as_euler('xyz', degrees=True))

                displacements = np.array(displacements)
                rotations = np.array(rotations)
                avg_disp = np.mean(np.abs(displacements), axis=0)
                avg_rot = np.mean(np.abs(rotations), axis=0)
                t_axis = np.arange(1, n_frames)

                # Plotting (4x3 grid with sensor data, 2x2 otherwise)
                title_str = f"[{split_label}] " if split_label else ""

                if has_sensor and sensor_kept is not None:
                    n_rows, n_cols = 4, 3
                    fig = plt.figure(figsize=(24, 20))
                else:
                    n_rows, n_cols = 2, 2
                    fig = plt.figure(figsize=(20, 10))

                fig.suptitle(
                    f"{title_str}QC: {folder_name} | Scan {i_scan:02d} | Frames: {n_frames}",
                    fontsize=14, fontweight='bold'
                )

                # Subplot 1: 3D trajectory (spans 2 cols)
                if has_sensor and sensor_kept is not None:
                    ax1 = fig.add_subplot(n_rows, n_cols, (1, 2), projection='3d')
                else:
                    ax1 = fig.add_subplot(n_rows, n_cols, (1, 3), projection='3d')

                ax1.plot(tracker_path[:, 0], tracker_path[:, 1], tracker_path[:, 2],
                         label="Tracker Path", color='black', linestyle='--', linewidth=2)
                for i_c in range(4):
                    ax1.plot(corner_paths[i_c][:, 0], corner_paths[i_c][:, 1], corner_paths[i_c][:, 2],
                             color=['red', 'green', 'blue', 'orange'][i_c], alpha=0.7)
                for plane in frames_planes:
                    ax1.plot(plane[:, 0], plane[:, 1], plane[:, 2], color='gray', linewidth=0.5, alpha=0.2)
                ax1.set_title("1. Scaled 3D Trajectory & Image Planes")
                set_axes_equal(ax1)

                # Subplot indices depend on the layout
                if has_sensor and sensor_kept is not None:
                    # 4×3 layout: trans(3), rot(4), imu_acc(5), euler(6), force(7), f_angle(8), dt(10,12)
                    idx_trans, idx_rot, idx_imu, idx_euler = 3, 4, 5, 6
                    idx_force, idx_fangle, idx_dt = 7, 8, (10, 12)
                else:
                    # 2×2 layout: trans(2), rot(4)
                    idx_trans, idx_rot = 2, 4

                # Translation subplot
                ax_trans = fig.add_subplot(n_rows, n_cols, idx_trans)
                ax_trans.plot(t_axis, displacements[:, 0], label=f'dx (Avg: {avg_disp[0]:.2f} mm)', color='r')
                ax_trans.plot(t_axis, displacements[:, 1], label=f'dy (Avg: {avg_disp[1]:.2f} mm)', color='g')
                ax_trans.plot(t_axis, displacements[:, 2], label=f'dz (Avg: {avg_disp[2]:.2f} mm)', color='b')
                ax_trans.set_title("2. Inter-frame Translation")
                ax_trans.legend(loc="upper right", fontsize=7)
                ax_trans.grid(True, linestyle=':', alpha=0.7)
                ax_trans.set_xlabel("Frame index")

                # Rotation subplot
                ax_rot = fig.add_subplot(n_rows, n_cols, idx_rot)
                ax_rot.plot(t_axis, rotations[:, 0], label=f'Roll (Avg: {avg_rot[0]:.2f}°)', color='r')
                ax_rot.plot(t_axis, rotations[:, 1], label=f'Pitch (Avg: {avg_rot[1]:.2f}°)', color='g')
                ax_rot.plot(t_axis, rotations[:, 2], label=f'Yaw (Avg: {avg_rot[2]:.2f}°)', color='b')
                ax_rot.set_title("3. Inter-frame Rotation")
                ax_rot.legend(loc="upper right", fontsize=7)
                ax_rot.grid(True, linestyle=':', alpha=0.7)
                ax_rot.set_xlabel("Frame index")

                # Sensor subplots (only if sensor data exists)
                if has_sensor and sensor_kept is not None:
                    # IMU linear acceleration
                    ax_imu = fig.add_subplot(n_rows, n_cols, idx_imu)
                    ax_imu.plot(range(n_frames), sensor_kept[:, 0], label='lin_acc_x', color='r', alpha=0.8)
                    ax_imu.plot(range(n_frames), sensor_kept[:, 1], label='lin_acc_y', color='g', alpha=0.8)
                    ax_imu.plot(range(n_frames), sensor_kept[:, 2], label='lin_acc_z', color='b', alpha=0.8)
                    ax_imu.set_title("4. IMU Linear Acceleration (norm. by 0.02g)")
                    ax_imu.legend(loc="upper right", fontsize=7)
                    ax_imu.grid(True, linestyle=':', alpha=0.7)
                    ax_imu.set_xlabel("Frame index")

                    # IMU Euler angles (from filtered quaternion: sensor_kept[:, 3:7])
                    imu_euler = R.from_quat(sensor_kept[:, 3:7]).as_euler('xyz', degrees=True)
                    ax_euler = fig.add_subplot(n_rows, n_cols, idx_euler)
                    ax_euler.plot(range(n_frames), imu_euler[:, 0], label='roll', color='r', alpha=0.8)
                    ax_euler.plot(range(n_frames), imu_euler[:, 1], label='pitch', color='g', alpha=0.8)
                    ax_euler.plot(range(n_frames), imu_euler[:, 2], label='yaw', color='b', alpha=0.8)
                    ax_euler.set_title("5. IMU Orientation (Euler, deg)")
                    ax_euler.legend(loc="upper right", fontsize=7)
                    ax_euler.grid(True, linestyle=':', alpha=0.7)
                    ax_euler.set_xlabel("Frame index")

                    # Tactile force magnitude
                    ax_force = fig.add_subplot(n_rows, n_cols, idx_force)
                    ax_force.plot(range(n_frames), sensor_kept[:, 7], label='force_x', color='r', alpha=0.8)
                    ax_force.plot(range(n_frames), sensor_kept[:, 8], label='force_y', color='g', alpha=0.8)
                    ax_force.plot(range(n_frames), sensor_kept[:, 9], label='force_z', color='b', alpha=0.8)
                    ax_force.set_title("6. Tactile Force Magnitude")
                    ax_force.legend(loc="upper right", fontsize=7)
                    ax_force.grid(True, linestyle=':', alpha=0.7)
                    ax_force.set_xlabel("Frame index")

                    # Force direction angle
                    force_angle = np.arctan2(sensor_kept[:, 11], sensor_kept[:, 10]) * 180.0 / np.pi
                    ax_fangle = fig.add_subplot(n_rows, n_cols, idx_fangle)
                    ax_fangle.plot(range(n_frames), force_angle, color='purple', alpha=0.8, linewidth=1.5)
                    ax_fangle.set_title("7. Force Direction Angle (XY)")
                    ax_fangle.set_ylabel("Angle (deg)")
                    ax_fangle.grid(True, linestyle=':', alpha=0.7)
                    ax_fangle.set_xlabel("Frame index")
                    ax_fangle.set_ylim(-185, 185)

                    # Frame time deltas (image dt from stored per-frame sensor)
                    ax_dt = fig.add_subplot(n_rows, n_cols, idx_dt)
                    ax_dt.bar(range(n_frames), dt_kept, color='steelblue', alpha=0.7, width=0.8)
                    mean_dt = np.mean(dt_kept[1:]) if n_frames > 1 else 0
                    if mean_dt > 0:
                        ax_dt.axhline(y=mean_dt, color='red', linestyle='--', linewidth=1.5,
                                      label=f'Mean image dt = {mean_dt:.4f}s (~{1.0/mean_dt:.1f} Hz)')
                    ax_dt.set_title("7. Frame Time Deltas (image inter-frame)")
                    ax_dt.legend(loc="upper right", fontsize=7)
                    ax_dt.grid(True, linestyle=':', alpha=0.5, axis='y')
                    ax_dt.set_xlabel("Frame index")
                    ax_dt.set_ylabel("dt (seconds)")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                safe_name = folder_name.replace("/", "_").replace("\\", "_")
                prefix = f"{split_label}_" if split_label else ""
                save_name = f"{prefix}qc_plot_sub{i_sub:03d}_{safe_name}_scan{i_scan:02d}.png"
                plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=200)
                plt.close(fig)

            if scans_processed == 0:
                print(f"[Info] Subject {i_sub:03d} ({folder_name}) has no valid scans, skipping.")

        if has_sensor:
            print(f"\n[OK] Sensor data found and plotted for: {h5_path}")
        else:
            print(f"\n[Info] No sensor data in: {h5_path}")


def main():
    print("=" * 50)
    print(" HDF5 Quality Check (with Sensor Validation)")
    print("=" * 50)

    # Find HDF5 files (one per split, or a single file)
    h5_pattern = os.path.join(DIR_DATA, 'frames_res*.h5')
    h5_files = sorted(glob.glob(h5_pattern))

    if not h5_files:
        print("No HDF5 files found in data/ directory.")
        return

    print(f"Found {len(h5_files)} HDF5 file(s):")
    for h5f in h5_files:
        print(f"  - {os.path.basename(h5f)}")

    # Stats accumulator
    global_stats = {
        'total_frames': 0, 'total_time': 0.0, 'n_scans': 0,
        'img_freqs': [], 'sensor_freqs': [],
        'img_freqs_per_scan': [], 'sensor_freqs_per_scan': [],
    }

    for h5_path in h5_files:
        basename = os.path.basename(h5_path)
        if '_train' in basename:
            split_label = 'train'
        elif '_valid' in basename:
            split_label = 'valid'
        elif '_test' in basename:
            split_label = 'test'
        else:
            split_label = ''

        check_one_h5(h5_path, split_label, global_stats)

    # Write frequency summary
    if global_stats['n_scans'] > 0:
        img_freqs = np.array(global_stats['img_freqs'])
        img_freqs_scan = np.array(global_stats['img_freqs_per_scan'])
        sensor_freqs_scan = np.array(global_stats['sensor_freqs_per_scan'])

        summary_path = os.path.join(PLOT_DIR, 'frequency_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Dataset Frequency Summary ===\n\n")
            f.write(f"Total scans:            {global_stats['n_scans']}\n")
            f.write(f"Total frames:           {global_stats['total_frames']}\n")
            f.write(f"Total scan time:        {global_stats['total_time']:.2f} s  "
                    f"({global_stats['total_time']/60:.2f} min)\n\n")
            f.write(f"--- Image Frequency (per inter-frame interval) ---\n")
            f.write(f"  Mean:                 {np.mean(img_freqs):.3f} Hz\n")
            f.write(f"  Std:                  {np.std(img_freqs):.4f} Hz\n")
            f.write(f"  Std Error:            {np.std(img_freqs)/np.sqrt(len(img_freqs)):.4f} Hz\n")
            f.write(f"  Median:               {np.median(img_freqs):.3f} Hz\n\n")
            f.write(f"--- Image Frequency (per scan) ---\n")
            f.write(f"  Mean:                 {np.mean(img_freqs_scan):.3f} Hz\n")
            f.write(f"  Std Error:            {np.std(img_freqs_scan)/np.sqrt(len(img_freqs_scan)):.4f} Hz\n\n")
            f.write(f"--- Sensor Frequency (per scan) ---\n")
            f.write(f"  Mean:                 {np.mean(sensor_freqs_scan):.3f} Hz\n")
            f.write(f"  Std Error:            {np.std(sensor_freqs_scan)/np.sqrt(len(sensor_freqs_scan)):.4f} Hz\n")
        print(f"\nFrequency summary saved to: {summary_path}")

    print(f"\n{'='*50}")
    print(" QC complete! Plots saved to:", PLOT_DIR)
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
