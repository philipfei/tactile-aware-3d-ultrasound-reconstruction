"""
Sensor data preprocessing: gravity removal, force parsing, SG filtering, normalisation.
Used by both data/prep.py and freehand/loader.py.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

# Normalisation constants
LIN_ACC_NORM = 0.02      # g, linear acceleration range after gravity removal
FORCE_NORM = 3.0        # N — max tactile force range
DT_NORM = 0.5            # s, upper bound on inter-frame time


def parse_force_values(force_str):
    """Parse the force_values CSV field. Only the first 3 of 78 values are used (fx, fy, fz in Newtons)."""
    parts = force_str.split(",")
    fx = float(parts[0])
    fy = float(parts[1])
    fz = float(parts[2])
    return np.array([fx, fy, fz], dtype=np.float64)


def remove_gravity(raw_acc, quats, g_world=None):
    """
    Remove gravity from accelerometer readings.

    Physics: accelerometer measures specific force f = a - g  =>  linear accel a = f + g.

    :param raw_acc: (N, 3) raw accelerometer in g units
    :param quats: (N, 4) IMU quaternion [x, y, z, w], sensor→world rotation
    :param g_world: (3,) world-frame gravity vector in g units. Default: [0, -1, 0] (Y-down)
    :return: (N, 3) linear acceleration in g units, (N, 3) gravity in sensor frame
    """
    if g_world is None:
        g_world = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    rotations = R.from_quat(quats)          # sensor → world
    rotations_inv = rotations.inv()          # world → sensor

    # Rotate world gravity into sensor frame for each row
    g_sensor = rotations_inv.apply(g_world)  # (N, 3)

    # a = f + g
    lin_acc = raw_acc + g_sensor

    return lin_acc, g_sensor


def normalise_quaternions(quats):
    """Normalise each quaternion row to unit length (fixes SG filter drift)."""
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return quats / norms


def filter_sensor_sequence(data, window_length=15, polyorder=3):
    """
    Apply Savitzky-Golay filter to a sensor sequence.
    Returns original data if sequence is too short.

    :param data: (N, C) array
    :return: (N, C) filtered array
    """
    if len(data) < window_length:
        return data
    try:
        return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)
    except Exception:
        return data


def normalise_linear_accel(lin_acc, norm=LIN_ACC_NORM):
    """Normalise linear acceleration and clip to [-1, 1]."""
    normalised = lin_acc / norm
    return np.clip(normalised, -1.0, 1.0)


def compute_force_direction(force_raw):
    """
    Compute the unit vector of the force in the XY plane.
    When ||(fx, fy)|| < 1e-6, returns (0, 0).

    :param force_raw: (N, 3) raw force in Newtons [fx, fy, fz]
    :return: (N, 2) unit vector [dir_x, dir_y] in [-1, 1]
    """
    mag_xy = np.sqrt(force_raw[:, 0] ** 2 + force_raw[:, 1] ** 2)
    mag_xy = np.where(mag_xy < 1e-6, 1.0, mag_xy)  # avoid division by zero
    dir_x = force_raw[:, 0] / mag_xy
    dir_y = force_raw[:, 1] / mag_xy
    # Where magnitude was too small, set direction to zero
    tiny = np.sqrt(force_raw[:, 0] ** 2 + force_raw[:, 1] ** 2) < 1e-6
    dir_x[tiny] = 0.0
    dir_y[tiny] = 0.0
    return np.stack([dir_x, dir_y], axis=1).astype(np.float64)


def normalise_force(force, norm=FORCE_NORM):
    """Normalise tactile force to [0, 1]."""
    return force / norm


def build_sensor_features(lin_acc, quats, force, force_dir):
    """
    Build the 12-channel sensor feature sequence stored in HDF5.

    :param lin_acc: (N, 3) normalised linear acceleration
    :param quats: (N, 4) normalised quaternion [x, y, z, w]
    :param force: (N, 3) normalised force
    :param force_dir: (N, 2) force direction unit vector in XY plane
    :return: (N, 12) sensor sequence [lin_acc(3), quat(4), force(3), force_dir(2)]
    """
    return np.concatenate([lin_acc, quats, force, force_dir], axis=1)


def build_per_frame_sensor(lin_acc, quat, force, force_dir, dt, csv_row_idx):
    """
    Build the 14-element per-frame sensor vector.

    :param force_dir: (2,) force direction unit vector in XY plane
    :param dt: time delta from previous kept frame in seconds (small, float32-safe).
               First frame has dt = 0.
    :return: (14,) array [lin_acc(3), quat(4), force(3), force_dir(2), dt, csv_row_idx]
    """
    return np.concatenate([
        lin_acc,         # 3
        quat,            # 4
        force,           # 3
        force_dir,       # 2 — unit vector in XY plane
        [dt],            # 1 — time delta from prev frame (float32-safe)
        [csv_row_idx],   # 1
    ]).astype(np.float32)


def validate_gravity_removal(lin_acc):
    """Print diagnostics to verify gravity removal correctness."""
    mean_lin = np.mean(lin_acc, axis=0)
    std_lin = np.std(lin_acc, axis=0)
    max_abs = np.max(np.abs(lin_acc), axis=0)
    print(f"  [Gravity check] lin_acc mean: [{mean_lin[0]:.4f}, {mean_lin[1]:.4f}, {mean_lin[2]:.4f}] g")
    print(f"  [Gravity check] lin_acc std:  [{std_lin[0]:.4f}, {std_lin[1]:.4f}, {std_lin[2]:.4f}] g")
    print(f"  [Gravity check] lin_acc max:  [{max_abs[0]:.4f}, {max_abs[1]:.4f}, {max_abs[2]:.4f}] g")
    if abs(mean_lin[1]) > 0.3:
        print(f"  [WARNING] lin_acc Y mean is {mean_lin[1]:.3f}g — gravity may not be fully removed. Try flipping g_world sign.")
    else:
        print(f"  [OK] Gravity removal appears correct.")


def get_sensor_channels(use_accel, use_quat, use_force):
    """
    Return the number of sensor channels and their slice indices based on ablation flags.

    The 10 sensor channels are ordered: [lin_acc(3), quat(4), force(3)]
    """
    channels = []
    offset = 0
    if use_accel:
        channels.append(("accel", slice(offset, offset + 3)))
        offset += 3
    else:
        channels.append(("accel", slice(0, 0)))
    if use_quat:
        channels.append(("quat", slice(offset, offset + 4)))
        offset += 4
    else:
        channels.append(("quat", slice(0, 0)))
    if use_force:
        channels.append(("force", slice(offset, offset + 3)))
        offset += 3
    else:
        channels.append(("force", slice(0, 0)))
    return offset, channels  # total channel count, list of (name, slice) pairs
