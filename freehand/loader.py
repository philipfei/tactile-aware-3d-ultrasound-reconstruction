
import random
import json

import h5py
import numpy as np


class SSFrameDataset():
    """Subject-Scan frame loader with optional sensor data support."""

    def __init__(self, filename_h5, indices_in_use=None, num_samples=2, sample_range=None,
                 sensor_in_use=False, sen_window_size=0):
        """
        :param filename_h5: file path
        :param indices_in_use:
            case 1: list of tuples (idx_subject, idx_scans)
            case 2: list of two lists, [indices_subjects] and [indices_scans], meshgrid
        :param sen_window_size: number of most recent sensor rows per frame (sensor trend window).
                               0 = no trend window (instantaneous sensor sub-branch only).
            case 3: None (default), use all available in the file
        :param num_samples: int, number of model input frames, > 1. -1 = all frames in scan.
        :param sample_range: int, range of sampling frames, default is num_samples.
        :param sensor_in_use: bool, whether to load sensor data.
        """

        self.filename = filename_h5
        self.file = h5py.File(self.filename, 'r')
        self.frame_size = self.file['frame_size'][()]
        self.num_frames = self.file['num_frames'][()]
        self.sensor_in_use = sensor_in_use

        if indices_in_use is None:
            self.indices_in_use = [
                (i_sub, i_scn)
                for i_sub in range(self.num_frames.shape[0])
                for i_scn in range(self.num_frames.shape[1])
                if self.num_frames[i_sub, i_scn] > 0
            ]
        elif all([isinstance(t, tuple) for t in indices_in_use]):
            self.indices_in_use = indices_in_use
        elif isinstance(indices_in_use[0], list) and isinstance(indices_in_use[1], list):
            self.indices_in_use = [
                (i_sub, i_scn)
                for i_sub in indices_in_use[0]
                for i_scn in indices_in_use[1]
            ]
        else:
            raise ValueError(
                "indices_in_use should be a list of tuples (idx_subject, idx_scans) "
                "or two lists [indices_subjects] and [indices_scans]."
            )

        if len(set(self.indices_in_use)) != len(self.indices_in_use):
            print("WARNING: Replicated indices found — not removed.")

        self.indices_in_use.sort()
        self.num_indices = len(self.indices_in_use)

        # sampling parameters
        if num_samples < 2:
            if num_samples == -1:
                if sample_range is not None:
                    sample_range = None
                    print("Sampling all frames. sample_range is ignored.")
            else:
                raise ValueError('num_samples must be >= 2, or -1 for all frames.')
        self.num_samples = num_samples

        if sample_range is None:
            self.sample_range = self.num_samples
        elif any([self.num_frames[indices] < sample_range for indices in self.indices_in_use]):
            raise ValueError(
                "sample_range is larger than the number of frames in at least one scan."
            )
        else:
            self.sample_range = sample_range

        # Pre-load sensor data if requested
        self._sensor_seqs = {}     # (i_sub, i_scan) -> (N_raw, 12) array
        self._sensor_maps = {}     # (i_sub, i_scan) -> (N_kept,) int array
        self.sen_window_size = sen_window_size

        if self.sensor_in_use:
            self._load_all_sensor_data()

        # Close the HDF5 handle so the dataset can be pickled by DataLoader workers.
        # It is reopened lazily in _get_file().
        self.file.close()
        self.file = None

    def _load_all_sensor_data(self):
        """Pre-load all sensor sequences and mappings into memory."""
        has_sensor = False

        for i_sub in range(self.num_frames.shape[0]):
            for i_scan in range(self.num_frames.shape[1]):
                n_frames = self.num_frames[i_sub, i_scan]
                if n_frames < 2:
                    continue

                base = '/sub{:03d}_scan{:02d}'.format(i_sub, i_scan)
                seq_key = base + '_sensor_seq'
                map_key = base + '_sensor_map'

                if seq_key not in self.file:
                    continue

                has_sensor = True
                sensor_seq = self.file[seq_key][:]     # (N_raw, 12)
                sensor_map = self.file[map_key][:]      # (N_kept,)
                self._sensor_seqs[(i_sub, i_scan)] = sensor_seq
                self._sensor_maps[(i_sub, i_scan)] = sensor_map

        if has_sensor:
            print(f"Sensor data loaded ({len(self._sensor_seqs)} scans).")
        elif self.sensor_in_use:
            print("WARNING: sensor_in_use=True but no sensor data found in HDF5. "
                  "Falling back to image-only mode.")
            self.sensor_in_use = False

    def partition_by_ratio(self, ratios, randomise=False, subject_level=False):
        num_sets = len(ratios)
        ratios = [ratios[i] / sum(ratios) for i in range(num_sets)]
        print("Partitioning into %d sets with normalised ratios %s." % (num_sets, ratios))

        if subject_level:
            raise NotImplementedError('Subject-level split has not been implemented.')

        set_sizes = [int(self.num_indices * r) for r in ratios]
        for ii in range(self.num_indices - sum(set_sizes)):
            set_sizes[ii] += 1
        if randomise:
            random.shuffle(self.indices_in_use)
        indices_sets = [
            self.indices_in_use[sum(set_sizes[:ii]):sum(set_sizes[:ii]) + n1]
            for (ii, n1) in enumerate(set_sizes)
        ]
        print("at scan-level, with %s scans." % (set_sizes,))

        return [
            SSFrameDataset(
                filename_h5=self.filename,
                indices_in_use=idx_list,
                num_samples=self.num_samples,
                sample_range=self.sample_range,
                sensor_in_use=self.sensor_in_use,
                sen_window_size=self.sen_window_size,
            )
            for idx_list in indices_sets
        ]

    def __add__(self, other):
        if self.filename != other.filename:
            raise ValueError('Combining datasets from different files is not supported.')
        if self.num_samples != other.num_samples:
            print('WARNING: different num_samples — using the first.')
        if self.sample_range != other.sample_range:
            print('WARNING: different sample_range — using the first.')
        if self.sensor_in_use != other.sensor_in_use:
            print('WARNING: different sensor_in_use — using the first.')
        indices_combined = self.indices_in_use + other.indices_in_use
        return SSFrameDataset(
            filename_h5=self.filename,
            indices_in_use=indices_combined,
            num_samples=self.num_samples,
            sample_range=self.sample_range,
            sensor_in_use=self.sensor_in_use,
        )

    def _get_file(self):
        """Reopen the HDF5 file on first access."""
        if self.file is None:
            self.file = h5py.File(self.filename, 'r')
        return self.file

    def __len__(self):
        return self.num_indices

    def __getitem__(self, idx):
        indices = self.indices_in_use[idx]
        if self.num_samples == -1:
            i_frames = list(range(self.num_frames[indices]))
        else:
            i_frames = self.frame_sampler(self.num_frames[indices])

        f = self._get_file()
        frames = [
            f['/sub{:03d}_scan{:02d}_frame{:04d}'.format(indices[0], indices[1], ii)]
            for ii in i_frames
        ]
        tforms = [
            f['/sub{:03d}_scan{:02d}_tform{:04d}'.format(indices[0], indices[1], ii)]
            for ii in i_frames
        ]
        tforms_inv = [
            f['/sub{:03d}_scan{:02d}_tform_inv{:04d}'.format(indices[0], indices[1], ii)]
            for ii in i_frames
        ]

        if not self.sensor_in_use:
            return np.stack(frames, axis=0), np.stack(tforms, axis=0), np.stack(tforms_inv, axis=0)

        # Sensor data
        scan_key = (indices[0], indices[1])
        sensor_seq = self._sensor_seqs.get(scan_key)
        sensor_map = self._sensor_maps.get(scan_key)

        # Per-frame instantaneous sensor features
        sensors = np.zeros((len(i_frames), 14), dtype=np.float32)
        for j, ifr in enumerate(i_frames):
            skey = '/sub{:03d}_scan{:02d}_sensor{:04d}'.format(indices[0], indices[1], ifr)
            sensors[j] = f[skey][:]

        # Sensor trend window: the last ws sensor rows up to each frame
        # Shape: (n_frames, ws, raw_ch+1) — raw_ch sensor channels + 1 time-position channel
        ws = self.sen_window_size
        raw_ch = sensor_seq.shape[1] if sensor_seq is not None else 12
        sensor_wins = np.zeros((len(i_frames), max(ws, 1), raw_ch + 1), dtype=np.float32)
        win_lens = np.full(len(i_frames), ws, dtype=np.int32)

        # Time-position channel: normalized row index [0=oldest, 1=newest]
        if ws > 1:
            time_pos = np.linspace(0, 1, ws, dtype=np.float32)
        else:
            time_pos = np.array([0], dtype=np.float32)
        sensor_wins[:, :, -1] = time_pos[np.newaxis, :]  # broadcast across all frames

        if ws > 0 and sensor_seq is not None and sensor_map is not None:
            for j, ifr in enumerate(i_frames):
                csv_row = int(sensor_map[ifr])
                start = max(0, csv_row - ws + 1)
                end = csv_row + 1
                window = sensor_seq[start:end]  # (actual_len, raw_ch)
                actual_len = window.shape[0]
                if actual_len >= ws:
                    sensor_wins[j, :, :raw_ch] = window[-ws:]
                elif actual_len > 0:
                    sensor_wins[j, -actual_len:, :raw_ch] = window
                    pad_len = ws - actual_len
                    sensor_wins[j, :pad_len, :raw_ch] = np.tile(window[0:1], (pad_len, 1))
                else:
                    own = sensor_seq[csv_row:csv_row+1]
                    sensor_wins[j, :, :raw_ch] = np.tile(own, (ws, 1))

        return (
            np.stack(frames, axis=0),
            np.stack(tforms, axis=0),
            np.stack(tforms_inv, axis=0),
            sensors,
            sensor_wins,
            win_lens,
        )

    def frame_sampler(self, n):
        n0 = random.randint(0, n - self.sample_range)
        idx_frames = random.sample(range(n0, n0 + self.sample_range), self.num_samples)
        idx_frames.sort()
        return idx_frames

    def consecutive_sequence_sampler(self, start_idx, seq_frames):
        """
        Return `seq_frames` consecutive retained-frame indices starting at `start_idx`.
        Used to draw contiguous frame sequences.
        """
        return list(range(start_idx, start_idx + seq_frames))

    def write_json(self, jason_filename):
        with open(jason_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "filename": self.filename,
                "indices_in_use": self.indices_in_use,
                "num_samples": self.num_samples,
                "sample_range": self.sample_range,
                "sensor_in_use": self.sensor_in_use,
                "sen_window_size": self.sen_window_size,
            }, f, ensure_ascii=False, indent=4)
        print("%s written." % jason_filename)

    @staticmethod
    def read_json(jason_filename, num_samples=None):
        with open(jason_filename, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            if num_samples is None:
                num_samples = obj['num_samples']
            return SSFrameDataset(
                filename_h5=obj['filename'],
                indices_in_use=[tuple(ids) for ids in obj['indices_in_use']],
                num_samples=num_samples,
                sample_range=obj['sample_range'],
                sensor_in_use=obj.get('sensor_in_use', False),
                sen_window_size=obj.get('sen_window_size', 0),
            )
