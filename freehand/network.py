
import torch
import torch.nn as nn


def build_model(model_function, in_frames, out_dim):
    """
    Build an image-only model with the first conv adapted to stacked input frames.

    :param model_function: torchvision classification model constructor
    :param in_frames: int, number of input channels (stacked grayscale frames)
    :param out_dim: int, output dimension
    """
    if model_function.__name__[:12] == "efficientnet":
        model = model_function(weights=None)
        model.features[0][0] = nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias,
        )
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=out_dim,
        )
    elif model_function.__name__[:6] == "resnet":
        model = model_function()
        model.conv1 = nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias,
        )
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=out_dim,
        )
    else:
        raise ValueError("Unknown model type.")
    return model


# ---------------------------------------------------------------------------
# Sensor Fusion Components
# ---------------------------------------------------------------------------

class SensorWindowCNN(nn.Module):
    """
    1D CNN that encodes a sensor trend window into a feature vector.
    All windows have the same length, so global average pooling is used.
    """

    def __init__(self, in_channels=10, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        :param x: (batch, win_size, in_channels) — sensor trend windows
        :return: (batch, hidden_dim)
        """
        x = x.transpose(1, 2)              # (batch, channels, win_size)
        x = self.conv(x)                    # (batch, hidden_dim, win_size)
        x = self.pool(x)                    # (batch, hidden_dim, 1)
        return x.squeeze(-1)                # (batch, hidden_dim)


class SensorFusionModel(nn.Module):
    """
    Intermediate fusion model combining:
      - EfficientNet image sub-branch (1280-dim features, backbone can be frozen)
      - 1D CNN sensor trend window branch (128-dim features)
      - Instantaneous sensor sub-branch, an MLP (64-dim features)
      - Fusion head (1472 -> 512 -> out_dim)
    """

    def __init__(self, efficientnet_features, image_feature_dim, out_dim,
                 sensor_cnn_channels, sensor_inst_dim):
        """
        :param efficientnet_features: nn.Sequential, the EfficientNet feature extractor
        :param image_feature_dim: int, dimension of image features (1280 for b1)
        :param out_dim: int, output dimension (270 for 45 pairs x 6-DoF)
        :param sensor_cnn_channels: int, number of sensor channels per window row for the 1D CNN (0 = no sensor)
        :param sensor_inst_dim: int, per-frame instantaneous sensor dim (0 = no sensor)
        """
        super().__init__()
        self.features = efficientnet_features
        self.image_feature_dim = image_feature_dim
        self.has_sensor = (sensor_cnn_channels > 0) or (sensor_inst_dim > 0)

        # Sensor branches (created only when sensor input is used)
        if sensor_cnn_channels > 0:
            self.sensor_cnn = SensorWindowCNN(in_channels=sensor_cnn_channels, hidden_dim=128)
        else:
            self.sensor_cnn = None

        if sensor_inst_dim > 0:
            self.sensor_inst_mlp = nn.Sequential(
                nn.Linear(sensor_inst_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
        else:
            self.sensor_inst_mlp = None

        # Fusion head
        fusion_dim = image_feature_dim
        if self.sensor_cnn is not None:
            fusion_dim += 128
        if self.sensor_inst_mlp is not None:
            fusion_dim += 64

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, frames, sensors=None, sensor_wins=None, win_lens=None):
        """
        :param frames: (batch, num_samples, H, W) — stacked grayscale frames
        :param sensors: (batch, num_samples, 14) — per-frame instantaneous sensor features, optional
        :param sensor_wins: (batch, num_samples, win_size, sensor_cnn_channels) — sensor trend windows, optional
        :param win_lens: unused
        :return: (batch, out_dim)
        """
        # Image branch
        img_features = self.features(frames)                     # (batch, 1280, H', W')
        img_features = img_features.mean(dim=[2, 3])             # (batch, 1280) — global avg pool

        features = [img_features]

        # Sensor trend window branch
        if self.sensor_cnn is not None and sensor_wins is not None:
            batch_size, num_samples, win_size, channels = sensor_wins.shape
            sw_flat = sensor_wins.reshape(batch_size * num_samples, win_size, channels)
            cnn_out = self.sensor_cnn(sw_flat)                   # (batch*num_samples, 128)
            cnn_out = cnn_out.reshape(batch_size, num_samples, 128)
            cnn_out = cnn_out.mean(dim=1)                        # (batch, 128) — mean pool across frames
            features.append(cnn_out)

        # Instantaneous sensor sub-branch
        if self.sensor_inst_mlp is not None and sensors is not None:
            batch_size, num_samples, inst_dim = sensors.shape
            inst_flat = sensors.reshape(batch_size, num_samples * inst_dim)
            inst_out = self.sensor_inst_mlp(inst_flat)           # (batch, 64)
            features.append(inst_out)

        fused = torch.cat(features, dim=1)
        return self.fusion_head(fused)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze all EfficientNet convolutional layers."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, stages=None):
        """
        Unfreeze specified EfficientNet stages.

        EfficientNet-b1 features layout (9 children):
          [0] ConvNormActivation (stem, 32ch)
          [1..7] Sequential MBConv blocks (stages 1-7)
          [8] ConvNormActivation (features head, 320→1280)

        :param stages: list of child indices to unfreeze. e.g. [7, 8] for last two stages.
                       None = unfreeze all.
        """
        if stages is None:
            for param in self.features.parameters():
                param.requires_grad = True
            return

        for idx, child in enumerate(self.features):
            if idx in stages:
                for param in child.parameters():
                    param.requires_grad = True
        unfrozen_names = [str(i) for i in stages]
        print(f"Unfrozen features stages: {unfrozen_names}")

    def trainable_parameters(self):
        """Return parameters that require gradients (for optimizer setup)."""
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameters(self):
        """Return frozen parameters."""
        return [p for p in self.parameters() if not p.requires_grad]


def load_pretrained_image_weights(model, weight_path, device='cuda'):
    """
    Load pre-trained image-only weights into the EfficientNet feature extractor.

    The weight file is a full model state dict. Only keys starting with 'features.'
    (the convolutional backbone) are loaded; the classifier head is ignored.

    :param model: SensorFusionModel instance
    :param weight_path: path to the image-only checkpoint
    :param device: torch device
    """
    state_dict = torch.load(weight_path, map_location=device)

    # Filter for feature extractor keys only
    feature_keys = {k: v for k, v in state_dict.items() if k.startswith('features.')}

    if not feature_keys:
        print("WARNING: No 'features.*' keys found in pre-trained weights. "
              "Trying to load all keys that match...")
        # Fall back to keys that exist in the model
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items()
                   if k in model_dict and k.startswith('features.')}
        if matched:
            model_dict.update(matched)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(matched)} matching keys out of {len(state_dict)} in checkpoint.")
        else:
            print("ERROR: Could not match any pre-trained weights to model.features.")
        return

    # strict=False: only the backbone is populated
    model_dict = model.state_dict()
    model_dict.update(feature_keys)
    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(feature_keys)} feature extractor weights.")
    if missing:
        num_missing = len([k for k in missing if 'features' not in k])
        if num_missing > 0:
            print(f"  (Expected: {num_missing} keys from sensor branch / fusion head will be trained from scratch)")
    loaded_params = sum(v.numel() for v in feature_keys.values())
    print(f"  Total loaded parameters: {loaded_params:,}")


# ===========================================================================
# Block sequence model: TemporalEncoder + BlockSequenceModel
# ===========================================================================

class TemporalEncoder(nn.Module):
    """
    Temporal context module over per-block image features.
    Supports TCN (causal 1D CNN) and unidirectional GRU.
    Input:  (batch, seq_len, 512)
    Output: (batch, seq_len, 512)
    """

    def __init__(self, model_type="tcn", feat_dim=512):
        super().__init__()
        self.model_type = model_type

        if model_type == "tcn":
            self.tcn = nn.Sequential(
                nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=8, dilation=4),
            )
        elif model_type == "gru":
            self.proj = nn.Linear(feat_dim, feat_dim)
            self.gru = nn.GRU(feat_dim, feat_dim, num_layers=2, batch_first=True,
                              bidirectional=False)
        else:
            raise ValueError(f"Unknown temporal model: {model_type}")

    def forward(self, x):
        """
        :param x: (batch, seq_len, feat_dim)
        :return: (batch, seq_len, feat_dim)
        """
        if self.model_type == "tcn":
            x_t = x.transpose(1, 2)                  # (batch, feat_dim, seq_len)
            x_t = self.tcn(x_t)                       # dilated convs, symmetric padding
            x_t = x_t[..., :x.shape[1]]               # trim trailing outputs so step t sees only steps <= t
            return x_t.transpose(1, 2)                # (batch, seq_len, feat_dim)
        else:  # gru
            x = self.proj(x)
            out, _ = self.gru(x)
            return out


class BlockSequenceModel(nn.Module):
    """
    Predicts relative probe transforms between adjacent blocks.

    Input:  N blocks of 10 stacked frames each, plus per-frame sensor data.
    Output: (N-1) adjacent-block relative transforms in 6D rotation + translation format.

    Architecture:
      EfficientNet (10-in) per block → (batch, N, 1280)
      → Linear(1280, 512) → TemporalEncoder → (batch, N, 512)
      → Per-pair MLP heads → (batch, N-1, 9)
    """

    def __init__(self, efficientnet_features, image_feature_dim=1280,
                 temporal_model="tcn", feat_dim=512,
                 sensor_cnn_channels=0, sensor_inst_dim=0):
        """
        :param efficientnet_features: nn.Sequential, EfficientNet feature extractor (10-in)
        :param temporal_model: "tcn" or "gru"
        :param sensor_cnn_channels: number of sensor trend window channels (0 = no CNN branch)
        :param sensor_inst_dim: per-frame instantaneous sensor dimension (0 = no instantaneous branch)
        """
        super().__init__()
        self.features = efficientnet_features
        self.feat_dim = feat_dim

        # Image feature projection
        self.img_proj = nn.Sequential(
            nn.Linear(image_feature_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal encoder
        self.temporal = TemporalEncoder(temporal_model, feat_dim)

        # Sensor branches (per-frame processing)
        if sensor_cnn_channels > 0:
            self.sensor_cnn = SensorWindowCNN(in_channels=sensor_cnn_channels, hidden_dim=128)
        else:
            self.sensor_cnn = None

        if sensor_inst_dim > 0:
            self.sensor_inst_mlp = nn.Sequential(
                nn.Linear(sensor_inst_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
        else:
            self.sensor_inst_mlp = None

        # Pair head: one MLP shared across all adjacent block pairs
        head_in = feat_dim * 2          # feat_i + feat_{i+1}
        if self.sensor_cnn is not None:
            head_in += 128 * 2           # sensor_cnn_i + sensor_cnn_{i+1}
        if self.sensor_inst_mlp is not None:
            head_in += 64 * 2            # sensor_inst_i + sensor_inst_{i+1}
        self.pair_head = nn.Sequential(
            nn.Linear(head_in, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 9),     # 6D rotation + translation
        )

    def forward(self, frames, sensors=None, sensor_wins=None, win_lens=None):
        """
        :param frames: (batch, N, 10, H, W) — N blocks of 10 stacked frames
        :param sensors: (batch, N, 10, 14) — per-frame instantaneous sensor features
        :param sensor_wins: (batch, N, 10, win_size, C) — per-frame sensor trend windows
        :param win_lens: (batch, N, 10) — unused
        :return: (batch, N-1, 9) — 6D rotation + translation per adjacent block pair
        """
        batch_size, num_blocks, num_frames, H, W = frames.shape

        # ---- Image features: process each block independently ----
        frames_flat = frames.reshape(batch_size * num_blocks, num_frames, H, W)
        img_feat = self.features(frames_flat)                       # (B*N, 1280, H', W')
        img_feat = img_feat.mean(dim=[2, 3])                        # (B*N, 1280)
        img_feat = self.img_proj(img_feat)                          # (B*N, feat_dim)
        img_feat = img_feat.reshape(batch_size, num_blocks, self.feat_dim)  # (B, N, feat_dim)

        # ---- Temporal context ----
        img_feat = self.temporal(img_feat)                           # (B, N, feat_dim)

        # ---- Sensor features: per frame, then mean over the frames of each block ----
        sens_cnn_block = None
        sens_inst_block = None

        if self.sensor_cnn is not None and sensor_wins is not None:
            B, N_blk, N_frm, win_sz, C = sensor_wins.shape
            sw_flat = sensor_wins.reshape(B * N_blk * N_frm, win_sz, C)
            cnn_out = self.sensor_cnn(sw_flat)                      # (B*N_blk*N_frm, 128)
            cnn_out = cnn_out.reshape(B, N_blk, N_frm, 128)
            sens_cnn_block = cnn_out.mean(dim=2)                     # (B, N_blk, 128)

        if self.sensor_inst_mlp is not None and sensors is not None:
            B, N_blk, N_frm, C_inst = sensors.shape
            si_flat = sensors.reshape(B * N_blk * N_frm, C_inst)
            inst_out = self.sensor_inst_mlp(si_flat)                # (B*N_blk*N_frm, 64)
            inst_out = inst_out.reshape(B, N_blk, N_frm, 64)
            sens_inst_block = inst_out.mean(dim=2)                   # (B, N_blk, 64)

        # ---- Per-adjacent-pair prediction ----
        pairs = []
        for i in range(num_blocks - 1):
            parts = [img_feat[:, i, :], img_feat[:, i + 1, :]]     # (B, feat_dim) each
            if sens_cnn_block is not None:
                parts.extend([sens_cnn_block[:, i, :], sens_cnn_block[:, i + 1, :]])
            if sens_inst_block is not None:
                parts.extend([sens_inst_block[:, i, :], sens_inst_block[:, i + 1, :]])
            pair_input = torch.cat(parts, dim=1)                    # (B, head_in)
            delta = self.pair_head(pair_input)                      # (B, 9)
            pairs.append(delta)

        return torch.stack(pairs, dim=1)                             # (B, N-1, 9)
