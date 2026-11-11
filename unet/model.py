import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        in_ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.up_convs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        rev_features = list(reversed(features))
        for i, f in enumerate(rev_features):
            self.up_convs.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(f * 2, f))

        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Pad input to the nearest multiple of 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder)):
            x = self.up_convs[i](x)
            skip = skip_connections[i]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)

        x = self.out_conv(x)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x
