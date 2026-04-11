"""Segmentation model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            conv_block(out_channels + skip_channels, out_channels),
            conv_block(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels)

        # Decoder
        self.up5 = UpBlock(512, 512, 512)  # 7→14 (skip block4)
        self.up4 = UpBlock(512, 256, 256)  # 14→28 (skip block3)
        self.up3 = UpBlock(256, 128, 128)  # 28→56 (skip block2)
        self.up2 = UpBlock(128, 64, 64)    # 56→112 (skip block1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 112→224

        self.head = nn.Sequential(
            conv_block(64, 64),
            CustomDropout(dropout_p),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bottleneck, features = self.encoder(x, return_features=True)

        x = self.up5(bottleneck, features["block4"])  # 7→14
        x = self.up4(x, features["block3"])           # 14→28
        x = self.up3(x, features["block2"])           # 28→56
        x = self.up2(x, features["block1"])           # 56→112

        x = self.up1(x)  

        x = self.head(x)

        return x