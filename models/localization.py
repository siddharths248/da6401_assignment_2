"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        
        self.encoder = VGG11Encoder(in_channels)


        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            CustomDropout(dropout_p),

            nn.Linear(512, 128),
            nn.ReLU(True),
            CustomDropout(dropout_p),

            nn.Linear(128, 4)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # : Implement forward pass.
        # raise NotImplementedError("Implement VGG11Localizer.forward")

        x = self.encoder(x)
        x = self.pool(x)
        x = self.regressor(x)
        x = torch.sigmoid(x) * 224
        return x
