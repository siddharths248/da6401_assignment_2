"""Classification components
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        
        self.encoder = VGG11Encoder(in_channels)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_classes)
        )
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # : Implement forward pass.
        # raise NotImplementedError("Implement VGG11Classifier.forward")

        x = self.encoder(x)
        x = self.classifier(x)
        return x
