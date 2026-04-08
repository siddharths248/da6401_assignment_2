"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn



def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        self.block1 = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            conv_block(64,128),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            conv_block(256, 512),
            conv_block(512, 512),
            nn.MaxPool2d(2)
        )

        self.block5 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512),
            nn.MaxPool2d(2)
        )



    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        # raise NotImplementedError("Implement VGG11Encoder.forward")
        
        features = {}

        x = self.block1(x)
        features["block1"] = x

        x = self.block2(x)
        features["block2"] = x

        x = self.block3(x)
        features["block3"] = x

        x = self.block4(x)
        features["block4"] = x

        x = self.block5(x)
        features["block5"] = x

        if return_features:
            return x,features
        
        return x

    


class VGG11():

    def __init__(self, num_classes: int=37, in_channels: int = 3):
        
        super.__init__()

        self.encoder = VGG11Encoder(in_channels)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x