"""Unified multi-task model
"""

import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth"
    ):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels)

        classifier = VGG11Classifier(num_classes=num_breeds)
        classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

        localizer = VGG11Localizer()
        localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))

        unet = VGG11UNet(num_classes=seg_classes)
        unet.load_state_dict(torch.load(unet_path, map_location="cpu"))


        self.classifier_head = classifier.classifier

        self.localizer_pool = localizer.pool
        self.localizer_head = localizer.regressor

        self.up5 = unet.up5
        self.up4 = unet.up4
        self.up3 = unet.up3
        self.up2 = unet.up2
        self.up1 = unet.up1
        self.seg_head = unet.head

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            dict with:
            - 'classification': [B, num_breeds]
            - 'localization': [B, 4]
            - 'segmentation': [B, seg_classes, H, W]
        """

        bottleneck, features = self.encoder(x, return_features=True)

        cls_feat = nn.functional.adaptive_avg_pool2d(bottleneck, (7, 7))
        cls_feat = torch.flatten(cls_feat, 1)
        cls_out = self.classifier_head(cls_feat)

        loc_feat = self.localizer_pool(bottleneck)
        loc_feat = self.localizer_head(loc_feat)
        loc_out = torch.relu(loc_feat)
        loc_out = torch.clamp(loc_out, max=224.0)

        seg = self.up5(bottleneck, features["block4"])
        seg = self.up4(seg, features["block3"])
        seg = self.up3(seg, features["block2"])
        seg = self.up2(seg, features["block1"])
        seg = self.up1(seg)  
        seg_out = self.seg_head(seg)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out
        }