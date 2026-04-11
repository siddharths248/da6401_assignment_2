"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'.
        """
        super().__init__()

        self.eps = eps

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("reduction must be one of: none, mean, sum")

        self.reduction = reduction

    def _to_xyxy(self, boxes: torch.Tensor):
        
        x_c, y_c, w, h = boxes.unbind(dim=-1)

        xmin = x_c - w / 2
        ymin = y_c - h / 2
        xmax = x_c + w / 2
        ymax = y_c + h / 2

        return xmin, ymin, xmax, ymax

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss."""

        pxmin, pymin, pxmax, pymax = self._to_xyxy(pred_boxes)
        txmin, tymin, txmax, tymax = self._to_xyxy(target_boxes)

        inter_xmin = torch.max(pxmin, txmin)
        inter_ymin = torch.max(pymin, tymin)
        inter_xmax = torch.min(pxmax, txmax)
        inter_ymax = torch.min(pymax, tymax)

        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)

        inter_area = inter_w * inter_h

        pred_w = (pxmax - pxmin).clamp(min=0)
        pred_h = (pymax - pymin).clamp(min=0)
        target_w = (txmax - txmin).clamp(min=0)
        target_h = (tymax - tymin).clamp(min=0)

        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        union = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union  # [0, 1]

        loss = 1.0 - iou  # convert similarity → loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss