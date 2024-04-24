"""Loss function for the DRAEM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from kornia.losses import FocalLoss, SSIMLoss
from torch import Tensor, nn


class DraemLoss(nn.Module):
    """Overall loss function of the DRAEM model.

    The total loss consists of the sum of the L2 loss and Focal loss between the reconstructed image and the input
    image, and the Structural Similarity loss between the predicted and GT anomaly masks.
    """

    def __init__(self, focal_alpha=1, focal_gamma=2, l1_loss=False) -> None:
        super().__init__()

        if not l1_loss:
            self.l_loss = nn.modules.loss.MSELoss()
        else:
            self.l_loss = nn.modules.loss.L1Loss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(self, input_image: Tensor, reconstruction: Tensor, anomaly_mask: Tensor, prediction: Tensor) -> Tensor:
        """Compute the loss over a batch for the DRAEM model."""
        l_loss_val = self.l_loss(reconstruction, input_image)
        focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image)
        return l_loss_val + ssim_loss_val + focal_loss_val, [l_loss_val, ssim_loss_val, focal_loss_val]
