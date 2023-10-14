from __future__ import annotations

import kornia
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from anomalib.models.fair.torch_model import ReconstructiveSubNetwork
from anomalib.models.fair.loss import FairLoss, MSGMSLoss


def mean_smoothing(amaps, kernel_size: int = 21):
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


__all__ = ["Fair"]


class Fair(AnomalyModule):
    def __init__(
            self,
            anomaly_source_path: str | None = None,
            in_channels: str = 3,
            out_channels: str = 3
    ) -> None:
        super(Fair, self).__init__()

        self.model = ReconstructiveSubNetwork(in_channels, out_channels)
        self.loss = FairLoss()
        self.augmenter = None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.

        input_image = batch["image"]
        gray_batch, gray_grayimage = self.augmenter.augment_batch(input_image, mode='train')
        gray_rec = self.model(gray_grayimage)
        # Compute loss
        loss = self.loss(gray_rec, gray_batch)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.

        msgms = MSGMSLoss()
        input_image = batch["image"]
        gray_batch, gray_grayimage = self.augmenter.augment_batch(input_image, mode='test')

        gray_rec = self.model(gray_grayimage)
        rec_lab = kornia.color.rgb_to_lab(gray_rec)
        ori_lab = kornia.color.rgb_to_lab(gray_batch)
        colordif = (ori_lab - rec_lab) * (ori_lab - rec_lab)
        colorresult = colordif[:, 1, :, :] + colordif[:, 2, :, :]
        colorresult = colorresult[None, :, :, :] * 0.0003

        out_map = msgms(gray_rec, gray_batch, as_loss=False) + colorresult
        out_mask_averaged = mean_smoothing(out_map, 21)
        out_mask_averaged = out_mask_averaged.detach().cpu().numpy()
        prediction = out_mask_averaged[0, 0, :, :]

        batch["anomaly_maps"] = prediction
        return batch
