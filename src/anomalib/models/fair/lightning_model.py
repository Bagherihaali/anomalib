from __future__ import annotations

import kornia
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from anomalib.models.fair.torch_model import FairModel
from anomalib.models.fair.loss import FairLoss, MSGMSLoss
from anomalib.data.utils import FairAugmenter


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

        self.model = FairModel(in_channels, out_channels)
        self.loss = FairLoss()
        self.augmenter = FairAugmenter(anomaly_source_path)

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

        prediction = []
        for i in range(gray_rec.shape[0]):
            rec_lab = kornia.color.rgb_to_lab(gray_rec[i].unsqueeze(0))
            ori_lab = kornia.color.rgb_to_lab(gray_batch[i].unsqueeze(0))
            colordif = (ori_lab - rec_lab) * (ori_lab - rec_lab)
            colorresult = colordif[:, 1, :, :] + colordif[:, 2, :, :]
            colorresult = colorresult[None, :, :, :] * 0.0003

            out_map = msgms(gray_rec[i].unsqueeze(0), gray_batch[i].unsqueeze(0), as_loss=False) + colorresult
            out_mask_averaged = mean_smoothing(out_map, 21)
            pred = out_mask_averaged[0, 0, :, :]
            prediction.append(pred)

        batch_prediction = torch.stack(prediction)
        batch["anomaly_maps"] = batch_prediction
        return batch
