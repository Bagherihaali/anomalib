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


def rgb_to_grayscale(images):
    # Apply luminance formula
    grayscale_images = images[:, 0, :, :] * 0.299 + images[:, 1, :, :] * 0.587 + images[:, 2, :, :] * 0.114

    # Expand dimensions to have a single channel
    grayscale_images = grayscale_images.unsqueeze(1)

    return grayscale_images


class Fair(AnomalyModule):
    def __init__(
            self,
            anomaly_source_path: str | None = None,
            in_channels: int = 1,
            out_channels: int = 3,
            base_width: int = 128
    ) -> None:
        super(Fair, self).__init__()

        self.model = FairModel(in_channels, out_channels, base_width)
        self.loss = FairLoss()
        self.augmenter = FairAugmenter(anomaly_source_path)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.

        input_image = batch["image"]
        _, gray_grayimage = self.augmenter.augment_batch(input_image, mode='train')
        gray_rec = self.model(gray_grayimage)

        # Compute loss in gray scale mode
        # loss = self.loss(rgb_to_grayscale(gray_rec), rgb_to_grayscale(input_image))

        # Compute loss
        if self.out_channels == 1:
            loss = self.loss(gray_rec, rgb_to_grayscale(input_image))
        else:
            loss = self.loss(gray_rec, input_image)

        self.log("l2_loss", self.loss.l2_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)
        self.log("ssim_loss", self.loss.ssim_los_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, on_step=False)
        batch['visualization'] = {
            "mask": batch["mask"],
            'input_image': input_image,
            'gray_grayimage': gray_grayimage,
            'gray_rec': gray_rec,
        }

        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.

        msgms = MSGMSLoss()
        input_image = batch["image"]
        _, gray_grayimage = self.augmenter.augment_batch(input_image, mode='test')

        gray_rec = self.model(gray_grayimage)
        prediction = []
        for i in range(gray_rec.shape[0]):
            if self.out_channels == 1:
                rec_rgb = kornia.color.grayscale_to_rgb(gray_rec[i].unsqueeze(0))
                rec_lab = kornia.color.rgb_to_lab(rec_rgb)
            else:
                rec_lab = kornia.color.rgb_to_lab(gray_rec[i].unsqueeze(0))
            ori_lab = kornia.color.rgb_to_lab(input_image[i].unsqueeze(0))

            colordif = (ori_lab - rec_lab) * (ori_lab - rec_lab)
            colorresult = colordif[:, 1, :, :] + colordif[:, 2, :, :]
            colorresult = colorresult[None, :, :, :] * 0.0003

            if self.out_channels == 1:
                out_map = msgms(rec_rgb, input_image[i].unsqueeze(0), as_loss=False) + colorresult
            else:
                out_map = msgms(gray_rec[i].unsqueeze(0), input_image[i].unsqueeze(0), as_loss=False) + colorresult

            out_mask_averaged = mean_smoothing(out_map, 21)
            pred = out_mask_averaged[0, 0, :, :]
            prediction.append(pred)

        batch_prediction = torch.stack(prediction)

        batch["anomaly_maps"] = batch_prediction
        batch['visualization'] = {
            "mask": batch["mask"],
            'input_image': input_image,
            'gray_grayimage': gray_grayimage,
            'gray_rec': gray_rec,
            'anomaly_maps': batch_prediction
        }
        return batch
