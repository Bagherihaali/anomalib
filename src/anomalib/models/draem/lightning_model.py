"""DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection.

Paper https://arxiv.org/abs/2108.07610
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from anomalib.data.utils import Augmenter
from anomalib.models.components import AnomalyModule
from anomalib.models.draem.loss import DraemLoss
from anomalib.models.draem.torch_model import DraemModel

__all__ = ["Draem", "DraemLightning"]


class Draem(AnomalyModule):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        anomaly_source_path (str | None): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty.
    """

    def __init__(
            self,
            enable_sspcab: bool = False,
            sspcab_lambda: float = 0.1,
            focal_alpha=1,
            focal_gamma=2,
            anomaly_source_path: str | None = None,
            beta: float | tuple[float, float] = (0.1, 1.0),
            perlin_scale: int | tuple[int, int] = (0, 6),
            rec_base_width=64,
            disc_base_width=16,
            l1_loss: bool = False,
            p_anomalous: float = 0.5,
            freeze_disc=False,

    ) -> None:
        super().__init__()

        self.augmenter = Augmenter(anomaly_source_path, beta=beta, perlin_scale=perlin_scale, p_anomalous=p_anomalous)
        self.model = DraemModel(sspcab=enable_sspcab, rec_base_width=rec_base_width, disc_base_width=disc_base_width,
                                freeze_disc=freeze_disc)
        self.loss = DraemLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma, l1_loss=l1_loss)
        self.sspcab = enable_sspcab
        self.removable_handles = {}

        if self.sspcab:
            self.sspcab_activations: dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda

    def setup_sspcab(self) -> None:
        """Prepare the model for the SSPCAB training step by adding forward hooks for the SSPCAB layer activations."""

        def get_activation(name: str) -> Callable:
            """Retrieves the activations.

            Args:
                name (str): Identifier for the retrieved activations.
            """

            def hook(_, __, output: Tensor) -> None:
                """Hook for retrieving the activations."""
                self.sspcab_activations[name] = output

            return hook

        self.removable_handles['input'] = self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(
            get_activation("input"))
        self.removable_handles['output'] = self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(
            get_activation("output"))

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Training Step of DRAEM.

        Feeds the original image and the simulated anomaly
        image through the network and computes the training loss.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            Loss dictionary
        """
        del args, kwargs  # These variables are not used.

        batch["image"] = batch["image"][:, 1, :, :].unsqueeze(1)  # make images monochrome

        input_image = batch["image"]

        # Apply corruption to input image
        augmented_image, anomaly_mask = self.augmenter.augment_batch(input_image)
        # Generate model prediction
        reconstruction, prediction = self.model(augmented_image)
        # Compute loss
        loss, loss_list = self.loss(input_image, reconstruction, anomaly_mask, prediction)

        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"], self.sspcab_activations["output"]
            )
        self.log("l_loss", loss_list[0].item(), on_epoch=True, prog_bar=False, logger=True)
        self.log("ssim_loss", loss_list[1].item(), on_epoch=True, prog_bar=False, logger=True)
        self.log("focal_loss", loss_list[2].item(), on_epoch=True, prog_bar=False, logger=True)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        batch['visualization'] = {
            "mask": anomaly_mask,
            'input_image': input_image,
            'augmented_image': augmented_image,
            'prediction': torch.softmax(prediction, dim=1)[:, 1, ...],
            'reconstruction': reconstruction
        }

        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation step of DRAEM. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch (dict[str, str | Tensor]): Batch of input images

        Returns:
            Dictionary to which predicted anomaly maps have been added.
        """
        del args, kwargs  # These variables are not used.

        batch["image"] = batch["image"][:, 1, :, :].unsqueeze(1)  # make images monochrome

        # prediction = self.model(batch["image"])

        prediction, reconstruction = self.model(batch["image"])
        # Compute loss
        loss, loss_list = self.loss(batch["image"], reconstruction, anomaly_mask=None, prediction=None)

        self.log("val_l_loss", loss_list[0].item(), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_ssim_loss", loss_list[1].item(), on_epoch=True, prog_bar=False, logger=True)

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)

        batch["anomaly_maps"] = prediction
        batch['visualization'] = {
            'input_image': batch["image"],
            'anomaly_maps': prediction,
        }
        return batch


class DraemLightning(Draem):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        hparams (DictConfig | ListConfig): Model parameters
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        # beta in config can be either float or sequence
        beta = hparams.model.beta
        # if sequence - change to tuple[float, float]
        if isinstance(beta, ListConfig):
            beta = tuple(beta)

        super().__init__(
            enable_sspcab=hparams.model.enable_sspcab,
            sspcab_lambda=hparams.model.sspcab_lambda,
            anomaly_source_path=hparams.model.anomaly_source_path,
            beta=beta,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[EarlyStopping]:
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        callbacks = []
        if "early_stopping" in self.hparams.model:
            early_stopping = EarlyStopping(
                monitor=self.hparams.model.early_stopping.metric,
                patience=self.hparams.model.early_stopping.patience,
                mode=self.hparams.model.early_stopping.mode,
            )
            callbacks.append(early_stopping)

        return callbacks

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        """Configure the Adam optimizer."""
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.model.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)
        return [optimizer], [scheduler]
