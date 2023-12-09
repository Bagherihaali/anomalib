import torch
import torch.nn.functional as F

from typing import Any
from torch import Tensor
from torch.optim import SGD, AdamW
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from anomalib.models.destseg.torch_model import DeSTSegModel
from anomalib.models.destseg.loss import cosine_similarity_loss, focal_loss, l1_loss
from anomalib.data.utils.augmenter import DetSegAugmenter


class DestSeg(AnomalyModule):
    def __init__(
            self,
            anomaly_source_path,
            lr_res=0.1,
            lr_seghead=0.01,
            lr_de_st=0.4,
            de_st_epochs=10,
            gamma=4,

    ) -> None:
        super().__init__()
        self.model = DeSTSegModel(dest=True, ed=True)

        self.gamma = gamma
        self.lr_res = lr_res
        self.lr_seghead = lr_seghead
        self.lr_de_st = lr_de_st
        self.de_st_epochs = de_st_epochs

        self.augmenter = DetSegAugmenter(anomaly_source_path)

        self.automatic_optimization = False

    def training_step(self, batch: dict[str, str | Tensor]) -> STEP_OUTPUT:

        de_st_optimizer, seg_optimizer = self.optimizers()
        de_st_optimizer.zero_grad()
        seg_optimizer.zero_grad()

        if self.global_step < self.de_st_epochs:
            self.model.student_net.train()
            self.model.segmentation_net.eval()
        else:
            self.model.student_net.eval()
            self.model.segmentation_net.train()

        img_aug, input_image, mask = self.augmenter.augment_batch(batch["image"])

        output_segmentation, output_de_st, output_de_st_list = self.model(
            img_aug, input_image
        )

        mask = F.interpolate(
            mask,
            size=output_segmentation.size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.where(
            mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
        )

        cosine_loss_val = cosine_similarity_loss(output_de_st_list)
        focal_loss_val = focal_loss(output_segmentation, mask, gamma=self.gamma)
        l1_loss_val = l1_loss(output_segmentation, mask)

        if self.current_epoch < self.de_st_epochs:
            loss = cosine_loss_val
            self.manual_backward(loss)
            de_st_optimizer.step()

        else:
            loss = focal_loss_val + l1_loss_val
            self.manual_backward(loss)
            seg_optimizer.step()

        self.log("cosine_loss", cosine_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)
        self.log("focal_loss", focal_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)
        self.log("l1_loss", l1_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)
        batch['visualization'] = {
            "mask": mask,
            'output_segmentation': output_segmentation,
            'output_de_st': output_de_st,
            'input_image': input_image,
            'img_aug': img_aug
        }

        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs  # These variables are not used.
        img = batch["image"]
        mask = batch["mask"].unsqueeze(1)

        self.model.segmentation_net.eval()
        self.model.student_net.eval()

        output_segmentation, output_de_st, output_de_st_list = self.model(img)

        output_segmentation = F.interpolate(
            output_segmentation,
            size=mask.size()[2:],
            mode="bilinear",
            align_corners=False,
        )

        cosine_loss_val = cosine_similarity_loss(output_de_st_list)
        focal_loss_val = focal_loss(output_segmentation, mask, gamma=self.gamma)
        l1_loss_val = l1_loss(output_segmentation, mask)

        batch["anomaly_maps"] = output_segmentation
        batch['visualization'] = {
            "mask": mask,
            'output_segmentation': output_segmentation,
            'output_de_st': output_de_st,
            'input_image': img,
        }

        self.log("val_cosine_loss", cosine_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)
        self.log("val_focal_loss", focal_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)
        self.log("val_l1_loss", l1_loss_val.item(), on_epoch=True, prog_bar=False, logger=True, on_step=False)

        return batch
