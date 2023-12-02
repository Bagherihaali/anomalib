import cv2
import pytorch_lightning as pl
import numpy as np

from pathlib import Path
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any
from anomalib.models.components import AnomalyModule

from anomalib.post_processing.visualizer import ImageGrid


class TrainingVisualizer(Callback):
    def __init__(self, image_save_path):
        self.image_save_path = Path(image_save_path)
        self.image_save_path.mkdir(parents=True, exist_ok=True)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT | None,
            batch: Any,
            batch_idx: int,
            *args, **kwargs
    ) -> None:
        if batch_idx == 0:
            img = TrainingVisualizer.batch_visualize(batch)

            save_path = self.image_save_path / 'validation'
            save_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path / f'{trainer.current_epoch}.png'), img)

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: AnomalyModule,
            outputs: STEP_OUTPUT | None,
            batch: Any,
            batch_idx: int,
    ) -> None:
        if batch_idx == 0:
            img = TrainingVisualizer.batch_visualize(batch)

            save_path = self.image_save_path / 'train'
            save_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path / f'{trainer.current_epoch}.png'), img)

    @staticmethod
    def batch_visualize(batch):
        batch_size = batch["image"].shape[0]
        visualisation = batch['visualisation']
        row_num = len(batch['visualisation'].keys())

        image_height, image_width = 64, 64
        margin_size = 30
        final_image = np.ones((margin_size + row_num * (image_height + margin_size),
                               margin_size + batch_size * (image_width + margin_size), 3), dtype=np.uint8) * 255

        for i, (key, value) in enumerate(visualisation.items()):
            row_caption = key
            for j, img in enumerate(value):
                if len(img.shape) == 2:
                    img = img.unsqueeze(0)

                img = img.detach().cpu().numpy().transpose(1, 2, 0) * 255
                img = np.uint8(img)
                if img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (image_height, image_width))

                image_with_margin = np.ones((image_height + margin_size,
                                             image_width + margin_size, 3), dtype=np.uint8) * 255
                image_with_margin[:image_height, :image_width] = img

                final_image[
                margin_size + i * (image_height + margin_size):margin_size + (i + 1) * (image_height + margin_size),
                margin_size + j * (image_width + margin_size): margin_size + (j + 1) * (
                        image_width + margin_size)] = image_with_margin

            cv2.putText(final_image, row_caption, (5 + margin_size, (i + 1) * (image_height + margin_size) - 10 + margin_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return final_image
