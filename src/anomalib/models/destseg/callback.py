import cv2
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any
from anomalib.models.components import AnomalyModule

from anomalib.post_processing.visualizer import ImageGrid


class ValidationVisualizer(Callback):
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
        img = ValidationVisualizer.batch_visualize(batch)

        save_path = self.image_save_path / 'validation'
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / f'{batch_idx}.png'), img)

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: AnomalyModule,
            outputs: STEP_OUTPUT | None,
            batch: Any,
            batch_idx: int,
    ) -> None:
        img = ValidationVisualizer.batch_visualize(batch)

        save_path = self.image_save_path / 'train'
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / f'{batch_idx}.png'), img)

    @staticmethod
    def batch_visualize(batch):
        batch_size = batch["image"].shape[0]
        visualisation = batch['visualisation']
        row_num = len(batch['visualisation'].keys())
        fig, axs = plt.subplots(row_num, batch_size, figsize=(8, 6), gridspec_kw={'wspace': 0.1, 'hspace': 0.5})

        for i, (key, value) in enumerate(visualisation.items()):
            for j, img in enumerate(value):
                if len(img.shape) == 2:
                    img = img.unsqueeze(0)
                ax = axs[i, j]
                ax.imshow(img.detach().cpu().numpy().transpose(1, 2, 0) * 255)
                ax.axis('off')

                if j == 0:
                    ax.text(-0.1, 0.5, f'{key}', va='center', ha='right', rotation='vertical',
                            transform=ax.transAxes, fontsize='x-small')
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return img
