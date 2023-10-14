"""Augmenter module to generates out-of-distribution samples for the DRAEM implementation."""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import glob
import math
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.utils.generators.perlin import random_2d_perlin
from anomalib.models.fair.perlin import rand_perlin_2d_np


def nextpow2(value):
    """Returns the smallest power of 2 greater than or equal to the input value."""
    return 2 ** (math.ceil(math.log(value, 2)))


def cut_patch(mb_img):  # for numpy
    h, w, c = mb_img.shape
    top = random.randrange(0, round(h))
    bottom = top + random.randrange(round(h * 0.8), round(h * 0.9))
    left = random.randrange(0, round(w))
    right = left + random.randrange(round(h * 0.8), round(h * 0.9))
    if (bottom - top) % 2 == 1:
        bottom -= 1
    if (right - left) % 2 == 1:
        right -= 1
    return mb_img[top:bottom, left:right, :]


def paste_patch(img, patch):
    imgh, imgw, imgc = img.shape
    patchh, patchw, patchc = patch.shape
    angle = random.randrange(-2 * round(math.pi), 2 * round(math.pi))
    # scale = random.randrange(0, 1)
    scale = 1
    affinematrix = np.float32(
        [[scale * math.cos(angle), scale * -math.sin(angle), 0], [scale * math.sin(angle), scale * math.cos(angle), 0]])
    affinepatch = cv2.warpAffine(patch, affinematrix, (patchw, patchh))
    patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    pasteimg = np.copy(img)
    pasteimg[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw, :] = affinepatch
    mask = np.zeros((img.shape[0], img.shape[1], 1))
    mask[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw] = 1
    return pasteimg, mask


class Augmenter:
    """Class that generates noisy augmentations of input images.

    Args:
        anomaly_source_path (str | None): Path to a folder of images that will be used as source of the anomalous
        noise. If not specified, random noise will be used instead.
        p_anomalous (float): Probability that the anomalous perturbation will be applied to a given image.
        beta (float): Parameter that determines the opacity of the noise mask.
    """

    def __init__(
            self,
            anomaly_source_path: str | None = None,
            p_anomalous: float = 0.5,
            beta: float | tuple[float, float] = (0.2, 1.0),
    ):
        self.p_anomalous = p_anomalous
        self.beta = beta

        self.anomaly_source_paths = []
        if anomaly_source_path is not None:
            for img_ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(glob.glob(anomaly_source_path + "/**/*" + img_ext, recursive=True))

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def rand_augmenter(self) -> iaa.Sequential:
        """Selects 3 random transforms that will be applied to the anomaly source images.

        Returns:
            A selection of 3 transforms.
        """
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])
        return aug

    def generate_perturbation(
            self, height: int, width: int, anomaly_source_path: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate an image containing a random anomalous perturbation using a source image.

        Args:
            height (int): height of the generated image.
            width: (int): width of the generated image.
            anomaly_source_path (str | None): Path to an image file. If not provided, random noise will be used
            instead.

        Returns:
            Image containing a random anomalous perturbation, and the corresponding ground truth anomaly mask.
        """
        # Generate random perlin noise
        perlin_scale = 6
        min_perlin_scale = 0

        perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale)

        perlin_noise = random_2d_perlin((nextpow2(height), nextpow2(width)), (perlin_scalex, perlin_scaley))[
                       :height, :width
                       ]
        perlin_noise = self.rot(image=perlin_noise)

        # Create mask from perlin noise
        mask = np.where(perlin_noise > 0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        # Load anomaly source image
        if anomaly_source_path:
            anomaly_source_img = cv2.imread(anomaly_source_path)
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(width, height))
        else:  # if no anomaly source is specified, we use the perlin noise as anomalous source
            anomaly_source_img = np.expand_dims(perlin_noise, 2).repeat(3, 2)
            anomaly_source_img = (anomaly_source_img * 255).astype(np.uint8)

        # Augment anomaly source image
        aug = self.rand_augmenter()
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Create anomalous perturbation that we will apply to the image
        perturbation = anomaly_img_augmented.astype(np.float32) * mask / 255.0

        return perturbation, mask

    def augment_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Generate anomalous augmentations for a batch of input images.

        Args:
            batch (Tensor): Batch of input images

        Returns:
            - Augmented image to which anomalous perturbations have been added.
            - Ground truth masks corresponding to the anomalous perturbations.
        """
        batch_size, channels, height, width = batch.shape

        # Collect perturbations
        perturbations_list = []
        masks_list = []
        for _ in range(batch_size):
            if torch.rand(1) > self.p_anomalous:  # include normal samples
                perturbations_list.append(torch.zeros((channels, height, width)))
                masks_list.append(torch.zeros((1, height, width)))
            else:
                anomaly_source_path = (
                    random.sample(self.anomaly_source_paths, 1)[0] if len(self.anomaly_source_paths) > 0 else None
                )
                perturbation, mask = self.generate_perturbation(height, width, anomaly_source_path)
                perturbations_list.append(Tensor(perturbation).permute((2, 0, 1)))
                masks_list.append(Tensor(mask).permute((2, 0, 1)))

        perturbations = torch.stack(perturbations_list).to(batch.device)
        masks = torch.stack(masks_list).to(batch.device)

        # Apply perturbations batch wise
        if isinstance(self.beta, float):
            beta = self.beta
        elif isinstance(self.beta, tuple):
            beta = torch.rand(batch_size) * (self.beta[1] - self.beta[0]) + self.beta[0]
            beta = beta.view(batch_size, 1, 1, 1).expand_as(batch).to(batch.device)  # type: ignore
        else:
            raise ValueError("Beta must be either float or tuple of floats")

        augmented_batch = batch * (1 - masks) + (beta) * perturbations + (1 - beta) * batch * (masks)

        return augmented_batch, masks


class FairAugmenter:
    def __init__(
            self,
            anomaly_source_path: str | None = None,
    ) -> None:
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))
        self.image_augmenters = [
            iaa.PiecewiseAffine(nb_rows=(4), nb_cols=(4), scale=(0.02, 0.02)),
            iaa.ShearX((-10, 10), mode='edge'),
            # iaa.WithPolarWarping(
            #     iaa.PiecewiseAffine(nb_rows=(2,6),nb_cols=(2,6),scale=(0.01, 0.01))
            # ),
        ]
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0),
                              per_channel=False
                              ),
            iaa.PiecewiseAffine(nb_rows=4, nb_cols=4, scale=(0.02, 0.02)),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.Affine(scale=(0.5, 1)),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Affine(scale=(0.2, 0.7)),
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def augment_batch(self, batch: Tensor, mode: str) -> tuple[Tensor, Tensor]:
        batch_size, channels, height, width = batch.shape

        images = []
        auggrays = []
        for i in range(batch_size):
            if mode == 'train':
                anomaly_source_paths = random.sample(self.anomaly_source_paths, 1)[0] if len(
                    self.anomaly_source_paths) > 0 else None
                image, auggray = self.transform_image(batch[i], anomaly_source_paths)
                images.append(image)
                auggrays.append(auggrays)
            else:
                image, auggray = self.transform_test_image(batch[i], anomaly_source_paths)
                images.append(image)
                auggrays.append(auggrays)

        images = torch.stack(images).to(batch.device)
        auggray = torch.stack(auggrays).to(batch.device)

        return images, auggray

    def transform_test_image(self, image, anomaly_source_path):

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32) / 255.0
        imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # fft
        f = np.fft.fft2(imagegray)
        fshift = np.fft.fftshift(f)

        # BHPF
        rows, cols = imagegray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        d = 30  # cutoff frequency
        n = 2  # BHPF order
        epsilon = 1e-6  # avoid dividing zero
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        maska = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        fshift_filtered = fshift * maska

        # inverse fft
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        imagegray = np.real(image_filtered).astype(np.float32)
        imagegray = imagegray[:, :, None]
        image = np.transpose(image, (2, 0, 1))
        imagegray = np.transpose(imagegray, (2, 0, 1))

        return image, imagegray

    def transform_image(self, image, anomaly_source_path):

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            pass
            # image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        what_anomaly = torch.rand(1).numpy()[0]
        if what_anomaly > 0.7:
            augmented_image, anomaly_mask = self.augment_cutpaste(image)
        else:
            augmented_image, anomaly_mask = self.augment_image(image, anomaly_source_path)
        auggray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

        # fft
        f = np.fft.fft2(auggray)
        fshift = np.fft.fftshift(f)

        # BHPF
        rows, cols = auggray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        d = 30  # cutoff frequency
        n = 2  # BHPF order
        epsilon = 1e-6  # avoid dividing zero
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        mask = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        fshift_filtered = fshift * mask

        # ifft
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.fft.ifft2(f_ishift)
        auggray = np.real(image_filtered).astype(np.float32)

        auggray = auggray[:, :, None]
        image = np.transpose(image, (2, 0, 1))
        # augmented_image = np.transpose(augmented_image, (2, 0, 1))
        auggray = np.transpose(auggray, (2, 0, 1))
        # anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, auggray

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def imageRandAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.image_augmenters)), 1, replace=False)
        aug = iaa.Sequential([self.image_augmenters[aug_ind[0]]])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        image_aug = self.imageRandAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        image = image_aug(image=image)

        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.8:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def augment_cutpaste(self, image):
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32), np.array([0.0],
                                                                                                    dtype=np.float32)
        else:
            patch = cut_patch(image)
            augmented_image, msk = paste_patch(image, patch)
            msk = msk.astype(np.float32)
            augmented_image = augmented_image.astype(np.float32)
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)
