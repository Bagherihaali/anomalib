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
from PIL import Image

import albumentations as alb
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.utils.generators.perlin import random_2d_perlin
from anomalib.data.utils.perlin import rand_perlin_2d_np, detseg_perlin


def nextpow2(value):
    """Returns the smallest power of 2 greater than or equal to the input value."""
    return 2 ** (math.ceil(math.log(value, 2)))


def cut_patch(mb_img):  # for numpy
    c, h, w = mb_img.shape
    top = random.randint(0, round(h))
    bottom = top + random.randint(round(h * 0.3), round(h * 0.5))
    left = random.randint(0, round(w))
    right = left + random.randint(round(w * 0.3), round(w * 0.5))

    bottom = bottom % h
    right = right % w

    if top > bottom:
        top, bottom = bottom, top
    if left > right:
        left, right = right, left

    # Ensure even dimensions
    bottom = bottom if (bottom - top) % 2 == 0 else bottom - 1
    right = right if (right - left) % 2 == 0 else right - 1

    return mb_img[:, top:bottom, left:right]

    # h, w, c = mb_img.shape
    # top = random.randrange(0, round(h))
    # bottom = top + random.randrange(round(h * 0.3), round(h * 0.5))
    # left = random.randrange(0, round(w))
    # right = left + random.randrange(round(h * 0.3), round(h * 0.5))
    # if (bottom - top) % 2 == 1:
    #     bottom -= 1
    # if (right - left) % 2 == 1:
    #     right -= 1
    # return mb_img[top:bottom, left:right, :]


def paste_patch(img, patch):
    imgc, imgh, imgw = img.shape
    patchc, patchh, patchw = patch.shape

    angle = (torch.mul(torch.rand(1), 4 * math.pi) - 2 * math.pi).item()
    scale = 1  # You can change this if needed
    affinematrix = torch.tensor([
        [scale * math.cos(angle), -scale * math.sin(angle), 0],
        [scale * math.sin(angle), scale * math.cos(angle), 0]
    ], dtype=torch.float32)

    affinepatch = F.affine_grid(affinematrix.view(1, 2, 3), torch.Size((1, patchc, patchh, patchw)))
    affinepatch = F.grid_sample(patch.unsqueeze(0), affinepatch)
    affinepatch = affinepatch.squeeze()

    patch_h_position = random.randint(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randint(1, round(imgw) - round(patchw) - 1)

    pasteimg = img.clone()
    pasteimg[:, patch_h_position:patch_h_position + patchh, patch_w_position:patch_w_position + patchw] = affinepatch

    return pasteimg

    # imgh, imgw, imgc = img.shape
    # patchh, patchw, patchc = patch.shape
    # angle = random.randrange(-2 * round(math.pi), 2 * round(math.pi))
    # # scale = random.randrange(0, 1)
    # scale = 1
    # affinematrix = np.float32(
    #     [[scale * math.cos(angle), scale * -math.sin(angle), 0], [scale * math.sin(angle), scale * math.cos(angle), 0]])
    # affinepatch = cv2.warpAffine(patch, affinematrix, (patchw, patchh))
    # patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    # patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    # pasteimg = np.copy(img)
    # pasteimg[patch_h_position:patch_h_position + patchh,
    # patch_w_position:patch_w_position + patchw, :] = affinepatch
    # mask = np.zeros((img.shape[0], img.shape[1], 1))
    # mask[patch_h_position:patch_h_position + patchh, patch_w_position:patch_w_position + patchw] = 1
    #
    # return pasteimg, mask


def rgb_to_grayscale(image_tensor):
    # Assuming image_tensor has shape (C, H, W) where C is the number of channels

    # Apply the weighted sum to convert to grayscale
    grayscale_tensor = 0.299 * image_tensor[0, :, :] + 0.587 * image_tensor[1, :, :] + 0.114 * image_tensor[2, :, :]

    # Add a singleton dimension to represent the channel
    grayscale_tensor = grayscale_tensor.unsqueeze(0)

    return grayscale_tensor


def to_fair(auggray):
    auggray = auggray.squeeze()

    # FFT
    f = torch.fft.fft2(auggray)
    fshift = torch.fft.fftshift(f)

    # BHPF
    rows, cols = auggray.shape
    crow, ccol = rows // 2, cols // 2
    d = 30  # cutoff frequency
    n = 2  # BHPF order
    epsilon = 1e-6  # avoid dividing zero
    Y, X = torch.meshgrid(torch.arange(rows), torch.arange(cols))
    dist = torch.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
    mask = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
    fshift_filtered = fshift * mask

    # IFFT
    f_ishift = torch.fft.ifftshift(fshift_filtered)
    image_filtered = torch.fft.ifft2(f_ishift)
    auggray = torch.real(image_filtered).float()

    auggray = auggray.unsqueeze(0)

    return auggray


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

        perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale)  # nosec: B311
        perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale)  # nosec: B311

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
        if anomaly_source_path is not None:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))
        # self.image_augmenters = [
        #     iaa.PiecewiseAffine(nb_rows=(4), nb_cols=(4), scale=(0.02, 0.02)),
        #     iaa.ShearX((-15, 15), mode='edge'),
        #     iaa.WithPolarWarping(
        #         iaa.PiecewiseAffine(nb_rows=(2, 6), nb_cols=(2, 6), scale=(0.02, 0.02))
        #     ),
        # ]

        self.image_augmenters = [
            alb.PiecewiseAffine(scale=(0.02, 0.02), nb_rows=4, nb_cols=4, p=1.0),
            alb.IAAAffine(shear=(-15, 15), mode='edge', p=1.0),
            alb.ElasticTransform(alpha=100, sigma=10, alpha_affine=1, p=1,always_apply=True),
            alb.OpticalDistortion(distort_limit=.15, shift_limit=0.05, p=1, always_apply=True)
        ]

        # self.augmenters = [
        #     iaa.GammaContrast((0.5, 2.0),
        #                       per_channel=False
        #                       ),
        #     iaa.PiecewiseAffine(nb_rows=4, nb_cols=4, scale=(0.02, 0.02)),
        #     iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        #     iaa.pillike.EnhanceSharpness(),
        #     iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        #     iaa.Solarize(0.5, threshold=(32, 128)),
        #     iaa.Posterize(),
        #     iaa.Invert(),
        #     iaa.pillike.Autocontrast(),
        #     iaa.pillike.Equalize(),
        #     iaa.Affine(rotate=(-45, 45)),
        #     iaa.Affine(scale=(0.2, 0.7)),
        # ]

        self.augmenters = [
            alb.RandomGamma(gamma_limit=(0.5, 2.0), p=1.0),
            alb.PiecewiseAffine(scale=(0.02, 0.02), nb_rows=4, nb_cols=4, p=1.0),
            alb.RandomBrightnessContrast(brightness_limit=(-30, 30), contrast_limit=(0.8, 1.2), p=1.0),
            alb.Sharpen(always_apply=True, p=1.0),
            alb.HueSaturationValue(hue_shift_limit=(-50, 50), sat_shift_limit=(-50, 50), val_shift_limit=(-50, 50),
                                   p=1.0),
            alb.Solarize(threshold=(32, 128), p=0.5),
            alb.Posterize(always_apply=True, p=1.0),
            alb.InvertImg(always_apply=True, p=1.0),
            alb.RandomContrast(always_apply=True, p=1.0),
            alb.Equalize(always_apply=True, p=1.0),
            alb.Affine(rotate=(-45, 45), interpolation=3, always_apply=True, p=1.0),
            alb.Affine(scale=(0.2, 0.7), interpolation=3, always_apply=True, p=1.0),
        ]
        # self.rot = iaa.Sequential([iaa.Affine(rotate=(-15, 15))])
        self.rot = alb.Affine(rotate=(-1, 1), interpolation=3, always_apply=True, p=1.0)

    def augment_batch(self, batch: Tensor, mode: str) -> tuple[Tensor, Tensor]:
        device = batch.device
        batch = batch.cpu()
        batch_size, channels, height, width = batch.shape

        images = []
        auggrays = []
        for i in range(batch_size):
            if mode == 'train':
                anomaly_source_path = random.sample(self.anomaly_source_paths, 1)[0] if len(
                    self.anomaly_source_paths) > 0 else None
                image, auggray = self.transform_image(batch[i], anomaly_source_path)
                images.append(image)
                auggrays.append(auggray)
            else:
                image, auggray = self.transform_test_image(batch[i])
                images.append(image)
                auggrays.append(auggray)

        images = torch.from_numpy(np.stack(images)).to(device)
        auggray = torch.from_numpy(np.stack(auggrays)).to(device)

        return images, auggray

    def transform_test_image(self, image):

        imagegray = rgb_to_grayscale(image)
        imagegray = to_fair(imagegray)

        # image = np.array(image)
        # image = np.transpose(image, (1, 2, 0))
        #
        # imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # # fft
        # f = np.fft.fft2(imagegray)
        # fshift = np.fft.fftshift(f)
        #
        # # BHPF
        # rows, cols = imagegray.shape
        # crow, ccol = int(rows / 2), int(cols / 2)
        # d = 30  # cutoff frequency
        # n = 2  # BHPF order
        # epsilon = 1e-6  # avoid dividing zero
        # Y, X = np.ogrid[:rows, :cols]
        # dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        # maska = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        # fshift_filtered = fshift * maska
        #
        # # inverse fft
        # f_ishift = np.fft.ifftshift(fshift_filtered)
        # image_filtered = np.fft.ifft2(f_ishift)
        # imagegray = np.real(image_filtered).astype(np.float32)
        # imagegray = imagegray[:, :, None]
        # image = np.transpose(image, (2, 0, 1))
        # imagegray = np.transpose(imagegray, (2, 0, 1))

        return image, imagegray

    def transform_image(self, image, anomaly_source_path):

        do_aug_orig = torch.rand(1).item() > 0.5
        if do_aug_orig:
            # image = self.rot(image=image)
            angle = torch.randint(-15, 15 + 1, ()).item()
            image = TF.rotate(image, angle)

        what_anomaly = torch.rand(1).item()
        if what_anomaly > 0.8:
            augmented_image = self.augment_cutpaste(image)
        else:
            augmented_image = self.augment_image(image, anomaly_source_path)

        auggray = rgb_to_grayscale(augmented_image)

        auggray = to_fair(auggray)

        # image = np.array(image)
        # image = np.transpose(image, (1, 2, 0))
        #
        # do_aug_orig = torch.rand(1).numpy()[0] > 0.8
        # if do_aug_orig:
        #     image = self.rot(image=image)
        #
        # what_anomaly = torch.rand(1).numpy()[0]
        # if what_anomaly > 0.7:
        #     augmented_image, anomaly_mask = self.augment_cutpaste(image)
        # else:
        #     augmented_image, anomaly_mask = self.augment_image(image, anomaly_source_path)
        # auggray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
        #
        # # fft
        # f = np.fft.fft2(auggray)
        # fshift = np.fft.fftshift(f)
        #
        # # BHPF
        # rows, cols = auggray.shape
        # crow, ccol = int(rows / 2), int(cols / 2)
        # d = 30  # cutoff frequency
        # n = 2  # BHPF order
        # epsilon = 1e-6  # avoid dividing zero
        # Y, X = np.ogrid[:rows, :cols]
        # dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
        # mask = 1 / (1 + (d / (dist + epsilon)) ** (2 * n))
        # fshift_filtered = fshift * mask
        #
        # # ifft
        # f_ishift = np.fft.ifftshift(fshift_filtered)
        # image_filtered = np.fft.ifft2(f_ishift)
        # auggray = np.real(image_filtered).astype(np.float32)
        #
        # auggray = auggray[:, :, None]
        # image = np.transpose(image, (2, 0, 1))
        # auggray = np.transpose(auggray, (2, 0, 1))

        return image, auggray

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        # aug = iaa.Sequential([self.augmenters[aug_ind[0]],
        #                       self.augmenters[aug_ind[1]],
        #                       self.augmenters[aug_ind[2]]]
        #                      )
        aug = alb.Compose([self.augmenters[aug_ind[0]],
                           self.augmenters[aug_ind[1]],
                           self.augmenters[aug_ind[2]]]
                          )
        return aug

    def imageRandAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.image_augmenters)), 1, replace=False)
        # aug = iaa.Sequential([self.image_augmenters[aug_ind[0]]])
        aug = alb.Compose([self.image_augmenters[aug_ind[0]]])
        return aug

    def augment_image(self, image, anomaly_source_path):

        image = image.cpu().numpy().transpose(1, 2, 0)
        aug = self.randAugmenter()
        image_aug = self.imageRandAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        aug_source = torch.rand(1).numpy()[0]

        if aug_source > 0.00001:
            augmented_image = image_aug(image=image)['image']
            final_image = augmented_image.astype(np.float32)

        else:
            anomaly_source_img = cv2.imread(anomaly_source_path)
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(image.shape[0], image.shape[1]))

            # anomaly_augmented = aug(image=anomaly_source_img)
            # image = image_aug(image=image)

            anomaly_augmented = aug(image=anomaly_source_img)['image']

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
            # perlin_noise = self.rot(image=perlin_noise)
            perlin_noise = self.rot(image=perlin_noise)['image']
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_augmented.astype(np.float32) * perlin_thr / 255.0

            beta = torch.rand(1).numpy()[0] * 0.8

            anomaly_augmented_image = image * (1 - perlin_thr) + (
                    1 - beta) * img_thr + beta * image * perlin_thr

            final_image = anomaly_augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            final_image = msk * final_image + (1 - msk) * image

        return torch.tensor(final_image).permute(2, 0, 1)

    def augment_cutpaste(self, image):

        no_anomaly = torch.rand(1).item()

        if no_anomaly > 0.7:
            image = image.float()
            return image, torch.zeros((image.shape[0], image.shape[1], 1), dtype=torch.float32)
        else:
            patch = cut_patch(image)  # Assuming 'cut_patch' is a PyTorch-compatible function
            augmented_image = paste_patch(image, patch)  # Assuming 'paste_patch' is a PyTorch-compatible function

            augmented_image = augmented_image.float()

            return augmented_image

        # no_anomaly = torch.rand(1).numpy()[0]
        # if no_anomaly > 0.5:
        #     image = image.astype(np.float32)
        #     return image, np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
        # else:
        #     patch = cut_patch(image)
        #     augmented_image, msk = paste_patch(image, patch)
        #     msk = msk.astype(np.float32)
        #     augmented_image = augmented_image.astype(np.float32)
        #     has_anomaly = 1.0
        #     if np.sum(msk) == 0:
        #         has_anomaly = 0.0
        #     return augmented_image, msk


class DetSegAugmenter:
    def __init__(self,
                 dtd_dir=None,
                 rotate_90=False,
                 random_rotate=0):
        self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
        self.rotate_90 = rotate_90
        self.random_rotate = random_rotate

    def augment_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        device = batch.device
        batch = batch.cpu()
        batch_size, channels, height, width = batch.shape

        aug_images = []
        images = []
        masks = []
        for image in batch:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize((height, width), Image.BILINEAR)

            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = iaa.Affine(rotate=degree),
                # image = image.rotate(
                #     degree, fillcolor=fill_color, resample=Image.BILINEAR
                # )
            # random_rotate
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = iaa.Affine(rotate=degree),

                # image = image.rotate(
                #     degree, fillcolor=fill_color, resample=Image.BILINEAR
                # )

            # perlin_noise implementation
            aug_image, aug_mask = detseg_perlin(image.permute(1, 2, 0), dtd_image, aug_prob=.6)
            aug_image = aug_image.transpose(2, 0, 1)

            images.append(image)
            aug_images.append(aug_image)
            masks.append(aug_mask)

        images = torch.from_numpy(np.stack(images)).to(device)
        aug_images = torch.from_numpy(np.stack(aug_images)).to(device)
        masks = torch.from_numpy(np.stack(masks)).to(device)

        return aug_images, images, masks
