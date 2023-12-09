"""Loss function for the FAIR model implementation."""
import torch
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from torch.nn.modules.loss import MSELoss
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2, asloss=True):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size,
                                 size_average=self.size_average)
        if asloss:
            return 1.0 - s_score
        else:
            return 1.0 - ssim_map


class FairLoss(nn.Module):
    def __init__(self) -> None:
        super(FairLoss, self).__init__()

        self.l2_loss = MSELoss()
        self.ssim_los = SSIM()

        self.l2_loss_val = 0
        self.ssim_los_val = 0

    def forward(self, input_image: Tensor, reconstruction: Tensor):
        self.l2_loss_val = self.l2_loss(input_image, reconstruction)
        self.ssim_los_val = self.ssim_los(input_image, reconstruction)

        return self.l2_loss_val + self.ssim_los_val


class MSGMSLoss(nn.Module):
    def __init__(self, num_scales: int = 2, in_channels: int = 3) -> None:

        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.prewitt_x, self.prewitt_y = self._create_prewitt_kernel()

    def forward(self, img1: Tensor, img2: Tensor, as_loss: bool = True) -> Tensor:

        if not self.prewitt_x.is_cuda or not self.prewitt_y.is_cuda:
            self.prewitt_x = self.prewitt_x.to(img1.device)
            self.prewitt_y = self.prewitt_y.to(img1.device)

        b, c, h, w = img1.shape
        msgms_map = 0
        for scale in range(self.num_scales):

            if scale > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)

            gms_map = self._gms(img1, img2)
            msgms_map += F.interpolate(gms_map, size=(h, w), mode="bilinear", align_corners=False)

        if as_loss:
            return torch.mean(1 - msgms_map / self.num_scales)
        else:
            return torch.mean(1 - msgms_map / self.num_scales, axis=1).unsqueeze(1)

    def _gms(self, img1: Tensor, img2: Tensor) -> Tensor:

        gm1_x = F.conv2d(img1, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm1_y = F.conv2d(img1, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm1 = torch.sqrt(gm1_x ** 2 + gm1_y ** 2 + 1e-12)

        gm2_x = F.conv2d(img2, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm2_y = F.conv2d(img2, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm2 = torch.sqrt(gm2_x ** 2 + gm2_y ** 2 + 1e-12)

        # Constant c from the following paper. https://arxiv.org/pdf/1308.3052.pdf
        c = 0.0026
        numerator = 2 * gm1 * gm2 + c
        denominator = gm1 ** 2 + gm2 ** 2 + c
        return numerator / (denominator + 1e-12)

    def _create_prewitt_kernel(self) -> Tuple[Tensor, Tensor]:

        prewitt_x = torch.Tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_x = prewitt_x.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        prewitt_y = torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_y = prewitt_y.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        return (prewitt_x, prewitt_y)
