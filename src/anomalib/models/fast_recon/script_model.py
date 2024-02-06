import numpy as np
import torch

from torchvision.models import Wide_ResNet50_2_Weights
from torchvision.transforms import Resize, Normalize
from torch import Tensor, nn
from torch.nn import functional as F
from collections import OrderedDict
from typing import Tuple
from abc import ABC
from anomalib.models.fast_recon.torch_model import embedding_concat


class DynamicBufferModule(ABC, nn.Module):
    """Torch module that allows loading variables from the state dict even in the case of shape mismatch."""

    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """Get attribute of the tensor given the name.

        Args:
            attribute_name (str): Name of the tensor

        Raises:
            ValueError: `attribute_name` is not a torch Tensor

        Returns:
            Tensor: Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, Tensor):
            return attribute

        raise ValueError(f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """Resizes the local buffers to match those stored in the state dict.

        Overrides method from parent class.

        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args:
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix):].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)

        super()._load_from_state_dict(state_dict, prefix, *args)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        return enc3, enc4

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class FastReconModelScript(DynamicBufferModule, nn.Module):
    def __init__(
            self,
            input_size: tuple[int, int] = [1024, 1024],
            feature_extractor: UNet = None,
            lambda_value: int = 2,
            s: int = 2
    ):
        super().__init__()
        self.input_size = input_size

        self.lambda_value = lambda_value
        self.feature_extractor = feature_extractor
        self.s = s
        self.register_buffer("Sc", Tensor())
        self.register_buffer("mu", Tensor())
        self.m = torch.nn.AvgPool2d(2, 2)
        self.resize = Resize((input_size[0], input_size[1]))
        self.normalize = Normalize(mean=(0.781, 0.781, 0.781), std=(0.201, 0.201, 0.201))

        self.Sc: Tensor
        self.mu: Tensor

    def forward(self, input_tensor: Tensor) -> Tensor:
        input_tensor = self.normalize(input_tensor)
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        sc = self.Sc
        mu = torch.t(self.mu)

        embeddings = [self.m(feature) for feature in features]

        embedding_ = embedding_concat(embeddings[0], embeddings[1], self.s).to(self.Sc.device)
        total_embedding = embedding_.permute(0, 2, 3, 1).contiguous().to(self.Sc.device)

        lamda = self.lambda_value
        q_batch = total_embedding.view(total_embedding.shape[0], -1, embedding_.shape[1])

        temp = torch.mm(sc, torch.t(sc)) + lamda * torch.mm(sc, torch.t(sc))

        temp2_batch = torch.mm(mu, torch.t(sc)).unsqueeze(0).repeat(total_embedding.shape[0], 1, 1)

        w_batch = torch.matmul((torch.matmul(q_batch, torch.t(sc)) + lamda * temp2_batch), torch.linalg.pinv(temp))
        q_hat_batch = torch.matmul(w_batch, sc)

        score_patches_batch = torch.abs(q_batch - q_hat_batch)
        score_patches_temp_batch = torch.norm(score_patches_batch, dim=2)
        anomaly_map_batch = score_patches_temp_batch.reshape(total_embedding.shape[0], embedding_.shape[-1],
                                                             embedding_.shape[-1])
        anomaly_map_resized_batch = self.resize(anomaly_map_batch).unsqueeze(1)

        min_values, _ = anomaly_map_resized_batch.min(dim=2, keepdim=True)
        min_values, _ = min_values.min(dim=3, keepdim=True)

        max_values, _ = anomaly_map_resized_batch.max(dim=2, keepdim=True)
        max_values, _ = max_values.max(dim=3, keepdim=True)

        anomaly_map_resized_batch_normal = (anomaly_map_resized_batch - min_values) / (max_values - min_values)

        return anomaly_map_resized_batch_normal
