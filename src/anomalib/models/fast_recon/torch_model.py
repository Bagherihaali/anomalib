import torch
import timm
from typing import Optional

import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights
from torchvision.transforms import Resize
from torch import Tensor, nn
from collections import OrderedDict

from anomalib.models.components import DynamicBufferModule


def embedding_concat(x: Tensor, y: Tensor, s: int):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    z = torch.cat((x, y.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)), dim=1)

    return z


def pool_embeddings(pool, features, map_ids):
    embeddings = [pool(feature) if i in map_ids else feature for i, feature in enumerate(features)]
    return embeddings


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

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

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


class FastReconModel(DynamicBufferModule, nn.Module):
    def __init__(
            self,
            input_size: tuple[int, int],
            layers: tuple[str, str],
            backbone: str = 'wide_resnet50',
            lambda_value: int = 2,
            m: Optional[torch.nn.AvgPool2d] = None,
            maps_to_pool=None,
            distance_metric: str = 'cosine_similarity',
            backbone_path: str = None
    ):
        super().__init__()
        self.input_size = input_size
        self.layers = layers
        self.backbone = backbone
        self.backbone_path = backbone_path
        self.lambda_value = lambda_value
        self.features = []
        self.feature_extractor = None
        self.maps_to_pool = maps_to_pool
        self.distance_metric = distance_metric
        self.m = m

        self.register_buffer("Sc", Tensor())
        self.register_buffer("mu", Tensor())

        self.Sc: Tensor
        self.mu: Tensor

    def init_feature_extractor(self):
        if self.backbone == 'wide_resnet50':

            out_indices = [int(item.split('layer')[1]) for item in self.layers]
            self.feature_extractor = timm.create_model(
                'wide_resnet50_2',
                checkpoint_path=self.backbone_path,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
                exportable=True,
            )
            self.feature_extractor.to(self.mu.device)
        elif self.backbone == 'unet':
            self.feature_extractor = UNet(in_channels=3, out_channels=1, init_features=32)
            weights = torch.load(self.backbone_path, map_location=self.mu.device)
            self.feature_extractor.load_state_dict(weights)

        def hook_t(module, input, output):
            self.features.append(output)

        if isinstance(self.feature_extractor, UNet):
            for layer_name in self.layers:
                layer = getattr(self.feature_extractor, layer_name, None)
                if layer is not None:
                    layer[-1].register_forward_hook(hook_t)

    def forward(self, input_tensor: Tensor) -> Tensor:
        self.features = []

        self.feature_extractor.eval()
        with torch.no_grad():
            if isinstance(self.feature_extractor, UNet):
                self.feature_extractor(input_tensor)
            else:
                self.features = self.feature_extractor(input_tensor)

        if self.training:
            output = self.features
        else:
            output = self.anomaly_map_generator(input_tensor, self.features)

        return output

    def anomaly_map_generator(self, input_tensor, features):

        sc = self.Sc
        mu = torch.t(self.mu)

        embeddings = pool_embeddings(self.m, self.features, self.maps_to_pool)

        s = int(embeddings[0].shape[2] / embeddings[1].shape[2])
        embedding_ = embedding_concat(embeddings[0], embeddings[1], s).to(self.Sc.device)
        total_embedding = embedding_.permute(0, 2, 3, 1).contiguous().to(self.Sc.device)

        lamda = self.lambda_value
        q_batch = total_embedding.view(total_embedding.shape[0], -1, embedding_.shape[1])

        temp = torch.mm(sc, torch.t(sc)) + lamda * torch.mm(sc, torch.t(sc))

        temp2_batch = torch.mm(mu, torch.t(sc)).unsqueeze(0).repeat(total_embedding.shape[0], 1, 1)

        w_batch = torch.matmul((torch.matmul(q_batch, torch.t(sc)) + lamda * temp2_batch), torch.pinverse(temp))
        q_hat_batch = torch.matmul(w_batch, sc)

        if self.distance_metric == 'cosine_similarity':
            cosine_sim = F.cosine_similarity(q_batch, q_hat_batch, dim=2)
            score_patches_temp_batch = 1 - torch.abs(cosine_sim)
        else:
            score_patches_batch = torch.abs(q_batch - q_hat_batch)
            score_patches_temp_batch = torch.norm(score_patches_batch, dim=2)

        anomaly_map_batch = score_patches_temp_batch.reshape(total_embedding.shape[0], embedding_.shape[-2],
                                                             embedding_.shape[-1])
        anomaly_map_resized_batch = Resize((self.input_size[0], self.input_size[1]))(
            anomaly_map_batch).unsqueeze(1)

        if self.distance_metric == 'cosine_similarity':
            anomaly_map_resized_batch_normal = anomaly_map_resized_batch

        else:
            min_values, _ = anomaly_map_resized_batch.min(dim=2, keepdim=True)
            min_values, _ = min_values.min(dim=3, keepdim=True)

            max_values, _ = anomaly_map_resized_batch.max(dim=2, keepdim=True)
            max_values, _ = max_values.max(dim=3, keepdim=True)

            max_values = torch.tensor([70] * anomaly_map_resized_batch.shape[0]).to(self.Sc.device)

            anomaly_map_resized_batch_normal = (anomaly_map_resized_batch - min_values) / (max_values - min_values)

        return anomaly_map_resized_batch_normal
