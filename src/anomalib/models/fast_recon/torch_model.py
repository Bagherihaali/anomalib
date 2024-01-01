import torch

from torchvision.models import Wide_ResNet50_2_Weights
from torchvision.transforms import Resize
from torch import Tensor, nn
from torch.nn import functional as F
from collections import OrderedDict

from anomalib.models.components import DynamicBufferModule


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


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
            lambda_value: int = 2
    ):
        super().__init__()
        self.input_size = input_size
        self.layers = layers
        self.backbone = backbone
        self.lambda_value = lambda_value

        if self.backbone == 'wide_resnet50':
            self.feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2',
                                                    weights=Wide_ResNet50_2_Weights.DEFAULT)
        elif self.backbone == 'unet':
            self.feature_extractor = UNet(in_channels=3, out_channels=1, init_features=32)
            weights = torch.load( r'C:\Users\Mohammad\.cache\torch\hub\mateuszbuda_brain-segmentation-pytorch_master\weights\unet.pt')
            self.feature_extractor.load_state_dict(weights)
            # self.feature_extractor = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            #                                         in_channels=3, out_channels=1, init_features=32, pretrained=True)

        self.features = []

        def hook_t(module, input, output):
            self.features.append(output)

        for layer_name in self.layers:
            layer = getattr(self.feature_extractor, layer_name, None)
            if layer is not None:
                layer[-1].register_forward_hook(hook_t)

        self.register_buffer("Sc", Tensor())
        self.register_buffer("mu", Tensor())

        self.Sc: Tensor
        self.mu: Tensor

    def forward(self, input_tensor: Tensor) -> Tensor:
        self.features = []

        with torch.no_grad():
            self.feature_extractor(input_tensor)
            features = self.features

        if self.training:
            output = features
        else:
            output = self.anomaly_map_generator(input_tensor, features)

        return output

    def anomaly_map_generator(self, input_tensor, features):

        sc = self.Sc
        mu = torch.t(self.mu)

        embeddings = []
        m = torch.nn.AvgPool2d(3, 1, 1)
        for feature in features:
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1]).to(self.Sc.device)
        embedding = embedding_.permute(0, 2, 3, 1).contiguous().to(self.Sc.device)
        q = embedding.view(-1, embedding_.shape[1])

        lamda = self.lambda_value

        temp = (torch.mm(sc, torch.t(sc)) + lamda * torch.mm(sc, torch.t(sc)))
        temp2 = torch.mm(mu, torch.t(sc))
        w = torch.mm((torch.mm(q, torch.t(sc)) + lamda * temp2), torch.linalg.inv(temp))
        q_hat = torch.mm(w, sc)

        # original
        score_patches = torch.abs(q - q_hat)
        score_patches_temp = torch.norm(score_patches, dim=1)

        # form heatmap
        score_patches = torch.t(score_patches_temp)
        # anomaly_map = score_patches.reshape((int(self.input_size[0] / 8), int(self.input_size[1] / 8)))
        anomaly_map = score_patches.reshape((embedding_.shape[-1], embedding_.shape[-1]))

        # max value
        anomaly_map_resized = Resize((self.input_size[0], self.input_size[1]))(anomaly_map.unsqueeze(0).unsqueeze(0))

        anomaly_map_resized = (anomaly_map_resized - anomaly_map_resized.min()) / (
                anomaly_map_resized.max() - anomaly_map_resized.min())

        final_anomaly_map = anomaly_map_resized.squeeze(0).to(self.Sc.device)

        return final_anomaly_map
