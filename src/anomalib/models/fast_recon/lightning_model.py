import torch
import numpy as np
import cv2
import os

from torchvision.models import Wide_ResNet50_2_Weights
from torch.nn import functional as F
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter

from anomalib.models.fast_recon.kcenter_greedy import KCenterGreedy
from anomalib.models.components import AnomalyModule
from anomalib.models.fast_recon.torch_model import FastReconModel

__all__ = ["FastRecon"]


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


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def save_anomaly_map(anomaly_map, input_img, file_name):
    # print('start save anomly_map picture:{}'.format(file_name))
    if anomaly_map.shape[0] != input_img.shape[0]:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[1], input_img.shape[0]))
        # print('done')
    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_img)

    # save images
    cv2.imwrite(os.path.join(r'C:\Users\Mohammad\Desktop\temp-fast-recon', f'{file_name}.jpg'), input_img)
    cv2.imwrite(os.path.join(r'C:\Users\Mohammad\Desktop\temp-fast-recon', f'{file_name}_amap.jpg'),
                anomaly_map_norm_hm)
    cv2.imwrite(os.path.join(r'C:\Users\Mohammad\Desktop\temp-fast-recon', f'{file_name}_amap_on_img.jpg'), hm_on_img)


class FastRecon(AnomalyModule):
    def __init__(self,
                 layers: tuple[str, str] = ('layer1', 'layer2'),
                 input_size: tuple[int, int] = (256, 256),
                 coreset_sampling_ratio: int = 0.01,
                 lambda_value: int = 2,
                 backbone: str = 'wide_resnet50'
                 ):
        super(FastRecon, self).__init__()

        self.input_size = input_size
        self.coreset_sampling_ratio = coreset_sampling_ratio

        self.embedding_temp = []

        self.model = FastReconModel(self.input_size, layers, backbone, lambda_value)

    @staticmethod
    def configure_optimizers() -> None:
        return None

    def training_step(self, batch, batch_idx):
        self.model.feature_extractor.eval()
        features = self.model(batch["image"])

        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        self.embedding_temp.extend(embedding_concat(embeddings[0], embeddings[1]).cpu())

    def on_validation_start(self):
        embedding_temp = torch.stack(self.embedding_temp)
        total_embeddings = embedding_temp.permute(0, 2, 3, 1).contiguous()
        total_embeddings = total_embeddings.view(-1, embedding_temp.shape[1])

        embedding_mu = embedding_temp.view(embedding_temp.shape[0], embedding_temp.shape[1], -1)

        self.random_projector = SparseRandomProjection(n_components='auto', eps=0.9)
        selector = KCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(
            model=self.random_projector,
            already_selected=[],
            N=int(total_embeddings.shape[0] * self.coreset_sampling_ratio)
        )
        self.model.Sc = total_embeddings[selected_idx]
        self.model.mu = torch.mean(embedding_mu, 0)

    def validation_step(self, batch, *args, **kwargs):  # Nearest Neighbour Search
        anomaly_maps = self.model(batch['image'])
        batch["anomaly_maps"] = anomaly_maps
        batch['visualization'] = {
            'anomaly_maps': anomaly_maps
        }
        return batch
