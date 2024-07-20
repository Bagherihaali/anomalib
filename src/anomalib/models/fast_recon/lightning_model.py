import torch

from anomalib.models.components import KCenterGreedy
from anomalib.models.components import AnomalyModule
from anomalib.models.fast_recon.torch_model import FastReconModel, embedding_concat, pool_embeddings

__all__ = ["FastRecon"]


class FastRecon(AnomalyModule):
    def __init__(self,
                 layers: tuple[str, str] = ('layer1', 'layer2'),
                 input_size: tuple[int, int] = (256, 256),
                 coreset_sampling_ratio: float = 0.01,
                 lambda_value: int = 2,
                 backbone: str = 'wide_resnet50',
                 pool_args: tuple = (1, 1, 0),
                 maps_to_pool: tuple = (0, 1),
                 distance_metric: str = 'cosine_similarity',
                 backbone_path: str = None,
                 script_mode: bool = False
                 ):
        super(FastRecon, self).__init__()

        self.layers = layers
        self.backbone = backbone
        self.backbone_path = backbone_path
        self.lambda_value = lambda_value
        self.pool_args = pool_args
        self.maps_to_pool = maps_to_pool
        self.m = torch.nn.AvgPool2d(pool_args[0], pool_args[1], pool_args[2])
        self.script_mode = script_mode
        self.distance_metric = distance_metric

        self.input_size = input_size
        self.coreset_sampling_ratio = coreset_sampling_ratio

        self.embedding_temp = []

        self.model = FastReconModel(
            input_size=self.input_size,
            layers=self.layers,
            backbone=self.backbone,
            lambda_value=self.lambda_value,
            m=self.m,
            maps_to_pool=self.maps_to_pool,
            backbone_path=self.backbone_path,
            distance_metric=self.distance_metric,
        )

        self.model.init_feature_extractor()
        self.model.feature_extractor.to(self.device)

    @staticmethod
    def configure_optimizers() -> None:
        return None

    def training_step(self, batch, batch_idx):
        features = self.model(batch["image"])

        embeddings = pool_embeddings(self.m, features, self.maps_to_pool)
        s = int(embeddings[0].shape[2] / embeddings[1].shape[2])
        self.embedding_temp.extend(embedding_concat(embeddings[0], embeddings[1], s))

        if self.script_mode:
            progress = ((batch_idx + 1) / len(self.train_dataloader())) * 100
            if int(progress) == 100:
                print(f'\rPhase one: {progress:.1f}%')
            print(f'\rPhase one: {progress:.1f}%', end='', flush=True)

    def on_validation_start(self):
        del self.model.feature_extractor
        torch.cuda.empty_cache()
        self.model.feature_extractor = None

        embedding_temp = torch.stack(self.embedding_temp)
        total_embeddings = embedding_temp.permute(0, 2, 3, 1).reshape(-1, embedding_temp.shape[1]).contiguous()

        embedding_mu = embedding_temp.view(embedding_temp.shape[0], embedding_temp.shape[1], -1)

        sampler = KCenterGreedy(embedding=total_embeddings.to(self.device), sampling_ratio=self.coreset_sampling_ratio)
        coreset = sampler.sample_coreset()

        self.model.Sc = coreset
        self.model.mu = torch.mean(embedding_mu, 0)

        self.model.init_feature_extractor()
        self.model.feature_extractor.to(self.device)

    def validation_step(self, batch, *args, **kwargs):  # Nearest Neighbour Search

        anomaly_maps = self.model(batch['image'])
        batch["anomaly_maps"] = anomaly_maps
        batch['visualization'] = {
            'anomaly_maps': anomaly_maps
        }
        return batch
