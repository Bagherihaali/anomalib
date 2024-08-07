import mlflow
from copy import deepcopy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from anomalib.models.draem import Draem


class MLFlowModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 dirpath: str,
                 requirements_path: str,
                 ):
        super().__init__()
        self.dirpath = dirpath
        self.requirements_path = requirements_path

        self.path_to_best_model = ''
        self.last_saved_model = ''
        self.best_score = None

    def on_train_epoch_end(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # there is two model checkpoint callbacks
                if callback.best_model_path:
                    self.path_to_best_model = callback.best_model_path
                    self.best_score = callback.best_model_score

        if self.path_to_best_model != '' and self.last_saved_model != self.path_to_best_model:
            model = pl_module.load_from_checkpoint(self.path_to_best_model)
            model_for_logging = deepcopy(model)

            if isinstance(model, Draem):
                for handle in model_for_logging.removable_handles.values():
                    handle.remove()
                model_for_logging.removable_handles = {}
                del model_for_logging.augmenter

            conda_env = self.requirements_path + r'\environment.yaml'
            mlflow.pytorch.log_model(model_for_logging, str(self.dirpath), conda_env=conda_env)
            pl_module.logger.log_metrics({'best_score': float(self.best_score)})
            self.last_saved_model = self.path_to_best_model
