import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


class MLFlowModelCheckpoint(Callback):
    def __init__(self, dirpath, requirements_path):
        super().__init__()
        self.dirpath = dirpath
        self.best_model_path = None
        self.last_saved_model = None
        self.requirements_path = requirements_path

    def on_validation_end(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # there is two model checkpoint callbacks
                if callback.best_model_path:
                    self.best_model_path = callback.best_model_path

        if self.best_model_path and self.last_saved_model != self.best_model_path:
            model = pl_module.load_from_checkpoint(self.best_model_path)
            conda_env = self.requirements_path + r'\environment.yaml'
            # pip_requirements = self.requirements_path + r'\requirements.txt'
            mlflow.pytorch.log_model(model, str(self.dirpath), conda_env=conda_env)
            self.last_saved_model = self.best_model_path
