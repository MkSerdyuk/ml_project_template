import wandb
from numpy.typing import ArrayLike

from src.models.model import ModelEnvelope


class Expirement_Tracker:

    def __init__(self, model: ModelEnvelope, project_name: str, wadb_token: str):

        self.project_name = project_name
        self.__model = model

        wandb.login(key=wadb_token)
        self.__wandb_run = wandb.init(
            project=self.project_name,
            config=self.__model.get_wandb_params(),
        )

    def run_experiment(self, X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike):
        log = self.__model.fit(X_train, y_train)
        for key, value in log.items():
            self.__wandb_run.log({key: value})
