import wandb
from typing import Dict
from numpy.typing import ArrayLike

from src.models.model import ModelEnvelope


class Expirement_Tracker:

    def __init__(self, models: Dict[str, ModelEnvelope], project_name: str, wadb_token: str):

        self.project_name = project_name
        self.__models = models

        wandb.login(key=wadb_token)
        self.__wandb_run = wandb.init(project=self.project_name)

    def run_experiment(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_test: ArrayLike | None = None,
        y_test: ArrayLike | None = None,
    ):

        log = {}

        for model_name in self.__models:
            model = self.__models[model_name]

            if X_test is not None and y_test is not None:
                log[model_name] = model.fit(X_train, y_train, X_test, y_test)
            else:
                log[model_name] = model.fit(X_train, y_train)

            print(f"{model_name} is trained")

        for i in range(len(log[model_name])):

            log_item = {}
            for model_name in log:
                log_item[model_name] = log[model_name][i]

            self.__wandb_run.log(log_item)
