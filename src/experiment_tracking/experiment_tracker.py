import wandb

from socket import timeout
from matplotlib.pyplot import ylabel
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
        X: ArrayLike,
        y: ArrayLike,
        scorer: str | None = None,
        plot_name="Cross-validation scores",
        cv: int = 10,
    ):

        log = {}

        for model_name in self.__models:
            model = self.__models[model_name]
            log[model_name] = model.fit(X, y, scorer=scorer, cv=cv)

            print(f"{model_name} is trained")

        self.__wandb_run.log(
            {
                "cv_scores": wandb.plot.line_series(
                    xs=[i for i in range(cv)],
                    ys=list(log.values()),
                    keys=list(log.keys()),
                    title=plot_name,
                    xname="Iteration",
                )
            }
        )

        self.__wandb_run.finish()
