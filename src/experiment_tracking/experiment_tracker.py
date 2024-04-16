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
        cv: int | None = 10,
    ):
        """
        Runs an experiment and logs it to wandb using the given input features, target values, and optional parameters.

        Parameters:
            X (ArrayLike): The input features of the training data.
            y (ArrayLike): The target values of the training data.
            scorer (str | None, optional): The scoring metric to use for evaluation. Defaults to None.
            plot_name (str, optional): The name of the plot to be generated. Defaults to "Cross-validation scores".
            cv (int | None, optional): The number of cross-validation folds to perform. Defaults to 10.

        """

        log = {}

        for model_name in self.__models:
            model = self.__models[model_name]
            log[model_name] = model.fit(X, y, scorer=scorer, cv=cv)

            print(f"{model_name} is trained")

        self.__wandb_run.log(
            {
                plot_name: wandb.plot.line_series(
                    xs=[i + 1 for i in range(cv)],
                    ys=list(log.values()),
                    keys=list(log.keys()),
                    title=plot_name,
                    xname="Iteration",
                )
            }
        )

        self.__wandb_run.finish()
