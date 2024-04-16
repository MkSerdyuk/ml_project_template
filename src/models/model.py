from abc import ABC
from typing import List
from sklearn.model_selection import learning_curve
from numpy.typing import ArrayLike


class ModelEnvelope(ABC):

    def __init__(self):
        pass

    def fit(
        self, X: ArrayLike, y: ArrayLike, scorer: str | None = None, cv: int | None = None
    ) -> List[float]:
        """
        Fits the model to the training data.

        Args:
            X (ArrayLike): The input features of the training data.
            y (ArrayLike): The target values of the training data.
            scorer (str | None, optional): The scoring metric to use for evaluation. Defaults to None.
            cv (int | None, optional): The number of cross-validation folds to perform. Defaults to None.

        Returns:
            List[float]: A list of test scores obtained from validation.
        """
        return []

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predicts the target values for the given input features.

        Args:
            X (ArrayLike): The input features for prediction.

        Returns:
            ArrayLike: The predicted target values.
        """
        return []

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Calculate the score of the model based on the given input features and target values.

        Parameters:
            X (ArrayLike): The input features.
            y (ArrayLike): The target values.

        Returns:
            float: The score of the model.
        """
        return 0.0

    def get_wandb_params(self) -> dict:
        """
        Get the wandb parameters.

        :return: A dictionary containing the wandb parameters.
        :rtype: dict
        """
        return {}
