import numpy as np

from typing import List
from sklearn.model_selection import learning_curve

from catboost import CatBoostRegressor
from numpy.typing import ArrayLike
from src.models.model import ModelEnvelope


class CatBoost(ModelEnvelope):

    def __init__(self):
        self.__model = CatBoostRegressor(
            random_seed=42,
            logging_level="Silent",
            iterations=50,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        scorer: str | None = None,
        cv: int = 10,
    ) -> List[float]:

        train_size, train_scores, test_scores = learning_curve(
            self.__model,
            X,
            y,
            train_sizes=np.linspace(1.0 / cv, 1.0, cv),
            cv=cv,
            scoring=scorer,
        )

        return list(test_scores[0])
