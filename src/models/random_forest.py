import numpy as np

from typing import List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve

from numpy.typing import ArrayLike

from src.models.model import ModelEnvelope


class RandomForest(ModelEnvelope):

    def __init__(self):
        super().__init__()
        self.__model = RandomForestRegressor(random_state=42, n_estimators=50)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        scorer: str = "neg_root_mean_squared_error",
        cv: int = 10,
    ) -> ArrayLike:

        train_size, train_scores, test_scores = learning_curve(
            self.__model,
            X,
            y,
            train_sizes=np.linspace(1.0 / cv, 1.0, cv),
            cv=cv,
            scoring=scorer,
        )

        return list(test_scores[0])


class RandomForestOptimized(ModelEnvelope):

    def __init__(self):
        super().__init__()
        self.__model = RandomForestRegressor(
            random_state=42,
            min_samples_leaf=2,
            max_depth=15,
            min_samples_split=10,
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
            scoring=scorer,
            cv=cv,
        )

        return list(test_scores[0])
