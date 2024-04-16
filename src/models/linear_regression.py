import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from numpy.typing import ArrayLike
from src.models.model import ModelEnvelope


class LinRegression(ModelEnvelope):

    def __init__(self):
        super().__init__()
        self.__model = LinearRegression()

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cv: int = 10,
    ) -> ArrayLike:

        train_size, train_scores, test_scores = learning_curve(
            self.__model,
            X,
            y,
            train_sizes=np.linspace(1.0 / cv, 1.0, cv),
            cv=cv,
        )

        return list(test_scores[0])
