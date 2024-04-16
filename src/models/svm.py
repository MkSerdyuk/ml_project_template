import numpy as np

from typing import List
from sklearn.svm import LinearSVR
from sklearn.model_selection import learning_curve

from numpy.typing import ArrayLike
from src.models.model import ModelEnvelope


class SVM_Model(ModelEnvelope):
    def __init__(self):
        super().__init__()
        self.__model = LinearSVR()

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
