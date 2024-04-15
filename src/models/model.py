from abc import ABC
from numpy.typing import ArrayLike


class ModelEnvelope(ABC):
    def __init__(self):
        pass

    def __fit_with_test(self, X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike):
        return []

    def __fit_without_test(self, X_train: ArrayLike, y_train: ArrayLike):
        return []

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_test: ArrayLike | None = None,
        y_test: ArrayLike | None = None,
    ) -> ArrayLike:
        if X_test is not None and y_test is not None:
            return self.__fit_with_test(X_train, y_train, X_test, y_test)
        else:
            return self.__fit_without_test(X_train, y_train)

    def predict(self, X: ArrayLike) -> ArrayLike:
        return []

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        return 0.0

    def get_wandb_params(self) -> dict:
        return {}
