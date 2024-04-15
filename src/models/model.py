from abc import ABC
from numpy.typing import ArrayLike


class ModelEnvelope(ABC):
    def __init__(self):
        pass

    def fit(self, X: ArrayLike, y: ArrayLike) -> dict:
        return {}

    def predict(self, X: ArrayLike) -> ArrayLike:
        return []

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        return 0.0

    def get_wandb_params(self) -> dict:
        return {}
