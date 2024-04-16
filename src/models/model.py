from abc import ABC
from typing import List
from sklearn.model_selection import learning_curve
from numpy.typing import ArrayLike


class ModelEnvelope(ABC):

    def __init__(self):
        pass

    def fit(self, X: ArrayLike, y: ArrayLike, scorer: str | None = None, cv: int = 10) -> List[float]:
        return []

    def predict(self, X: ArrayLike) -> ArrayLike:
        return []

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        return 0.0

    def get_wandb_params(self) -> dict:
        return {}
