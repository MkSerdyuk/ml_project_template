from tabnanny import verbose
from catboost import CatBoostRegressor
from numpy.typing import ArrayLike
from src.models.model import ModelEnvelope


class CatBoost(ModelEnvelope):

    def __init__(self):
        super().__init__()
        self.__model = CatBoostRegressor(
            random_seed=42,
            logging_level="Verbose",
            iterations=100,
        )

    def __fit_with_test(self, X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike):

        log = []

        def list_logger(text, log=log):
            text = text.split("\t")
            if len(text) < 3:
                return
            loss = float(text[2].split(" ")[1])

            log.append({"val_loss": loss})

        self.__model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            logging_level="Verbose",
            log_cout=list_logger,
        )

        return log

    def __fit_without_test(self, X_train: ArrayLike, y_train: ArrayLike):

        log = []

        def list_logger(text, log=log):
            text = text.split("\t")
            if len(text) < 2:
                return
            loss = float(text[1].split(" ")[1])
            log.append({"train_loss": loss})

        self.__model.fit(
            X_train,
            y_train,
            logging_level="Verbose",
            log_cout=list_logger,
        )

        return log

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
