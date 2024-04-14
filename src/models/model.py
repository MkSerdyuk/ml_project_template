class Model:
    def __init__(self, model, criterion):
        self.__model = model
        self.__criterion = criterion

    def fit(self, X, y):
        self.__model.fit(X, y)

    def predict(self, X):
        return self.__model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.__criterion(y, y_pred)
