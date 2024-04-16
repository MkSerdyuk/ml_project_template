import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression


def one_hot_encode(df, columns, inplace=False):
    if not inplace:
        df = df.copy()
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


def process_data(path):
    df = pd.read_csv(path)

    df.drop("Id", axis=1, inplace=True)

    object_columns = df.select_dtypes(include="object").columns
    df = one_hot_encode(df, object_columns)

    df.fillna(df.mean(), inplace=True)

    df = df.drop(index=df[df["SalePrice"] <= 0].index, axis=0)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X = SelectKBest(f_regression, k=100).fit_transform(X, y)

    return X, y
