import pandas as pd
from sklearn.model_selection import train_test_split


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

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
