import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(path):
    df = pd.read_csv(path)

    df.drop("Id", axis=1, inplace=True)

    object_columns = df.select_dtypes(include="object").columns
    df = df.drop(object_columns, axis=1)

    df.fillna(df.mean(), inplace=True)

    df = df.drop(index=df[df["SalePrice"] <= 0].index, axis=0)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    return X, y
