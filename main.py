from src.data_loading.data_loading import load_data
from src.data_processing.data_processing import process_data


def main():
    path = load_data()
    X_train, X_test, y_train, y_test = process_data(path)


if __name__ == "__main__":
    main()
