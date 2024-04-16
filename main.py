import dotenv

from src.data_loading.data_loading import load_data
from experiments.experiments import (
    one_hot_experiment,
    no_processing_experiment,
    iqr_one_hot_experiment,
    feature_selection_experiment,
    full_processing_experiment,
)


def main():
    path = load_data()

    wadb_token = dotenv.get_key(".env", "WANDB_TOKEN")

    one_hot_experiment(path, wadb_token)
    no_processing_experiment(path, wadb_token)
    iqr_one_hot_experiment(path, wadb_token)
    feature_selection_experiment(path, wadb_token)
    full_processing_experiment(path, wadb_token)


if __name__ == "__main__":
    main()
