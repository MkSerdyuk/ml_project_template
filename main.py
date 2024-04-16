import dotenv

from src.data_loading.data_loading import load_data
from experiments.experiments import one_hot_experiment


def main():
    path = load_data()

    wadb_token = dotenv.get_key(".env", "WANDB_TOKEN")

    one_hot_experiment(path, wadb_token)


if __name__ == "__main__":
    main()
