import dotenv

from src.data_loading.data_loading import load_data
from src.data_processing.simple_one_hot import process_data
from src.experiment_tracking.experiment_tracker import Expirement_Tracker
from src.models.catboost import CatBoost


def main():
    path = load_data()
    X_train, X_test, y_train, y_test = process_data(path)

    wadb_token = dotenv.get_key(".env", "WANDB_TOKEN")

    models = {"CatBoost": CatBoost()}

    experiment_models = Expirement_Tracker(models, "ml_project_template", wadb_token)
    experiment_models.run_experiment(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
