import dotenv

from src.data_loading.data_loading import load_data
from src.data_processing.simple_one_hot import process_data
from src.experiment_tracking.experiment_tracker import Expirement_Tracker
from src.models.catboost import CatBoost
from src.models.random_forest import RandomForest, RandomForestOptimized
from src.models.linear_regression import LinRegression


def main():
    path = load_data()
    X, y = process_data(path)

    wadb_token = dotenv.get_key(".env", "WANDB_TOKEN")

    models = {
        "CatBoost": CatBoost(),
        "RandomForest": RandomForest(),
        "OptimizedRandomForest": RandomForestOptimized(),
        "LinearRegression": LinRegression(),
    }

    experiment_models = Expirement_Tracker(models, "ml_project_template", wadb_token)
    experiment_models.run_experiment(X, y, 10)


if __name__ == "__main__":
    main()
