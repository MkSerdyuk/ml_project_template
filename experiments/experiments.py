from src.data_processing.simple_one_hot import process_data as process_data_one_hot
from src.experiment_tracking.experiment_tracker import Expirement_Tracker
from src.models.catboost import CatBoost
from src.models.random_forest import RandomForest, RandomForestOptimized
from src.models.linear_regression import LinRegression
from src.models.svm import SVM_Model


def one_hot_experiment(path, wadb_token):
    X, y = process_data_one_hot(path)

    models = {
        "LinearRegression": LinRegression(),
        "CatBoost": CatBoost(),
        "RandomForest": RandomForest(),
        "OptimizedRandomForest": RandomForestOptimized(),
        "SVM": SVM_Model(),
    }

    experiment_models = Expirement_Tracker(models, "ml_project_template_test", wadb_token)
    experiment_models.run_experiment(
        X,
        y,
        plot_name="Cross-validation scores, simple one hot",
        scorer="neg_root_mean_squared_error",
        cv=10,
    )
