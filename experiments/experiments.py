from src.experiment_tracking.experiment_tracker import Expirement_Tracker
from src.models.catboost import CatBoost
from src.models.random_forest import RandomForest, RandomForestOptimized
from src.models.linear_regression import LinRegression
from src.models.svm import SVM_Model
from src.data_processing.simple_one_hot import process_data as process_data_one_hot
from src.data_processing.no_processing import process_data as process_data_no_processing
from src.data_processing.iqr_one_hot_processing import process_data as process_data_iqr_one_hot
from src.data_processing.feature_selection_processing import process_data as process_data_feature_selection
from src.data_processing.full_processing import process_data as process_data_full


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


def no_processing_experiment(path, wadb_token):
    X, y = process_data_no_processing(path)

    models = {
        "LinearRegression": LinRegression(),
        "CatBoost": CatBoost(),
        "OptimizedRandomForest": RandomForestOptimized(),
    }

    experiment_models = Expirement_Tracker(models, "ml_project_template_test", wadb_token)
    experiment_models.run_experiment(
        X,
        y,
        plot_name="Cross-validation scores, no processing",
        scorer="neg_root_mean_squared_error",
        cv=10,
    )


def iqr_one_hot_experiment(path, wadb_token):
    X, y = process_data_iqr_one_hot(path)

    models = {
        "LinearRegression": LinRegression(),
        "CatBoost": CatBoost(),
        "OptimizedRandomForest": RandomForestOptimized(),
    }

    experiment_models = Expirement_Tracker(models, "ml_project_template_test", wadb_token)
    experiment_models.run_experiment(
        X,
        y,
        plot_name="Cross-validation scores, iqr one hot",
        scorer="neg_root_mean_squared_error",
        cv=10,
    )


def feature_selection_experiment(path, wadb_token):
    X, y = process_data_feature_selection(path)

    models = {
        "LinearRegression": LinRegression(),
        "CatBoost": CatBoost(),
        "OptimizedRandomForest": RandomForestOptimized(),
    }

    experiment_models = Expirement_Tracker(models, "ml_project_template_test", wadb_token)
    experiment_models.run_experiment(
        X,
        y,
        plot_name="Cross-validation scores, feature selection",
        scorer="neg_root_mean_squared_error",
        cv=10,
    )


def full_processing_experiment(path, wadb_token):
    X, y = process_data_full(path)

    models = {
        "LinearRegression": LinRegression(),
        "CatBoost": CatBoost(),
        "OptimizedRandomForest": RandomForestOptimized(),
    }

    experiment_models = Expirement_Tracker(models, "ml_project_template_test", wadb_token)
    experiment_models.run_experiment(
        X,
        y,
        plot_name="Cross-validation scores, full processing",
        scorer="neg_root_mean_squared_error",
        cv=10,
    )
