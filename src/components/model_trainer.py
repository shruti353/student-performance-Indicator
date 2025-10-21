import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_depth': [2, 4, 6, 8, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 6, 8, 10],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9]
                },
                "XGB Regressor": {
                    'learning_rate': [0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                },
                "CatBoost Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'iterations': [50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200]
                }
            }

            logging.info(" Starting model evaluation with GridSearchCV for all models...")

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f" Best Model: {best_model_name}")
            logging.info(f" Best R2 Score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                logging.warning(" No suitable model found with R2 > 0.6")
                raise CustomException("No best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f" Model saved successfully at {self.model_trainer_config.trained_model_file_path}")
            return best_model_score

        except Exception as e:
            logging.error("Error occurred during pipeline execution", exc_info=True)
            raise CustomException(e, sys)
