import os
import sys
import pickle
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.Airbnb.logger import logging
from src.Airbnb.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                param = params[model_name]

                if param:
                    logging.info(f"Tuning hyperparameters for {model_name}...")
                    rs = RandomizedSearchCV(
                        estimator=model, 
                        param_distributions=param, 
                        n_iter=5, 
                        cv=3, 
                        n_jobs=-1, 
                        verbose=1,
                        random_state=42
                    )
                    rs.fit(X_train, y_train)
                    model.set_params(**rs.best_params_)

                logging.info(f"Training final {model_name}...")
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score

            return report
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                "XGBoost Regressor": XGBRegressor(random_state=42)
            }

            params = {
                "Decision Tree Regressor": {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None]
                },
                "Gradient Boosting Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.8, 0.9, 1.0]
                },
                "Linear Regression": {},
                "Ridge Regression": {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                "XGBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }

            logging.info("Training and evaluating models...")
            model_report: dict = self.evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                models=models, params=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.warning("No best model found with an R-squared score above 0.60")

            logging.info(f"Best Model Found: {best_model_name} | R-squared Score: {best_model_score}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, "wb") as file_obj:
                pickle.dump(best_model, file_obj)

            # THE FIX: Now returning the model_report dictionary as well
            return best_model_score, best_model_name, model_report

        except Exception as e:
            raise CustomException(e, sys)
