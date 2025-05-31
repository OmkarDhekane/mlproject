import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'best_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """ This function is responsible for training model"""

        try:
            logging.info("Splitting training and testing input data into features and target")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost Regressor": XGBRegressor(eval_metric='logloss'),
                "KNN Regresssor": KNeighborsRegressor(),
                
            }

            model_report: dict = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models
            )

            # get the best model score from the model report
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    f"Best model score is {best_model_score}, which is less than 0.6. Model training failed."
                )
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            test_preds = best_model.predict(x_test)
            r2_test_score = r2_score(y_test, test_preds)
            return r2_test_score

        except Exception as e:
            raise CustomException(e, sys)
