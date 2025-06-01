import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


from sklearn.metrics import r2_score

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from src.components.tuner import Tuner
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'best_model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = [
                "Random Forest",
                "Decision Tree",
                "Gradient Boosting",
                "AdaBoost",
                "CatBoost",
                "XGBoost Regressor",
                "KNN Regresssor"
            ]

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

            
            logging.info("Initiate Hyperparameter Tuning of models using Optuna")
            
            tuner = Tuner(n_trials=50)

            tuner.tune_model(
                x_train=x_train,
                y_train=y_train,
                x_val=x_test,
                y_val=y_test,
                models=self.models
            )
            
            logging.info("Hyperparameter tuning completed. Refitting with best params.")
            best_model_name, best_model, _score = tuner.get_best_model(x_train,y_train)
            
            
            y_test_pred = best_model.predict(x_test)
            test_r2score = r2_score(y_test, y_test_pred)


            
            if test_r2score < 0.6:
                raise CustomException(
                    f"Best model test r2 score is {test_r2score}, which is less than 0.6. Model training failed."
                )
            
            logging.info(f"Best model from Optuna: {best_model_name} with test R2 score: {test_r2score}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
            return test_r2score

        except Exception as e:
            raise CustomException(e, sys)
