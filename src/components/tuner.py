
import sys

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from sklearn.metrics import mean_squared_error


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)



class Tuner:
    def __init__(self, n_trials=50):
        self.study  = optuna.create_study(direction='minimize')
        self.n_trials = n_trials

    def get_model(self,trial, model_name):
        try:
            if model_name == "Random Forest":
                return RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    max_depth=trial.suggest_int("max_depth", 4, 20)
                )
            elif model_name == "Decision Tree":
                return DecisionTreeRegressor(
                    max_depth=trial.suggest_int("max_depth", 4, 20)
                )
            elif model_name == "Gradient Boosting":
                return GradientBoostingRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    max_depth=trial.suggest_int("max_depth", 3, 10)
                )
            elif model_name == "AdaBoost":
                return AdaBoostRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0)
                )
            elif model_name == "CatBoost":
                return CatBoostRegressor(
                    iterations=trial.suggest_int("iterations", 100, 300),
                    depth=trial.suggest_int("depth", 4, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    verbose=False
                )
            elif model_name == "XGBoost Regressor":
                return XGBRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    verbosity=0
                )
            elif model_name == "KNN Regresssor":
                return KNeighborsRegressor(
                    n_neighbors=trial.suggest_int("n_neighbors", 3, 15),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]))
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            raise CustomException(e,sys)        


    def objective(self, trial, x_train, y_train, x_val, y_val, models):
        model_name = trial.suggest_categorical("model", models)
        model = self.get_model(trial, model_name)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        return mean_squared_error(y_val, preds)
    
    def tune_model(self, x_train, y_train, x_val, y_val, models):
        self.study.optimize(lambda trial: self.objective(trial, x_train, y_train, x_val, y_val, models), n_trials=self.n_trials)

    def get_best_model(self, x_train, y_train):
        if self.study.best_trial is None:
            raise ValueError("No trials have been completed yet.")
        best_trial = self.study.best_trial
        best_model_name = best_trial.params['model']
        best_model = self.get_model(best_trial, best_model_name)
        best_model.fit(x_train, y_train)
        
        return best_model_name, best_model, self.study.best_value

    
