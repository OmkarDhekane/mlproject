import os
import sys

import pickle
import dill

from src.exception import CustomException

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train,y_train,x_test,y_test,models):

    try:
        model_report ={}

        for model_name, model in models.items():
            model.fit(x_train,y_train)
            #y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            #train_r2score = r2_score(y_train, y_train_pred)
            test_r2score = r2_score(y_test, y_test_pred)
            model_report[model_name] = test_r2score

        
        return model_report 

    except Exception as e:
        raise CustomException(e, sys)