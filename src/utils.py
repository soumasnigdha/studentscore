import os
import sys
import dill

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(obj, file_path):
    """
    save_object method is responsible for saving object.
    """
    try:
        logging.info(f"Saving object at {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open (file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    evaluate_models method is responsible for evaluating models.
    """
    model_report = {}
    try:
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            param = params[list(models.keys())[i]]
            model = list(models.values())[i]

            logging.info(f"Evaluating model {model_name}")
            grid = GridSearchCV(model, param, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2
            logging.info(f"Model {model_name} evaluated with R2 score {r2}")
        
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    


