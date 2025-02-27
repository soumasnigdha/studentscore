import os
import sys
import dill

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score

def save_obj(obj, file_path):
    """
    save_obj method is responsible for saving preprocessor object.
    """
    try:
        logging.info(f"Saving preprocessor object at {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open (file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    evaluate_models method is responsible for evaluating models.
    """
    model_report = {}
    try:
        logging.info("Evaluating models")
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
    

