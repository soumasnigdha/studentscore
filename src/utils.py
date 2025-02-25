import os
import sys
import dill

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

def save_obj(obj, file_path):
    """
    save_preprocessor method is responsible for saving preprocessor object.
    """
    try:
        logging.info(f"Saving preprocessor object at {file_path}")
        dir_path = os.path.dirname(file_path)
        with open (file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
