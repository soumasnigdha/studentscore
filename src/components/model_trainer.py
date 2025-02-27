import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model


@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    """
    initiate_model_trainer method is responsible for initiating model training.
    """
    logging.info("Initiating model training")
    try:
      logging.info("Splitting input data into training and testing data")
      X_train, y_train,X_test, y_test = (
        train_array[:, :-1],
        train_array[:, -1],
        test_array[:, :-1],
        test_array[:, -1]
      )
      models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "XGBoost": XGBRegressor(),
        "KNN": KNeighborsRegressor(),
        "CatBoost": CatBoostRegressor(verbose=False),
      } 
      
      model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

      # to get the best model score from the model_report:dict
      best_model_score = max(sorted(model_report.values()))

      # to get the best model name from the model_report:dict
      best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]

      best_model = models[best_model_name]

      if best_model_score<0.6:
        raise CustomException("No best model found", sys)
      logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

      save_obj(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
        )
      
      predicted = best_model.predict(X_test)
      r2 = r2_score(y_test, predicted)

      return r2

    except Exception as e:
      raise CustomException(e, sys)