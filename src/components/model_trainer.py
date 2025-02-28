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
from src.utils import save_object, evaluate_model


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

      params = {
        "Linear Regression": {},
        "Decision Tree": {
          'criterion': ['squared_error', 'friedman_mse'],
          'splitter': ['best', 'random'],
          'max_depth': [2, 4, 6, 8, 10, None],
          'max_features': ['sqrt', 'log2'],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
        },
        "Random Forest": {
          'n_estimators': [100, 200, 300, 400, 500],
          'criterion': ['squared_error', 'friedman_mse'],
          'max_depth': [2, 4, 6, 8, 10, None],
          'max_features': ['sqrt', 'log2'],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
        },
        "Gradient Boosting": {
          'n_estimators': [100, 200, 300, 400, 500],
          'learning_rate': [0.001, 0.01, 0.1],
          'subsample': [0.5, 0.7, 1.0],
          'max_depth': [2, 4, 6, 8, 10, None],
          'max_features': ['sqrt', 'log2'],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
        },
        "AdaBoost": {
          'n_estimators': [50, 100, 200, 400, 800],
          'learning_rate': [0.001, 0.01, 0.1, 1],
          'loss': ['linear', 'exponential'],
        },
        "XGBoost": {
          'n_estimators': [100, 200, 300, 400, 500],
          'learning_rate': [0.001, 0.01, 0.1, 1],
          'subsample': [0.5, 0.7, 1.0],
          'max_depth': [2, 4, 6, 8, 10, None],
          'colsample_bytree': [0.5, 0.7, 1.0],
          'colsample_bylevel': [0.5, 0.7, 1.0],
          'reg_alpha': [0, 0.5, 1],
        },
        "KNN": {
          'n_neighbors': [3, 5, 7, 9],
          'weights': ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
          'leaf_size': [10, 20, 30, 40, 50],
          'metric': ['minkowski', 'euclidean', 'manhattan'],
        },
        "CatBoost": {
          'iterations': [100, 200, 300, 400, 500],
          'depth': [2, 4, 6, 8, 10],
          'learning_rate': [0.001, 0.01, 0.1, 1],
          'l2_leaf_reg': [1, 3, 5, 7, 9],
          'border_count': [32, 64, 128],
        },
      }


      
      model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)

      # to get the best model score from the model_report:dict
      best_model_score = max(sorted(model_report.values()))

      # to get the best model name from the model_report:dict
      best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]

      best_model = models[best_model_name]

      if best_model_score<0.6:
        raise CustomException("No best model found", sys)
      logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
        )
      
      predicted = best_model.predict(X_test)
      r2 = r2_score(y_test, predicted)

      return r2

    except Exception as e:
      raise CustomException(e, sys)