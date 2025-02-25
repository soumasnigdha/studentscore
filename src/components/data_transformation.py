import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    """
    DataTransformationConfig class is a dataclass that holds the configuration for the data transformation process.
    """
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    DataTransformation class is responsible for creating object of data transformation config.
    """
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        get_data_transformer method is responsible for data trasformation.
        """
        logging.info("Initiating data transformation process")
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),

                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        initiate_data_transformation method is responsible for initiating data transformation process
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read training and testing data")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            innput_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing on training dataframe and test dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(innput_features_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saved preprocessor object")

            save_obj(obj=preprocessing_obj, file_path=self.datatransformation_config.preprocessor_file_path)

            return (
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_file_path,
                )

        except Exception as e:
            raise CustomException(e, sys)