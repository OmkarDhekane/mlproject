# Helps to transform data into a suitable format for analysis or modeling.

#aim: feature engineering, cleaning, one hotting


import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        It creates a preprocessing pipeline that includes:
        - Imputation for missing values
        - One-hot encoding for categorical features
        - Standard scaling for numerical features
        """
        try:
            logging.info("Data Transformation initiated")
            

            # Define numerical and categorical features
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                "gender",
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]


            numerical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ] )

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
            ])

            logging.info("Numerical and categorical pipelines created")
            
            preprocesor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocesor 
        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")
            logging.info("Getting preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target_column = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            logging.info("splitting XY => X,y for Train+Test")
            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor.transform(input_features_test_df)
            logging.info("Preprocessing completed")

            # 
            transformed_train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            transformed_test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            return (
                transformed_train_arr,
                transformed_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        



