import sys  # Importing the sys module for system-specific parameters and functions
from dataclasses import dataclass  # Importing the dataclass decorator for creating data classes

import numpy as np  # Importing the numpy library for numerical operations
import pandas as pd  # Importing the pandas library for data manipulation
from sklearn.compose import ColumnTransformer  # Importing the ColumnTransformer class from scikit-learn for column-wise transformations
from sklearn.impute import SimpleImputer  # Importing the SimpleImputer class from scikit-learn for data imputation
from sklearn.pipeline import Pipeline  # Importing the Pipeline class from scikit-learn for creating a pipeline of transformations
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Importing OneHotEncoder and StandardScaler from scikit-learn for categorical and numerical data preprocessing

from src.exception import CustomException  # Importing a custom exception class
from src.logger import logging  # Importing the logging module for logging purposes
import os  # Importing the os module for file and directory operations

from src.utils import save_object  # Importing a custom utility function for saving objects

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Setting the default preprocessor object file path

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Creating an instance of DataTransformationConfig

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]  # List of numerical columns
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]  # List of categorical columns

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Imputing missing values using the median strategy
                ("scaler", StandardScaler())  # Scaling numerical features
                ]
            )  # Creating a pipeline for numerical features

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Imputing missing values using the most frequent strategy
                ("one_hot_encoder", OneHotEncoder()),  # Performing one-hot encoding for categorical features
                ("scaler", StandardScaler(with_mean=False))  # Scaling categorical features
                ]
            )  # Creating a pipeline for categorical features

            logging.info(f"Categorical columns: {categorical_columns}")  # Logging categorical columns
            logging.info(f"Numerical columns: {numerical_columns}")  # Logging numerical columns

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),  # Applying num_pipeline to numerical columns
                ("cat_pipelines", cat_pipeline, categorical_columns)  # Applying cat_pipeline to categorical columns
                ]
            )  # Creating a ColumnTransformer object to apply different transformations to different columns

            return preprocessor  # Returning the preprocessor object
        
        except Exception as e:
            raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)  # Reading the train data into a DataFrame
            test_df = pd.read_csv(test_path)  # Reading the test data into a DataFrame

            logging.info("Read train and test data completed")  # Logging an info message

            logging.info("Obtaining preprocessing object")  # Logging an info message

            preprocessing_obj = self.get_data_transformer_object()  # Getting the preprocessing object

            target_column_name = "math_score"  # Name of the target column
            numerical_columns = ["writing_score", "reading_score"]  # List of numerical columns

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Separating input features from the train DataFrame
            target_feature_train_df = train_df[target_column_name]  # Extracting the target feature from the train DataFrame

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Separating input features from the test DataFrame
            target_feature_test_df = test_df[target_column_name]  # Extracting the target feature from the test DataFrame

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )  # Logging an info message

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)  # Applying preprocessing to the training input features
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Applying preprocessing to the testing input features

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]  # Combining preprocessed training input features with the target feature
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # Combining preprocessed testing input features with the target feature

            logging.info(f"Saved preprocessing object.")  # Logging an info message

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )  # Saving the preprocessing object

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )  # Returning the train and test arrays along with the path of the saved preprocessing object

        except Exception as e:
            raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module
