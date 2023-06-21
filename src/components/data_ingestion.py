import os  # Importing the os module for file and directory operations
import sys  # Importing the sys module for system-specific parameters and functions
from src.exception import CustomException  # Importing a custom exception class
from src.logger import logging  # Importing the logging module for logging purposes
import pandas as pd  # Importing the pandas library for data manipulation

from sklearn.model_selection import train_test_split  # Importing train_test_split from scikit-learn for splitting data
from dataclasses import dataclass  # Importing the dataclass decorator for creating data classes

from src.components.data_transformation import DataTransformation  # Importing a custom data transformation class
from src.components.data_transformation import DataTransformationConfig  # Importing a custom data transformation config class

from src.components.model_trainer import ModelTrainerConfig  # Importing a custom model trainer config class
from src.components.model_trainer import ModelTrainer  # Importing a custom model trainer class

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Setting the default train data path
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Setting the default test data path
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Setting the default raw data path

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Creating an instance of DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Logging an info message

        try:
            df = pd.read_csv('notebook\data\stud.csv')  # Reading a CSV file into a DataFrame
            logging.info('Read the dataset as dataframe')  # Logging an info message

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Creating directories

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Saving the raw data to a CSV file

            logging.info("Train test split initiated")  # Logging an info message
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Performing train-test split

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Saving train data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Saving test data

            logging.info("Ingestion of the data is completed")  # Logging an info message

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )  # Returning train and test data paths

        except Exception as e:
            raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module

if __name__ == "__main__":
    obj = DataIngestion()  # Creating an instance of DataIngestion
    train_data, test_data = obj.initiate_data_ingestion()  # Initiating data ingestion

    data_transformation = DataTransformation()  # Creating an instance of DataTransformation
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)  # Initiating data transformation

    model_trainer = ModelTrainer()  # Creating an instance of ModelTrainer
