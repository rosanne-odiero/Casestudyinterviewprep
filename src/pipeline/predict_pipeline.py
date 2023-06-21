import os
import sys  # Importing the sys module for system-specific parameters and functions
import pandas as pd  # Importing the pandas library for data manipulation
from src.exception import CustomException  # Importing a custom exception class
from src.utils import load_object  # Importing a custom utility function for loading objects

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")  # Setting the path of the trained model
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')  # Setting the path of the preprocessor object
            print("Before Loading")  # Printing a message before loading the objects
            model = load_object(file_path=model_path)  # Loading the trained model object
            preprocessor = load_object(file_path=preprocessor_path)  # Loading the preprocessor object
            print("After Loading")  # Printing a message after loading the objects
            data_scaled = preprocessor.transform(features)  # Applying the preprocessor to the input features
            preds = model.predict(data_scaled)  # Making predictions using the trained model
            return preds  # Returning the predictions

        except Exception as e:
            raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }  # Creating a dictionary of input data

            return pd.DataFrame(custom_data_input_dict)  # Creating a DataFrame from the input data dictionary

        except Exception as e:
            raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module
