import os  # Importing the os module for file and directory operations
import sys  # Importing the sys module for system-specific parameters and functions

import numpy as np  # Importing the numpy library for numerical operations
import pandas as pd  # Importing the pandas library for data manipulation
import dill  # Importing the dill library for object serialization
import pickle  # Importing the pickle module for object serialization
from sklearn.metrics import r2_score  # Importing r2_score from scikit-learn for model evaluation
from sklearn.model_selection import GridSearchCV  # Importing GridSearchCV from scikit-learn for hyperparameter tuning

from src.exception import CustomException  # Importing a custom exception class

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Getting the directory path from the file path

        os.makedirs(dir_path, exist_ok=True)  # Creating the directory if it doesn't exist

        with open(file_path, "wb") as file_obj:  # Opening the file in binary write mode
            pickle.dump(obj, file_obj)  # Serializing the object and saving it to the file

    except Exception as e:
        raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Creating an empty dictionary to store model evaluation scores

        for i in range(len(list(models))):
            model = list(models.values())[i]  # Getting the model from the dictionary
            para = param[list(models.keys())[i]]  # Getting the corresponding hyperparameters

            gs = GridSearchCV(model, para, cv=3)  # Performing grid search with cross-validation
            gs.fit(X_train, y_train)  # Fitting the model with the best hyperparameters

            model.set_params(**gs.best_params_)  # Setting the model's parameters to the best parameters found
            model.fit(X_train, y_train)  # Training the model

            y_train_pred = model.predict(X_train)  # Making predictions on the training data
            y_test_pred = model.predict(X_test)  # Making predictions on the test data

            train_model_score = r2_score(y_train, y_train_pred)  # Calculating the R-squared score for the training data
            test_model_score = r2_score(y_test, y_test_pred)  # Calculating the R-squared score for the test data

            report[list(models.keys())[i]] = test_model_score  # Storing the test model score in the report dictionary

        return report  # Returning the report containing the model evaluation scores

    except Exception as e:
        raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:  # Opening the file in binary read mode
            return pickle.load(file_obj)  # Loading the serialized object from the file

    except Exception as e:
        raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module
