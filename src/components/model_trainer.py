import os  # Importing the os module for file and directory operations
import sys  # Importing the sys module for system-specific parameters and functions
from dataclasses import dataclass  # Importing the dataclass decorator for creating data classes

from catboost import CatBoostRegressor  # Importing CatBoostRegressor from the catboost library
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)  # Importing ensemble regressors from scikit-learn
from sklearn.linear_model import LinearRegression  # Importing LinearRegression from scikit-learn
from sklearn.metrics import r2_score  # Importing r2_score from scikit-learn for model evaluation
from sklearn.neighbors import KNeighborsRegressor  # Importing KNeighborsRegressor from scikit-learn
from sklearn.tree import DecisionTreeRegressor  # Importing DecisionTreeRegressor from scikit-learn
from xgboost import XGBRegressor  # Importing XGBRegressor from the xgboost library

from src.exception import CustomException  # Importing a custom exception class
from src.logger import logging  # Importing the logging module for logging purposes

from src.utils import save_object, evaluate_models  # Importing custom utility functions

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Setting the default trained model file path

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Creating an instance of ModelTrainerConfig

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")  # Logging an info message
# Splitting the train and test input data
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )  
# Creating a dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }  
# Creating a dictionary of hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }  
# Evaluating models and getting a report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)  
# Getting the best model score from the report

            best_model_score = max(sorted(model_report.values()))  

 # Getting the name of the best model from the report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 
# Retrieving the best model based on the name

            best_model = models[best_model_name]  
 # Raising an exception if the best model score is less than 0.6

            if best_model_score < 0.6:
                raise CustomException("No best model found") 
            
            logging.info(f"Best found model on both training and testing dataset")  # Logging an info message

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )  # Saving the trained model object

            predicted = best_model.predict(X_test)  # Making predictions using the best model

            r2_square = r2_score(y_test, predicted)  # Calculating the R-squared score

            return r2_square  # Returning the R-squared score

        except Exception as e:
            raise CustomException(e, sys)  # Raising a custom exception with the original exception and sys module
