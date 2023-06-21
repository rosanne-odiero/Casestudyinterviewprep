
import logging 
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # Creating a log file name based on the current date and time
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # Creating the logs path by joining the current working directory, "logs" folder, and the log file name
os.makedirs(logs_path, exist_ok=True) # Creating the logs path directory if it doesn't exist

# Creating the log file path by joining the logs path and the log file name
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configuring the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Setting the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Setting the log message format
    level=logging.INFO,  # Setting the log level to INFO
)

if __name__=="__main__": 

    logging.info("Logging has started")
