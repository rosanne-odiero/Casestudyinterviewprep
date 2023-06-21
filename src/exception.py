import sys  # Importing the sys module
from logger import logging  # Importing the logging module from src.logger

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Getting the exception traceback
    file_name = exc_tb.tb_frame.f_code.co_filename  # Getting the filename where the error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))  # Creating the error message with the file name, line number, and error

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Generating error message using the error_message_detail function
    
    def __str__(self):
        return self.error_message

#if __name__=="__main__": 
#    try:
#        a=1/0
#    except Exception as e:
#        logging.info("divide by zero")
#       raise CustomException(e,sys)
