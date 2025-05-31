import sys
# import
# src.logger import logging

def error_message_details(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_name = exc_tb.tb_lineno
    error_message  = str(error)
    error_message = "Error occurred in python script: [{0}] at line number: [{1}] with error message: [{2}]".format(
        file_name,
        line_name,
        error_message
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details = error_details)

    def __str__(self):
        return self.error_message