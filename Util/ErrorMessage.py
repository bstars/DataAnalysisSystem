#TODO: Error processing

from PyQt5.QtWidgets import QMessageBox

class Error(Exception):
    def __init__(self, msg:str):
        self.msg = msg

    def what(self):
        return self.msg

def print_error_message(str):
    error_str = " Error: " + str

    msg = QMessageBox()
    msg.setText(error_str)
    msg.exec_()

