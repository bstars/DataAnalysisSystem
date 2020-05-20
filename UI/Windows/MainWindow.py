import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from Algorithm.summary import algorithms
from UI.Windows.FrameWindow import FrameWindow
from UI.Windows.WindowRender import render_window
from Util.ErrorMessage import Error, print_error_message


class MainWindow(FrameWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setUpWindowContent()

    def setUpWindowContent(self):
        self.setWindowTitle("Data Analysis App")

        # mainLayout = QHBoxLayout()
        mainLayout = QGridLayout()
        typeLayouts = {}

        for algoType in algorithms.keys():

            typeLayout = QVBoxLayout()
            typeLayout.setSpacing(10)
            typeLayout.setAlignment(Qt.AlignTop)

            typeLabel = QLabel()
            typeLabel.setText(algoType)
            typeLabel.setFixedHeight(20)
            typeLayout.addWidget(typeLabel)

            for classname in algorithms[algoType]:
                button = QPushButton()
                button.setText(classname)

                # question about the lambda expression in loop
                # https: // stackoverflow.com / q / 49184163
                button.clicked.connect(lambda checked, arg = classname: self.createNewWindow(arg))
                typeLayout.addWidget(button)

            typeLayouts[algoType] = typeLayout


        mainLayout.addLayout(typeLayouts['regression'], 1,1)
        mainLayout.addLayout(typeLayouts['classification'], 1, 2)
        mainLayout.addLayout(typeLayouts['clustering'], 2, 1)

        # qwid.setLayout(mainLayout)
        self.setCentralLayout(mainLayout)

    def createNewWindow(self, classname:str):
        try:
            self.newwindow =  render_window(classname)
            self.newwindow.show()
        except Error as e:
            print_error_message(e.what())

if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()
    app.exec_()




