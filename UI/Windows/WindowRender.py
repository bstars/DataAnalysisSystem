import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QPushButton, QMainWindow

from UI.Windows.ModelFrameWindow import ModelFrameWindow
from UI.Widgets.PCAWidget import PCAWidget
from UI.Widgets.LinearRegressionWidget import LinearRegressionWidget
from UI.Widgets.DataArea import DataArea
from DataUtil.DataHolder import DataHolder
from Util.ErrorMessage import print_error_message, Error

def render_window(classname):
    holder = DataHolder()
    window = ModelFrameWindow()
    window.setWindowTitle(classname
                          )
    if classname == "PCA":
        widget = PCAWidget(holder)
    elif classname == "Linear Regression":
        widget = LinearRegressionWidget(holder)
    else:
        raise Error("Algorithm not implemented")


    dataArea = DataArea(holder)


    window.insertWidget(0, dataArea)
    window.insertWidget(0, widget)


    return window

