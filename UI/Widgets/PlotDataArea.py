import sys
sys.path.append("../..")


from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QScrollArea, QHBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

from UI.Windows.FrameWindow import FrameWindow

from Util.Params import Params
from Util.ErrorMessage import print_error_message
from DataUtil.DataHolder import DataHolder

plt.rcParams.update({'font.size': 5})


class PlotDataArea(QScrollArea):
    def __init__(self, holder:DataHolder):
        super(PlotDataArea, self).__init__()
        self.h = Params.PLOT_AREA_HEIGHT
        self.holder = holder
        self.setFixedHeight(self.h)

        self.scrollWidget = QWidget()
        self.scrollWidget.setFixedHeight(self.h)
        self.scrollLayout = QHBoxLayout()
        self.scrollWidget.setLayout(self.scrollLayout)
        self.setWidgetResizable(True)
        self.setWidget(self.scrollWidget)

    def plot(self):
        self.clear()
        X, title = self.holder.fetchAll()
        m = len(X)
        if title is None:
            title = ["" for i in range(m)]
        for i in range(m):
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(canvas.size())
            ax = fig.add_subplot()
            ax.plot(X[i])
            self.addWidget(canvas)

    def imshow(self, num=None):
        self.clear()
        X, title = self.holder.fetchAll()
        m = len(X)

        cmap = None
        if len(X.shape) == 3:
            cmap = 'gray'

        if num is None:
            m = min(10, m)

        for i in range(m):
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(canvas.size())
            ax = fig.add_subplot()
            ax.imshow(X[i], cmap=cmap)
            self.addWidget(canvas)


    def histogram(self):
        self.clear()
        X, title = self.holder.fetchAll()
        X = X.T
        m = len(X)
        if title is None:
            title = ["" for i in range(m)]

        for i in range(m):
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(canvas.size())
            ax = fig.add_subplot()
            ax.hist(X[i], bins=70, label=title[i])
            ax.legend()
            self.addWidget(canvas)




    def addWidget(self, widget:QWidget):
        scale = self.h / widget.height()
        widget.setFixedHeight(widget.height() * scale - 15)
        widget.setFixedWidth(widget.width() * scale - 15)
        self.scrollLayout.addWidget(widget)

    def clear(self):
        layout = self.scrollLayout
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)





