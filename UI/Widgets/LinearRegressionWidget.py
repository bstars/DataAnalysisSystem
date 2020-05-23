import sys
sys.path.append("../..")


from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QFormLayout, QPushButton, QFileDialog
from PyQt5.QtWidgets import QScrollArea, QTextEdit, QComboBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.io import savemat

from DataUtil.DataHolder import DataHolder
from Algorithm.Model.LinearRegression import LinearRegression
from Util.Params import Params
from Util.ErrorMessage import print_error_message
from Algorithm.Optimizer.OptimizerRender import optimizers


class LinearRegressionWidget(QWidget):
    def __init__(self, holder:DataHolder):
        super(LinearRegressionWidget, self).__init__()
        self.holder = holder
        self.model = LinearRegression()
        self.setupView()

    def setupView(self):
        self.layout = QHBoxLayout()
        #self.layout.setSpacing(10)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setAlignment(Qt.AlignLeading)

        self.setLayout(self.layout)

        self.infoWidget = self.buildInfoWidget()
        self.layout.addWidget(self.infoWidget)

        self.historyWidget = self.buildHistoryWidget()
        self.layout.addWidget(self.historyWidget)

        self.plotWidget = self.buildPlotWidget()
        self.layout.addWidget(self.plotWidget)



    def buildInfoWidget(self):
        infoWidget = QWidget()
        infoWidget.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        infoWidget.setFixedWidth(250)

        layout = QFormLayout()
        self.regField = QLineEdit()
        self.regField.setValidator(QDoubleValidator())

        layout.addRow("Regularization:", self.regField)

        self.orderField = QLineEdit()
        self.orderField.setValidator(QIntValidator())
        self.orderField.setText('1')
        layout.addRow("Order:", self.orderField)


        self.learningrateField = QLineEdit()
        self.learningrateField.setValidator(QDoubleValidator())
        self.learningrateField.setText("0.1")
        layout.addRow("Learning Rate:", self.learningrateField)
        # self.trainingPortion = QLineEdit()
        # layout.addRow("Training set portion:", self.trainingPortion)

        self.optimizerField = QComboBox()
        for opt in optimizers:
            self.optimizerField.addItem(opt)
        layout.addRow("Optimizer:", self.optimizerField)

        self.maxiterField = QLineEdit()
        self.maxiterField.setValidator(QIntValidator())
        self.maxiterField.setText('200')
        layout.addRow("Max Iter:", self.maxiterField)


        self.fitButton = QPushButton()
        self.fitButton.setText("Fit")
        self.fitButton.clicked.connect(self.fit)
        layout.addWidget(self.fitButton)

        self.saveButton = QPushButton()
        self.saveButton.setText('Save Model')
        self.saveButton.clicked.connect(self.save)
        layout.addWidget(self.saveButton)

        infoWidget.setLayout(layout)
        return infoWidget

    def buildHistoryWidget(self):
        # hisWidget = QWidget()
        # hisWidget.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        # hisWidget.setFixedWidth(150)
        #
        # layout = QHBoxLayout()
        # hisWidget.setLayout(layout)

        scroll = QScrollArea()
        scroll.setFixedWidth(200)
        scroll.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)

        layout = QHBoxLayout()

        self.historyField = QTextEdit()
        layout.addWidget(self.historyField)

        scroll.setLayout(layout)

        return scroll

    def buildPlotWidget(self):
        widget = QWidget()
        widget.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        widget.setFixedWidth(500)
        layout = QHBoxLayout()
        widget.setLayout(layout)

        fig = plt.Figure(figsize=[5,5])
        self.fig = fig
        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(canvas.size())
        self.ax = fig.add_subplot()

        layout.addWidget(canvas)
        return widget



    def fit(self):
        self.historyField.setText("")
        if not self.holder.loaded():
            print_error_message("Please load data first")
            return

        try:
            self.reg = float(self.regField.text())
            print("reg: ", self.reg)
        except:
            print_error_message("Invalid regularization, must be a real number")
            return

        try:
            self.order = int(self.orderField.text())
            print("order: ", self.order)
        except:
            print_error_message("Invalid order, must be a integer")
            return

        try:
            self.learning_rate = float(self.learningrateField.text())
            print("learning rate: ", self.learning_rate)
        except:
            print_error_message("Invalid learning rate, must be a real number")

        try:
            self.maxiter = int(self.maxiterField.text())
            print("max iter: ", self.maxiter)
        except:
            print_error_message("Invalid max iter, must be a real number")

        self.optimizer = self.optimizerField.currentText()
        print("optimizer: ", self.optimizer)

        X, y, title = self.holder.fetch_ordered(self.order)


        self.model.setup(X, y, regularization=self.reg)

        def callback(iter, loss):
            self.historyField.append("iter: %d, loss: %.2f"%(iter, loss))
        self.model.fit_gradient_descent(self.learning_rate, self.maxiter, self.optimizer, callback=callback)


        if len(self.holder.shape()) == 1:
            self.plot()

    def save(self):
        if not self.model.set():
            print_error_message("Please train the model")
            return
        path = QFileDialog.getSaveFileName(
            self, "Save file", "", ".mat"
        )[0]

        savemat(path, self.model.getWeightDic())


    def plot(self):
        print('hi')
        self.ax.cla()

        X, y, title = self.holder.fetch_ordered(1)
        self.ax.plot(X,y, 'ro', markersize=1)

        Xorder, y, title = self.holder.fetch_ordered(self.order)
        ypred = self.model.predict(Xorder)
        self.ax.plot(X, ypred, 'b--')
        self.fig.canvas.draw_idle()







