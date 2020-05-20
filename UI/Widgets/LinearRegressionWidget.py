import sys
sys.path.append("../..")


from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QFormLayout, QPushButton
from PyQt5.QtGui import QDoubleValidator

from DataUtil.DataHolder import DataHolder
from Algorithm.Model.LinearRegression import LinearRegression


class LinearRegressionWidget(QWidget):
    def __init__(self, holder:DataHolder):
        super(LinearRegressionWidget, self).__init__()
        self.holder = holder
        self.model = LinearRegression()
        self.setupView()

    def setupView(self):
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.infoLayout = self.buildInfoLayout()
        self.layout.addLayout(self.infoLayout)

    @classmethod
    def buildInfoLayout(self):
        layout = QFormLayout()

        self.regField = QLineEdit()
        self.regField.setValidator(QDoubleValidator())

        layout.addRow("Regularization:", self.regField)

        self.trainingPortion = QLineEdit()
        layout.addRow("Training set portion:", self.trainingPortion)

        self.numOfParam = QLabel()
        layout.addRow("Number of parameters:", self.numOfParam)

        self.numOfTrainingSamples = QLabel()
        layout.addRow("Training set size:", self.numOfTrainingSamples)

        self.fitButton = QPushButton()
        self.fitButton.setText("Fit")
        self.fitButton.clicked.connect(self.fit)
        layout.addWidget(self.fitButton)

        return layout

    def fit(self):
        X,title = self.holder.fetchAll()




