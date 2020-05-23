import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QPushButton, QCheckBox, QFormLayout
from PyQt5.QtWidgets import QLabel, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import numpy as np

from Algorithm.Model.PrincipleComponentAnalysis import PCA
from Util.Params import Params
from Util.ErrorMessage import print_error_message
from UI.Widgets.PlotDataArea import PlotDataArea
from DataUtil.DataHolder import DataHolder


class PCAWidget(QWidget):
    def __init__(self, holder:DataHolder):
        super(PCAWidget, self).__init__()
        self.model = PCA()
        self.holder = holder

        self.checkboxes = []
        self.setupView()

    def setupView(self):
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(Qt.AlignLeading)

        self.setLayout(self.layout)

        self.buttonWidget = self.setupButtonWidget()
        self.layout.addWidget(self.buttonWidget)

        self.infoWidget = self.setupInfoWidget()
        self.layout.addWidget(self.infoWidget)

        self.plotWidget = self.setupPlotWidget()
        self.layout.addWidget(self.plotWidget)

    def setupPlotWidget(self):
        plotWidget = PlotDataArea(DataHolder(), h=Params.WINDOW_UPPER_PART_HEIGHT)
        plotWidget.setFixedWidth(500)
        return plotWidget


    def setupButtonWidget(self):
        widget = QWidget()
        widget.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        widget.setFixedWidth(200)
        layout = QFormLayout()


        widget.setLayout(layout)

        self.fitButton = QPushButton()
        self.fitButton.setText("Fit")
        self.fitButton.clicked.connect(self.fit)
        layout.addRow(self.fitButton)

        self.firstKField = QLineEdit()
        self.firstKField.setValidator(QIntValidator())

        self.firstKButton = QPushButton("First K")
        self.firstKButton.clicked.connect(self.chooseFirstKEigenValues)
        layout.addRow(self.firstKField, self.firstKButton)


        self.projectButton = QPushButton()
        self.projectButton.setText("Project")
        layout.addRow(self.projectButton)

        self.plotEigenVectorButton = QPushButton()
        self.plotEigenVectorButton.setText("Plot Eigenvector")
        self.plotEigenVectorButton.clicked.connect(self.plotEigenValues)
        layout.addRow(self.plotEigenVectorButton)

        self.nearestField = QLineEdit()
        self.nearestField.setValidator(QIntValidator())
        self.nearestButton = QPushButton()
        self.nearestButton.setText("Nearest")
        layout.addRow(self.nearestField, self.nearestButton)

        self.keepRatioField = QLabel()
        layout.addRow("Keep Ratio:", self.keepRatioField)

        # self.eigenVecSizeField = QLabel()
        # layout.addRow("Eigenvector size:", self.eigenVecSizeField)

        return widget


    def setupInfoWidget(self):
        self.scroll = QScrollArea()
        self.scroll.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        self.scroll.setFixedWidth(200)
        self.scrollWidget = QWidget()
        # self.scrollWidget.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        self.scrollWidget.setFixedWidth(200)
        self.scrollLayout = QVBoxLayout()
        self.scrollWidget.setLayout(self.scrollLayout)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scrollWidget)
        return self.scroll
        # widget = QWidget()
        # widget.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
        # widget.setFixedWidth(200)
        #
        # layout = QVBoxLayout()
        # widget.setLayout(layout)
        #
        #
        #
        # self.scroll = QScrollArea()
        # self.scroll.setWidgetResizable(False)
        # layout.addWidget(self.scroll)
        #
        # return widget

    def fit(self):
        self.checkboxes = []
        self.clearPlot()

        if not self.holder.loaded():
            print_error_message("Please load data first")
            return

        X = self.holder.fetchAll()

        self.model.setup(X)
        self.model.fit()

        evs = self.model.getEigenvalues()
        self.showEigenValues(evs)

    def showEigenValues(self, evs):
        # scroll = self.scroll
        layout = self.scrollLayout
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        for i in range(len(evs)):
            checkbox = QCheckBox()
            self.checkboxes.append(checkbox)
            checkbox.setText("lamb_%d = %.2f" % (i, evs[i]))
            layout.addWidget(checkbox)

    def plotEigenValues(self):
        if not self.model.isFitted():
            print_error_message("Please fit the model first")
            return
        idx = self.getSelectedEigenValues()
        self.model.getEigenvectors()

        vecs = self.model.getEigenvectors()[idx]
        self.plotWidget.renderHolder().give(vecs)
        self.plotWidget.imshow()

    def getSelectedEigenValues(self):
        ret = []
        for i in range(len(self.checkboxes)):
            chbx = self.checkboxes[i]
            if chbx.isChecked():
                ret.append(i)
        selected_evs = np.array(self.model.getEigenvalues()[ret])
        all_evs = self.model.getEigenvalues()
        keep_ratio = np.sum(selected_evs) / np.sum(all_evs)
        self.keepRatioField.setText("%.4f"%(keep_ratio))

        return ret

    def clearPlot(self):
        layout = self.scrollLayout
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    def chooseFirstKEigenValues(self):

        k = self.firstKField.text()
        if k == "":
            print_error_message("K is required")
            return
        k = int(k)
        if not self.model.isFitted():
            print_error_message("Please fit the model first")
            return

        for i in range(k):
            self.checkboxes[i].setChecked(True)





