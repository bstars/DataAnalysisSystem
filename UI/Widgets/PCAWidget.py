import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QWidget


from Algorithm.Model.PrincipleComponentAnalysis import PCA
from DataUtil.DataHolder import DataHolder
from Util.Params import Params


class PCAWidget(QWidget):
    def __init__(self, holder:DataHolder):
        super(PCAWidget, self).__init__()
        self.pca = PCA()
        self.holder = holder

    def setupView(self):
        self.setFixedHeight(Params.WINDOW_UPPER_PART_HEIGHT)
