import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from DataUtil.DataHolder import DataHolder
from UI.Windows.FrameWindow import FrameWindow
from UI.Widgets.DataArea import DataArea

from Util.Params import Params

class ModelFrameWindow(FrameWindow):
    def __init__(self):
        super(ModelFrameWindow, self).__init__()
        self.holder = DataHolder()
        self.setupView()

    def setupView(self):
        self.dataArea = DataArea(self.holder)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(2,2,2,2)
        self.layout.setSpacing(5)
        self.layout.setAlignment(Qt.AlignLeft)
        self.layout.addStretch(1)
        self.setCentralLayout(self.layout)

    def insertWidget(self, idx, widget):
        self.layout.insertWidget(idx, widget, alignment=Qt.AlignBottom)















