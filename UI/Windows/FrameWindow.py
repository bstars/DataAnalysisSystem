import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtCore import Qt
from Util.Params import Params

class FrameWindow(QMainWindow):
    def __init__(self):
        super(FrameWindow, self).__init__()
        self.w = Params.WINDOW_WIDTH
        self.h = Params.WINDOW_HEIGHT

        self.setFixedHeight(self.h)
        self.setFixedWidth(self.w)

        self.qwid = QWidget()
        self.setCentralWidget(self.qwid)

    def setUpWindowContent(self):
        pass

    def setCentralLayout(self, layout):
        self.qwid.setLayout(layout)

    def addWidget(self, widget):
        self.qwid.layout().addWidget(widget)



