import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QApplication, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt

from DataUtil.DataHolder import DataHolder
from UI.Widgets.LoadDataArea import LoadDataArea
from UI.Widgets.PlotDataArea import PlotDataArea
from UI.Windows.FrameWindow import FrameWindow
from Util.Params import Params


class DataArea(QWidget):
    def __init__(self, holder:DataHolder):
        super(DataArea, self).__init__()
        self.holder = holder


        self.plotDataArea = PlotDataArea(holder)
        self.loadDataArea = LoadDataArea(holder, self.plotDataArea)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.setupView()

    def setupView(self):
        self.loadDataArea.setFixedWidth(Params.LOAD_DATA_AREA_WIDTH)
        self.loadDataArea.setFixedHeight(Params.WINDOW_LOWER_PART_HEIGHT)

        self.plotDataArea.setFixedHeight(Params.PLOT_AREA_HEIGHT)
        self.plotDataArea.setFixedWidth(Params.PLOT_AREA_WIDTH)

        self.layout.addWidget(self.loadDataArea)
        self.layout.addWidget(self.plotDataArea)



if __name__ == "__main__":
    app = QApplication([])

    window = FrameWindow()
    layout = QVBoxLayout()
    window.setCentralLayout(layout)

    holder = DataHolder()


    dataArea = DataArea(holder)
    window.addWidget(dataArea)

    btn = QPushButton()
    btn.setText("hi")
    window.addWidget(btn)

    window.show()
    app.exec_()




