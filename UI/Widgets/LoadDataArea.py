import sys
sys.path.append("../..")

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import Qt

from UI.Windows.FrameWindow import FrameWindow
from UI.Widgets.PlotDataArea import PlotDataArea

from Util.Params import Params
from DataUtil.DataHolder import DataHolder
from Util.ErrorMessage import print_error_message, Error



class LoadDataArea(QWidget):
    def __init__(self, holder:DataHolder, partener:PlotDataArea):
        super(LoadDataArea, self).__init__()
        self.partener = partener
        self.holder = holder
        self.setupView()

    def setupView(self):
        self.setFixedHeight(Params.WINDOW_LOWER_PART_HEIGHT)
        self.layout = QVBoxLayout()
        btn_csv = QPushButton()
        btn_csv.setText("Load CSV")
        btn_csv.clicked.connect(self.load_csv)

        btn_mat = QPushButton()
        btn_mat.setText("Load mat")
        btn_mat.clicked.connect(self.load_mat)

        btn_img = QPushButton()
        btn_img.setText("Load images")
        btn_img.clicked.connect(self.load_imgs)



        self.layout.addWidget(btn_csv)
        self.layout.addWidget(btn_mat)
        self.layout.addWidget(btn_img)
        self.setLayout(self.layout)

    def load_csv(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file',
                                            '~', "CSV files (*.csv)")
        if filename[0] != "":
            try:
                self.holder.parse_csv(filename[0])
                self.partener.histogram()
            except Error as e:
                print_error_message(e.what())


    def load_mat(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file',
                                               '~', "mat files (*.mat)")
        if filename[0] != "":
            try:
                self.holder.parse_mat(filename[0])
                self.partener.histogram()
            except Error as e:
                print_error_message(e.what())

    def load_imgs(self):
        files = QFileDialog.getOpenFileNames(self, 'Open file',
                                            '~', "Image files (*.jpg *.jpeg *.png)")
        filenames = files[0]
        if len(filenames) != 0:
            try:
                self.holder.parse_images(filenames)
                self.partener.imshow(num=min(10, len(filenames)))
            except Error as e:
                print_error_message(e.what())








