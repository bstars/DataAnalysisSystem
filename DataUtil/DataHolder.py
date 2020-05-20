import sys
sys.path.append("..")

import pandas as pd
from matplotlib import image
import numpy as np

from Util.ErrorMessage import Error, print_error_message

class DataHolder:

    def __init__(self):
        self.data = None
        self.title = None

    def parse_csv(self, filename):
        dataframe = pd.read_csv(filename, index_col=False)

        self.title = dataframe.keys()
        self.data  = dataframe[self.title].values


    def parse_mat(self, filename):
        pass

    def parse_images(self, filenames):

        X = []
        for i in range(len(filenames)):
            f = filenames[i]
            img = image.imread(f)

            if i == 0:
                self.shape = img.shape
            else:
                if img.shape != self.shape:
                    raise Error("Images must have the same shape.")

            X.append(img)
        self.data = np.array(X)





    def fetchAll(self):
        return self.data, self.title

