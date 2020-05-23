import sys
sys.path.append("..")

import pandas as pd
from matplotlib import image
import numpy as np
from scipy.io import loadmat

from Util.ErrorMessage import Error, print_error_message

class DataHolder:

    def __init__(self):
        self.X = None
        self.y = None
        self.title = None
        self.data_loaded = False

    def give(self, X):
        self.X = X

    def loaded(self):
        return self.data_loaded

    def shape(self):
        return self.X.shape

    def parse_csv(self, filename):
        dataframe = pd.read_csv(filename, index_col=False)

        self.title = dataframe.keys()
        X  = dataframe[self.title].values
        self.X = X[:,:-1]
        self.y = X[:,-1]
        self.data_loaded = True

    def parse_mat(self, filename):
        dic = loadmat(filename)
        self.X = np.array(dic['X']).squeeze()
        try:
            self.y = np.array(dic['y']).squeeze()
        except:
            self.y = None
        self.data_loaded = True

    def fetchAll(self):
        if self.y is None:
            return self.X
        else:
            return np.vstack([self.X.T, self.y]).T

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
        self.X = np.array(X, dtype=np.float)
        self.data_loaded = True

    # def fetchAll(self, order=1):
    #     if order > 1:
    #         X = self.fetch_ordered(order)
    #         return X[:,:-1], X[:,-1], self.title
    #     return self.X[:,:-1], self.X[:,-1], self.title

    def fetchPlot(self):
        if self.y is not None:
            return np.vstack([self.X.T, self.y]).T, self.title
        else:
            return self.X, self.title

    def fetch_ordered(self, order=1):
        ret = np.array(self.X.copy())
        for i in range(2, order+1):
            if len(ret.shape)==1:
                ret = np.vstack([ret, np.power(self.X,i)]).T
            else:
                ret = np.vstack([ret.T, np.power(self.X, i)]).T
        return ret, self.y, self.title


