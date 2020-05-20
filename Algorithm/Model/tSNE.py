import numpy as np
import sys
from Algorithm.Optimizer.GradientDescentOptimizer import GradientDescentOptimizer
from Algorithm.Optimizer.MomentumGradientDescentOptimizer import MomentumGradientDescentOptimizer
from Util.ErrorMessage import *

sys.path.append("..")

# TODO: Figure out how to compute sigma's in tSNE and complete the implementation


class tSNE():
    def __init__(self, X, dim):
        """
        Perform t-SNE dimensionality reduction
        :param X:
        :param dim: Number of dimension of output
        """
        original_shape = X.shape
        m = original_shape[0]
        self.n = np.array(original_shape[1:])
        self.X = np.reshape(X, [m,-1])
        self.y_dict = {
            'y': np.random.normal(size=[m,dim])
        }

    def fit(self, optim, max_iter):
        pass





    def gradeint(self, y_dict, X, y=None):
        """
        Gradient callback used in Gradient Descent Optimizer is of form
            gradient(W_dict, X, y)
        To make this function compatible with Optimizers, we treat y as weight used in gradient function.
        The gradient of tSNE loss w.r.t y
        """
        pass

    def loss(self):
        pass



