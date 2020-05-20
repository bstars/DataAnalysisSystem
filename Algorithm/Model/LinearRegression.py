import sys
sys.path.append("../..")

import numpy as np

from Algorithm.Optimizer.OptimizerRender import render_optimizer
from Util.ErrorMessage import *
from .helper_functions import *


class LinearRegression():
    def __init__(self):
        pass

    def setup(self, X, y, regularization):
        self.y = y
        self.X = addBias(X)
        m,n = self.X.shape

        self.W_dict = {'W': np.random.normal(size=[n])}
        self.regularization = regularization

    def fit_gradient_descent(self, learning_rate, max_iter, optim, **kwargs):
        optimizer = render_optimizer(learning_rate, max_iter, optim, **kwargs)


        try:
            self.W_dict = optimizer.fit(self.W_dict, self.gradient)
        except Exception as e:
            print_error_message("LinearRegression, Gradient Error")


    def fit_close_form(self):
        X = self.X
        m,n = X.shape
        if m <= n:
            print_error_message("LinearRegression, X^T * X noninvertible")

        try:
            self.W_dict['W'] = np.linalg.inv(X.T @ X) @ X.T @ self.y
        except Exception as e:
            print_error_message('LinearRegression, Computation Error, cannot use close-form')

    def predict(self, X):
        Xb = addBias(X)
        return Xb @ self.W_dict['W']

    @classmethod
    def loss(self, y, yhat):
        return np.mean( (y-yhat) ** 2 )


    def gradient(self, W_dict):
        X = self.X
        y = self.y

        # forward pass
        m,n = X.shape
        W = W_dict['W']
        yhat = X @ W
        loss = self.loss(y, yhat)
        print("loss: ", loss)

        # backward pass
        dl_dyhat = 2 / m * (yhat - y)
        dl_dw = np.transpose(X) @ dl_dyhat
        dl_dw += self.regularization * 2 * W
        return {'dW': dl_dw}

    def getWeight(self):
        return self.W_dict['W']

    # @classmethod
    # def addBias(self, X):
    #     if len(X.shape) == 1:
    #         m = X.shape[0]
    #         Xb = np.vstack([X, np.ones(m)]).T
    #         n = 2
    #
    #     else:
    #         m, n = X.shape
    #         Xb = np.vstack([X.T, np.ones(m)]).T
    #         n += 1
    #     return Xb
