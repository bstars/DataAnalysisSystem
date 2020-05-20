import sys
sys.path.append("../..") # Adds higher directory to python modules path.


import numpy as np
from Algorithm.Optimizer.OptimizerRender import render_optimizer
from Util.ErrorMessage import *
from .helper_functions import *
from .NeuralNetworkHelperFunctions import *


class LogisticRegression():
    def __init__(self, X, y, regularization):
        self.y = y
        self.X = addBias(X)
        m, n = self.X.shape

        self.W_dict = {'W':np.random.normal(size=[n])}
        self.regularization = regularization


    def fit(self, optim, learning_rate, max_iter, **kwargs):
        optimizer = render_optimizer(learning_rate, max_iter, optim, **kwargs)

        try:
            self.W_dict = optimizer.fit(self.W_dict, self.gradient)
        except Exception as e:
            print_error_message("LogisticRegression: Gradient Error")

    def gradient(self, W_dict):

        X = self.X
        y = self.y
        m, n = X.shape
        # forward pass
        W = W_dict['W']
        yhat = sigmoid_forward(X @ W)
        loss = self.loss(y, yhat)
        print("loss: ", loss)

        dl_dyhat = yhat - y
        dl_dw = np.transpose(X) @ dl_dyhat
        dl_dw += self.regularization * 2 * W
        return {'dW':dl_dw}


    def predict(self, X):
        Xb = addBias(X)
        z = Xb @ self.W_dict['W']
        yhat = sigmoid_forward(z)
        return np.ones_like(yhat) * (yhat>=0.5)


    @classmethod
    def loss(self, y, yhat):
        return np.mean(
            -y * np.log(yhat + 1e-6) - (1. - y) * np.log(1. - yhat + 1e-6)
        )

    def getWeight(self):
        return self.W_dict['W']

