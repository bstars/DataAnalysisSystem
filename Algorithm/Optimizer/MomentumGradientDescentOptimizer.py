from .Optimizer import Optimizer
import numpy as np

class MomentumGradientDescentOptimizer(Optimizer):
    def __init__(self, momentum, learning_rate, max_iter):
        """
        :param momentum: momentum
        :param learning_rate: learning rate of gradient descent.
        :param max_iter: maximum number of iteration
        """
        self.momentum = momentum
        self.learning_rate = learning_rate

        self.max_iter = max_iter




    def fit(self, W, fn_grad):
        """
        :param W: A dictionary which contains the initial value of weights
        :param X: X
        :param y: y
        :param fn_grad: A function
                    def gradient(W_dict, X, y):
                        return grad
                which takes in the X, y and current value of weights,
                and returns the gradient as a dictionary.
                eg. return {"dW1":dW1, "dW2":dW2}
        :return: Run gradient descent.
        """
        print("Running GradientDescentOptimizer ...")
        W_dict = W
        momentum = self.momentum

        dW_prev_dict = {}
        for key in W.keys():
            gradient_key = "d" + key
            dW_prev_dict[gradient_key] = np.zeros_like(W_dict[key])



        for iter in range(self.max_iter):
            grad_dict = dict(fn_grad(W_dict))
            for grad_key in grad_dict.keys():
                weight_key = grad_key[1:]
                v = self.learning_rate * grad_dict[grad_key] + momentum * dW_prev_dict[grad_key]
                W_dict[weight_key] -= v
                dW_prev_dict[grad_key] = v

        return W_dict
