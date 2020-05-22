from .Optimizer import Optimizer
import numpy as np

class AdagradOptimizer(Optimizer):
    """
    AdagradOptimizer which can use a comparably large learning rate at first.
    """

    def __init__(self, learning_rate, max_iter):
        """
        :param learning_rate: learning rate of gradient descent.
        :param max_iter: maximum number of iteration
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = 1e-6


    def fit(self, W, fn_grad, callback=None):
        """
        :param W: A dictionary which contains the initial value of weights
        :param X: X
        :param y: y
        :param fn_grad: A function
                    def gradient(W_dict):
                        return grad
                which takes in the X, y and current value of weights,
                and returns the gradient as a dictionary.
                eg. return {"dW1":dW1, "dW2":dW2}

        :return: Run gradient descent.
        """
        print("Running AdagradOptimizer ...")
        W_dict = W
        eps = self.eps
        learning_rate = self.learning_rate

        # initialize the accumulated gradient
        accumulated_grad = {}
        for key in W_dict.keys():
            accumulated_grad['d' + key] = np.zeros_like(W_dict[key])



        for iter in range(self.max_iter):
            grad_dict = dict(fn_grad(W_dict))
            for grad_key in grad_dict.keys():
                weight_key = grad_key[1:]

                accumulated = accumulated_grad[grad_key]
                accumulated += grad_dict[grad_key] ** 2
                accumulated_grad[grad_key] = accumulated

                """
                The accumulated gradient is used to 'normalize' the computed gradient element-wise.
                Weights that receive large gradients will have their effective learning rate reduced;
                Weights that receive small gradients will have their effective learning rate increased.
                """
                W_dict[weight_key] -= learning_rate * grad_dict[grad_key] / (np.sqrt(accumulated) + eps)
                if callback is not None:
                    callback(iter, W_dict)
        return W_dict
