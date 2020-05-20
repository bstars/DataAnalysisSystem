from .Optimizer import Optimizer
import numpy as np

class AdamOptimizer(Optimizer):
    def __init__(self, beta1, beta2, learning_rate, max_iter):
        """
        :param beta1: controls the exponential decay for the momentum part, typically beta1 = 0.9
        :param beta2: controls the exponential decay for the RMSprop part, typically beta1 = 0.99

        :param max_iter: maximum number of iteration
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-6

    def fit(self, W, fn_grad):
        """
        :param learning_rate: learning rate of gradient descent.
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
        print("Running AdamOptimizer ...")
        W_dict = W

        eps = self.eps
        learning_rate = self.learning_rate

        beta1 = self.beta1
        beta2 = self.beta2

        # initialize the accumulated gradient
        accumulated_grad = {}
        momentum = {}
        for key in W_dict.keys():
            accumulated_grad['d' + key] = np.zeros_like(W_dict[key])
            momentum['d' + key] = np.zeros_like(W_dict[key])


        for iter in range(self.max_iter):
            grad_dict = dict(fn_grad(W_dict))
            for grad_key in grad_dict.keys():
                weight_key = grad_key[1:]

                # Momentum part
                m = momentum[grad_key]
                m = beta1 * m + (1 - beta1) * grad_dict[grad_key]   # exponential decay for momentum
                mbc = m / (1 - beta1 ** (iter+1))                       # bias correction for exponential decay of momentum
                momentum[grad_key] = m


                # RMSprop part
                accumulated = accumulated_grad[grad_key]

                # exponential decay for accumulated square gradient
                accumulated = beta2 * accumulated + (1 - beta2) * (grad_dict[grad_key] ** 2)
                accumulated_bc = accumulated / (1 - beta2 ** (iter+1))  # bias correction for exponential decay
                accumulated_grad[grad_key] = accumulated

                W_dict[weight_key] -= learning_rate * mbc / (np.sqrt(accumulated_bc) + eps)
        return W_dict





