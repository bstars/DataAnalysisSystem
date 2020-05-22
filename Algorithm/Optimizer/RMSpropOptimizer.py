from .Optimizer import Optimizer
import numpy as np

class RMSpropOptimizer(Optimizer):
    def __init__(self, decay_rate, learning_rate, max_iter):
        """
        :param decay_rate: decay rate which controls the exponential decay of accumulated gradient
                            typically [0.9, 0.99, 0.999]
        :param learning_rate: learning rate of gradient descent.
        :param max_iter: maximum number of iteration
        """
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = 1e-6


    def fit(self, W, fn_grad, callback=None):
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
        print("Running RMSpropOptimizer ...")
        W_dict = W
        eps = self.eps
        learning_rate = self.learning_rate
        decay_rate = self.decay_rate

        # initialize the accumulated gradient
        accumulated_grad = {}
        for key in W_dict.keys():
            accumulated_grad['d' + key] = np.zeros_like(W_dict[key])



        for iter in range(self.max_iter):
            grad_dict = dict(fn_grad(W_dict))
            for grad_key in grad_dict.keys():
                weight_key = grad_key[1:]

                accumulated = accumulated_grad[grad_key]
                # accumulated square gradient with exponential decay
                accumulated = (1 - decay_rate) * (grad_dict[grad_key] ** 2) + decay_rate * accumulated
                accumulated_grad[grad_key] = accumulated

                """
                The accumulated gradient is used to 'normalize' the computed gradient element-wise.
                Weights that receive large gradients will have their effective learning rate reduced;
                Weights that receive small gradients will have their effective learning rate increased.
                What's different with Adagrad is that RMSprop uses a moving average of squard gradient,
                which means that earlier gradient contributes less to current normalization.
                """
                W_dict[weight_key] -= learning_rate * grad_dict[grad_key] / (np.sqrt(accumulated) + eps)
                if callback is not None:
                    callback(iter, W_dict)
        return W_dict
