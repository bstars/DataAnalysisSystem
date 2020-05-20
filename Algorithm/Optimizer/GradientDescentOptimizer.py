from .Optimizer import Optimizer

class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate, max_iter):
        """
        :param learning_rate: learning rate of gradient descent.

        :param max_iter: maximum number of iteration
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter


    def fit(self, W, fn_grad):
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
        print("Running GradientDescentOptimizer ...")
        W_dict = W

        learning_rate = self.learning_rate


        for iter in range(self.max_iter):
            grad_dict = dict(fn_grad(W_dict))
            for key in grad_dict.keys():
                weight_key = key[1:]
                W_dict[weight_key] -= learning_rate * grad_dict[key]

        return W_dict


