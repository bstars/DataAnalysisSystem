import numpy as np

class Optimizer:
    """
    Virtual Base class of all Optimizers:
        GradientDescentOptimizer
        MomentumGradientDescentOptimizer
        AdamOptimizer
    """
    def __init__(self):
        pass

    def fit(self, W, fn_grad):
        pass


