from .GradientDescentOptimizer import GradientDescentOptimizer
from .AdagradOptimizer import AdagradOptimizer
from .MomentumGradientDescentOptimizer import MomentumGradientDescentOptimizer
from .RMSpropOptimizer import RMSpropOptimizer
from .AdamOptimizer import AdamOptimizer

optimizers = [
    "GradientDescent", "MomentumGradientDescent", "Adagrad", "RMSprop", "Adam"
]

def render_optimizer(learning_rate, max_iter, optim, **kwargs):
    if optim == "GradientDescent":
        optimizer = GradientDescentOptimizer(learning_rate, max_iter)
    elif optim == "MomentumGradientDescent":
        momentum = kwargs.pop('momentum', 0.9)
        optimizer = MomentumGradientDescentOptimizer(momentum, learning_rate, max_iter)
    elif optim == "Adagrad":
        optimizer = AdagradOptimizer(learning_rate, max_iter)
    elif optim == "RMSprop":
        decay_rate = kwargs.pop('decay_rate', 0.9)
        optimizer = RMSpropOptimizer(decay_rate, learning_rate, max_iter)
    elif optim == "Adam":
        beta1 = kwargs.pop('beta1', 0.9)
        beta2 = kwargs.pop('beta2', 0.99)
        optimizer = AdamOptimizer(beta1, beta2, learning_rate, max_iter)
    else:
        optimizer = GradientDescentOptimizer(learning_rate, max_iter)
    return optimizer