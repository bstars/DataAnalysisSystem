import numpy as np
import sys
sys.path.append("..")

from Algorithm.ConvexOptimization.UnconstrainedOptimization import steepest_descent
import Util.plot_ellipsoid as pesd

sigma = np.array([
    [1,0],
     [0,6]
])

u = np.array([1,1])

P = np.array([
    [0.9, 0],
    [0, 6.1]
])

def f(x):
    return (x-u) @ sigma @ (x-u)

def gradient(x):
    return 2 * (sigma @ (x-u))

def feasible(x):
    return True


x0 = np.array([4,6])

x, vals, steps = steepest_descent(f, gradient, feasible, x0, norm='L2', normalized=False)
x, vals, steps = steepest_descent(f, gradient, feasible, x0, norm='quad', normalized=fa, P=P)


pesd.plot_2d_ellipsoid(u,sigma, np.linspace(0.5, 10.5, 10))
pesd.show()