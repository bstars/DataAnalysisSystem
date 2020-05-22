import numpy as np
from Algorithm.Model.LinearRegression import LinearRegression
import matplotlib.pyplot as plt

m = 100



Xs = np.linspace(1, 4, m)
ones = np.ones(100)

ys = 0.5 * Xs + 2 + np.random.normal(size=[m]) * 0.05


# Xs = np.vstack([Xs, Xs]).T
# ys = 0.5 * Xs[:,0] + 0.3 * Xs[:,1]  + 0.1 + np.random.normal(size=[100]) * 0.05



lr = LinearRegression()
lr.setup(Xs, ys, 0)
# lr.fit_gradient_descent(0.02, max_iter=1000, optim='GradientDescent')
# lr.fit_gradient_descent(learning_rate=0.3, max_iter=200, optim="Adam", beta1=0.9, beta2=0.99)
# lr.fit_gradient_descent(learning_rate=0.15, max_iter=500, optim="RMSprop", decay_rate=0.9)
# lr.fit_gradient_descent(learning_rate=0.25, max_iter=1000, optim="Adagrad")
lr.fit_gradient_descent(learning_rate=0.1, max_iter=600, optim="MomentumGradientDescent", momentum=0.9)
# lr.fit_close_form()

print(lr.getWeight())



