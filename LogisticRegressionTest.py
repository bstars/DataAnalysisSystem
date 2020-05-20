import numpy as np
from Algorithm.Model.LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt



m = 20
X1 = np.random.normal(size=[m, 2], loc=[.5, .5], scale=0.5)
plt.plot(X1[:,0], X1[:,1], 'ro')
y1 = np.ones(shape=[m])


X2 = np.random.normal(size=[m, 2], loc=[5, 5], scale=0.5)
plt.plot(X2[:,0], X2[:,1], 'bo')
y2 = np.zeros(shape=[m])

X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)


lr = LogisticRegression(X, y, regularization=0)

lr.fit(optim='Adam', learning_rate=0.2, beta1=0.9, beta2=0.99, max_iter=100)


print(lr.predict(X1))


weights = lr.getWeight()

xs = np.linspace( np.min(X[:,0]), np.max(X[:,0]), num=100)
ys = (-weights[0] * xs - weights[2]) / weights[1]

plt.plot(xs, ys, 'b--')
plt.show()



