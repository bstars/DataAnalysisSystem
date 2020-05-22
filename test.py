import numpy as np

a = np.ones(shape=[2,3])
b = np.array([2,2])

c = np.vstack([a.T,b]).T
print(c)