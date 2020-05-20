import numpy as np

def addBias(X):
    if len(X.shape) == 1:
        m = X.shape[0]
        Xb = np.vstack([X, np.ones(m)]).T
        n = 2

    else:
        m, n = X.shape
        Xb = np.vstack([X.T, np.ones(m)]).T
        n += 1
    return Xb

