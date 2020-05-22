import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat



m = 300

Xs = np.linspace(1, 6, m)
ones = np.ones(100)

ys = -Xs + 2 * Xs**2 - 0.3 * Xs**3
# ys = np.sin(Xs)

savemat(
    'LinearReg.mat',
    {'X':Xs, 'y':ys}
)

for i in range(2,2+1):
    print(i)
