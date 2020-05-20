from Algorithm.Model.tSNE import tSNE
import numpy as np

X = np.ones(shape=[10, 28, 28])

tsne = tSNE(X, 3)