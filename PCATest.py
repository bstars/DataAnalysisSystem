from Algorithm.Model.PrincipleComponentAnalysis import PCA
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    data = loadmat('./data/ex7faces.mat')['X']
    m, n = data.shape
    n = int(np.sqrt(n))
    data = np.reshape(data, newshape=[m, n, n])
    data = np.transpose(data, [0,2,1])
    print(data.shape)
    return data

def showData(data, numrow, numcol, labels=None):

    data = data[:numrow * numcol,...]
    # m, n = data.shape
    # n = int(np.sqrt(n))
    # data = np.reshape(data, newshape=[m,n,n])

    fig = plt.figure(figsize=(9, 13))

    for i in range(numrow):
        for j in range(numcol):
            idx = i * numcol + j
            img = data[idx]
            ax = fig.add_subplot(numrow, numcol, idx + 1)
            if labels is not None:
                ax.set_title(str(labels[idx]))
            ax.imshow(img, cmap='gray')
    plt.show()


X = loadData()
pca = PCA()
pca.setup(X)
pca.fit()


idx = 269
num_vecs = 36

# proj = pca.project(X, num_vecs)
# reconstruct = pca.reconstruct(proj)
# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.imshow(X[idx], cmap='gray')
# ax2.imshow(reconstruct[idx], cmap='gray')
# plt.show()

eigenvecs = pca.getEigenvectors(num_vecs)
showData(eigenvecs, 6, 6)



# nearest_faces = pca.nearest(X, X[idx], num_vecs, 4)
# showData(nearest_faces, 2, 2)
#
# print(pca.getEigenvalues())
