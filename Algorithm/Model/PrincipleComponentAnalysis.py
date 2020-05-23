import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from Util.ErrorMessage import *

class PCA():
    """
    Principle Component Analysis
    """
    def __init__(self):
        self.fitted = False

    def isFitted(self):
        return self.fitted

    def setup(self, X):
        original_shape = X.shape
        m = original_shape[0]
        self.n = np.array(original_shape[1:])
        print("n:",self.n)
        self.X = self.normalize(np.reshape(X, [m,-1]))  # Normalize the data before running PCA

    @classmethod
    def normalize(self,x):
        mean = np.mean(x)
        x -= mean
        cov = np.mean(x ** 2)
        x /= np.sqrt(cov)
        return x



    def fit(self):
        """
        Perform Singular Value Decomposition on X
        X = U * Lamb * Vt where U and V are orthonormal matrices, and Lamb is a diagonal matrix
        X^T * X = V * Lamb^T * U^T * U * Lamb * V^T = V * Lamb^2 * V^T
        V's columns are eigenvalues of X^T*X

        :return: eigenvalues and eigenvectors of X^T * X
        """
        U, lamb, Vt = np.linalg.svd(self.X)
        print(Vt.shape, U.shape, lamb.shape)
        self.eigenvectors = Vt

        self.lamb = lamb
        self.fitted = True

    def project(self, X, k):
        """
        :param X:
        :param k: Number of principle component (eigenvectors)
        :return: The projection of X onto principle components
        """
        m = X.shape[0]
        n = X.shape[1:]

        # if n!=self.n:
        #     pass

        X = np.reshape(X, [m,-1])

        retain = np.sum(self.lamb[:k] ** 2)
        all = np.sum(self.lamb** 2)
        print("PCA_Project: Variation retained: %.2f" % float(retain / all))    # Retained variation
        return np.array(X @ np.transpose(self.eigenvectors[:k]))

    def projectOnEigenvectors(self, X, vecs):
        return np.array(X @ np.transpose(vecs))

    def reconstruct(self, proj):
        """
        :param proj: Projection on each principle components
        :return: Reconstruction from projection and eigenvectors
        """
        m, k = proj.shape
        ret = proj @ self.eigenvectors[:k]

        shape = np.insert(self.n, 0, m)
        return np.reshape(ret, shape)



    def getEigenvectors(self, k=None):
        if k is None:
            ret = self.eigenvectors
        else:
            ret = self.eigenvectors[:k]

        m = ret.shape[0]
        shape = np.insert(self.n, 0, m)
        return np.reshape(ret, shape)

    def getEigenvalues(self):
        return self.lamb

    def nearest(self,X,x_ground,k,num):
        if num >= len(X):
            print_error_message("PrincipleComponentAnalysis, too large a number")


        X = np.concatenate([X, [x_ground]], axis=0)
        proj = self.project(X, k)

        diff = np.sum( (proj - proj[-1]) ** 2, axis=1)
        idx = np.argsort(diff)[1:k+1]
        return X[idx]
