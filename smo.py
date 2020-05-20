import numpy as np
import matplotlib.pyplot as plt

def selectJrand(i,m):
    j = i
    while j==i:
        j = int(np.random.uniform(0,m))
    return j

def smo(X, y, C, tol=1e-4, maxiter=1000):
    """
    primal problem:

        minimize 1/2 * ||w||_2^2 + C * \sum_i eps_i
        subject to
            yi(W * xi + b) >= 1 - eps
            eps_i >= 0



    dual problem:
    (let alpha be dual variables associated with yi(W * xi + b) >= 1 - eps_i
         beta be dual variables associated with eps_i >= 0)

    minimize -1/2 \sum_i \sum_j alpha_i * alpha_j * yi * yj * <xi, xj> + sum_i alpha_i
    subject to
        alpha_i >= 0                        -|
        beta_i >= 0                          |-->  \sum_i alpha_i * yi = 0
        \sum_i alpha_i * yi = 0              |-->  0 <= alpha_i <= C
        C - alpha_i - beta_i = 0            -|



    :param X: Training data
    :param y: Training label
    :param C: Trade-off between ||w|| and eps
    :param tol: Tolerance to compare to 0
    :param maxiter:
    :return:
        w: SVM weights
        b: SVM bias
        alphas: dual variables
        idx1: idx of support vectors in positive group
        idx2: idx of support vectors in negative group
    """


    # TODO: Kernel function

    b = 0
    m,n = X.shape
    alphas = np.zeros(shape=[m])    # dual variables
    iter = 0

    while iter < maxiter:
        alphaPairChanged = 0

        for i in range(m):
            alpha_y = alphas * y  # [m, ]

            xi = X[i,:]
            yi = y[i]
            ai = alphas[i].copy()


            inner_i = X @ xi
            # [m, ]
            # inner product of each sample with xi, can be replaced by a kernel function
            fxi = alpha_y @ inner_i + b

            # In Lagrangian, we set W = \sum_i alpha_i * yi * xi
            # W xj + b = \sum_i alpha_i * yi * <xi, xj> + b

            # KKT condition
            #       Primal constraints:
            #           yi(W * xi + b) >= 1 - eps_i     (dual variable alpha)
            #           eps_i >= 0                      (dual variable beta)
            #       Dual constraints:
            #           alpha_i >= 0
            #           beta_i >= 0
            #           \sum_i alpha_i * yi = 0
            #           C - alpha_i - beta_i = 0
            #
            #       Complementary slackness:
            #           if alpha_i = 0
            #               beta_i = C, eps_i = 0, yi(W * xi + b) >= 1 - eps_i = 1
            #               KKT violation #1, when alpha_i>0 and yi(W * xi + b) > 1
            #           if alpha_i = C
            #               beta_i = 0, eps_i >= 0, yi(W * xi + b) = 1 - eps_i <= 1
            #               KKT violation #2, when alpha_i<C and yi(W * xi + b) < 1
            #           if 0 < alpha_i < C
            #              C > beta_i > 0, eps_i = 0, yi(W * xi + b) = 1 - eps_i = 1
            slack_ness = yi * fxi - 1

            if (ai>0 and slack_ness > tol) or (ai<C and slack_ness <-tol):
                print('hi')
                j = selectJrand(i, m)
                xj = X[j,:]
                yj = y[j]
                aj = alphas[j].copy()
                inner_j = X @ xj

            #     if yi != yj:
            #         #
            #         L = max(0, aj - ai)
            #         H = min(C, aj - ai + C)
            #     else:
            #         # yj + yi = yi_old + yj_old
            #         # upper-left - lower-right case
            #         L = max(0, ai + aj - C)
            #         H = min(C, ai + aj)
            #     if L==H:
            #         continue
            #
            #     eta = xi @ xi + xj @ xj - 2 * xi @ xj
            #     Ei = alpha_y @ inner_i + b - yi
            #     Ej = alpha_y @ inner_j + b - yj
            #
            #     alphas[j] = aj + yj * (Ei - Ej) / eta
            #
            #     alphas[j] = max(L, alphas[j])
            #     alphas[j] = min(H, alphas[j])
            #
            #     if abs(alphas[j] - aj) <= tol:
            #         alphas[j] = aj
            #     else:
            #         alphaPairChanged += 1
            #
            #     alphas[i] = (ai * yi + aj * yj - alphas[j] * yj) / yi
            #
            # if alphaPairChanged == 0:
            #     iter += 1

                # self-version
                # take the derivative of ai
                if yi != yj:
                    L = max(0, ai - aj)
                    H = min(C, ai - aj + C)
                else:
                    L = max(0, ai + aj - C)
                    H = min(C, ai + aj)
                if L==H:
                    continue


                eta = xi @ xi + xj @ xj - 2 * xi @ xj
                Ei = alpha_y @ inner_i + b - yi
                Ej = alpha_y @ inner_j + b - yj

                alphas[i] = ai + yi * (Ej - Ei) / eta

                alphas[i] = max(L, alphas[i])
                alphas[i] = min(H, alphas[i])

                if abs(alphas[i] - ai)<=tol:
                    alphas[i] = ai
                else:
                    alphaPairChanged += 1

                alphas[j] = (ai * yi + aj * yj - alphas[i] * yi) / yj

            if alphaPairChanged==0:
                iter += 1
        # end for i = 1 : m
    # end while iter < maxiter


    W = (alphas * y) @ X

    # From complementary slackness,
    # alpha_i > 0 -->  yi(W * xi + b) = 1
    # choose the data points which correspond to the "support vectors" and calculate b
    idx1 = np.where((alphas>tol) & (y==1)>0)[0]
    idx2 = np.where((alphas>tol) & (y==-1)>0)[0]

    b = (- W @ X[idx1[0],:] - W @ X[idx2[0],:]) / 2

    return W, b, alphas, idx1, idx2



def loadDataSet(filename):
    datamat = []
    labelmat = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        datamat.append([lineArr[0], lineArr[1]])
        labelmat.append(lineArr[2])
    return np.array(datamat,dtype=np.float32), np.array(labelmat,dtype=np.float32)





X, y = loadDataSet('data/testSet.txt')

W, b, alphas, idx1, idx2 = smo(X, y, 1)

print(b)



xmin = np.min(X[:,0])
xmax = np.max(X[:,0])
x1s = np.linspace(xmin, xmax, 100)

x2s = -1 * W[0] / W[1] * x1s - b / W[1]

print(W,b)

plt.plot(X[:,0], X[:,1], 'ro')
plt.plot(X[idx1,0], X[idx1, 1], 'bo')
plt.plot(X[idx2,0], X[idx2, 1], 'bo')
plt.plot(x1s, x2s)
plt.show()