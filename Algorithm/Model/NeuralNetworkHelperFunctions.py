import numpy as np

def sigmoid_forward(X):
    """
    Compute the element-wise sigmoid function sigmoid(x) = 1 / (1 + exp(-x)) with numerical stability
    """

    # when x>=0, exp(x) will be small, perform computation with exp(-x)
    mask1 = (X>=0)
    X1 = mask1 * X
    Z1 = np.exp(-X1)
    S1 = mask1 * (1 / (1 + Z1))

    # when x<=0, exp(-x) will be small, perform computation with exp(x)
    # 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
    mask2 = (X<0)
    X2 = mask2 * X
    Z2 = np.exp(X2)
    S2 = mask2 * (Z2 / (1 + Z2))

    return S1 + S2

def sigmoid_backward(X):
    return X * (1. - X)

""" 
Following codes are from Assigment of CS231n
http://cs231n.github.io/assignments2019/assignment2/
"""

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    num_train = x.shape[0]
    X = np.reshape(x, (num_train,-1))
    out = np.matmul(X,w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    original_shape = x.shape
    num_train = original_shape[0]
    X = np.reshape(x, (num_train, -1))

    db = np.sum(dout,axis=0)
    dw = np.matmul(
        np.transpose(X),dout
    )
    dx = np.matmul(
        dout,np.transpose(w)
    ).reshape(original_shape)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = np.maximum(x,0)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    where = np.where(x<=0.)
    dx = dout
    dx[where] = 0.

    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx



