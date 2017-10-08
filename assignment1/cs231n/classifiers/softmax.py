import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_of_data = X.shape[0]
  num_of_class = W.shape[1]
  for i in range (num_of_data):
    # Compute the score vector
    s = X[i].dot(W)
    s_min = np.min(s)
    s -= s_min
    # Compute soft-max vector
    s = np.exp(s)
    s_total = np.sum(s)
    p = s/s_total
    # Compute Loss 
    L = -np.log(p[y[i]])
    loss += L
    # Compute gradient dW
    for j in range (num_of_class):
      target = 1 if (j == y[i]) else 0
      dW[:,j] += (p[j] - target)*X[i]

  loss /= num_of_data
  dW /= num_of_data
  #regularization
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_of_data = X.shape[0]
  # Compute Score Matrix S & Normalize
  S = X.dot(W)
  S_min = S.min(axis=1, keepdims=True)
  S -= S_min
  # Compute Soft-Max Matrix P
  S = np.exp(S)
  S_total = S.sum(axis=1, keepdims=True)
  P = S/S_total
  # Compute Loss vector L
  Prediction = np.choose(y, P.T)
  L = -np.log(Prediction)
  loss = np.sum(L)
  # Compute Gradient dW
  Target = np.zeros_like(P)
  Target[np.arange(num_of_data),y] = 1
  dS = P - Target
  dW += X.T.dot(dS)

  loss /= num_of_data
  dW /= num_of_data
  #regularization
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

