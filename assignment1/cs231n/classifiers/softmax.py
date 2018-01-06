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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    max_score = scores.max()
    scores -= max_score
    loss += -scores[y[i]] + np.log((np.exp(scores)).sum())
    for k in range(num_classes):
      dW[:, k] += X[i].transpose() * (np.exp(scores[k]) / (np.exp(scores)).sum())
    dW[:, y[i]] += -X[i].transpose()
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  pass
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  score_mat = X.dot(W)
  score_mat -= score_mat.max(axis=1).reshape(num_train, 1)
  loss_arr = -score_mat[range(num_train), y] + \
             np.log((np.exp(score_mat)).sum(axis=1))
  loss = loss_arr.sum() / num_train + reg * np.sum(W * W)

  temp_mat1 = np.exp(score_mat) / (np.exp(score_mat)).sum(axis=1).reshape(num_train, 1)
  temp_mat2 = np.array([y == i for i in range(num_classes)]).transpose()
  dW = np.dot(X.transpose(), temp_mat1 - temp_mat2)
  dW = dW / num_train + reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

