from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073,10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means  
      that X[i] has label c, where 0 <= c < C. (500,)
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    # numeric instability- softmax is prone to 2 issues : 
    # overflow(very large numbers can be approximated as infinity)
    # undeflow(very small numbers can be approximated(rounded) as zero)
    stable_scores = scores-np.max(scores, axis=1,keepdims=True)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
      softmax = np.exp(stable_scores[i])/np.sum(np.exp(stable_scores[i]))
      loss += -np.log(softmax[y[i]])
      for j in range(num_classes):
        dW[:, j] += X[i] * softmax[j]
      dW[:,y[i]] -= X[i]

    loss /= num_train
    loss += reg*np.sum(X*X)

    dW /= num_train
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    stable_scores = scores-np.max(scores, axis=1,keepdims=True)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    softmax = np.exp(stable_scores)/np.exp(stable_scores).sum(axis=1, keepdims=True)
    loss = np.sum(-np.log(softmax[np.arange(num_train),y]))

    softmax[np.arange(num_train),y] -= 1
    dW = X.T.dot(softmax)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg*2*W 
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
