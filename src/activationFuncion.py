__author__ = 'tmorales'


import numpy as np

"""
Softmax function
================

Softmax or normalized exponential, is a generalization of the logistic function that "squasted" a k-dimension
vector Z or arbitrary real value to a k-dimensional vector $\sigma(z)$ of the real values in the range (0,1) that
add up to 1.


$\sigma(Z)_{j}=\frac{e^{z_{j}}}{\sum_{i=1}^{k}e^{zk}}$ for ${j=1,2,....,k}$


In Neuronal Network simulation is often implemented at the final layer of a network used to classification.
"""

def softmax(w, t=1.0):
    """
    Softmax activation function
    :param w: numpy array
    :param t:
    :return: values of the softmax function.
    """
    # Note: There should be a exception for knowing if the variables is a numpy array

    e = np.exp(w/t)
    dist = e / np.sum(e)
    return dist