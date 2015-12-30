__author__ = 'tmorales'

import numpy as np
import matplotlib.pyplot as plt


"""
Distribution function of U, Y and W weights.

"""

word_dim=5
hidden_dim=100

def boundary(x):
    return np.sqrt(1./x)

distributionFunction = np.random.uniform(-boundary(word_dim), boundary(word_dim),
                                         (hidden_dim,word_dim))

print distributionFunction

print "Shape of weight matix = {0}".format(distributionFunction.shape)
print

print "limits of the distribution function"
print boundary(word_dim)

print "plotting ..."
#
# The data are noot well representation ?
#
plt.hist(distributionFunction)
plt.savefig('uniform-distribution.png')