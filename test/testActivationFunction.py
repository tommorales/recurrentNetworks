__author__ = 'tmorales'

import numpy as np

from src.activationFuncion import softmax

w =3
print softmax(w)

w = np.array([0.1, 0.2])
print softmax(w)

w = np.array([-0.1,0.2])
print w

w = np.array([0.9,-10])
print w