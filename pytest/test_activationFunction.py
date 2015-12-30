__author__ = 'tmorales'

import numpy as np

from src.activationFuncion import softmax


def test_softmax():
    w = np.array([0.1, 0.2])
    assert softmax(w) == [0.47502081, 0.52497919]
