import numpy as np


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
