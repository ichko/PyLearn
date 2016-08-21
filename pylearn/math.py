import numpy as np


def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def sigmoid(x):
    """Mapping z to ~1 if z >> 0 else ~0."""
    return 1 / (1 + np.exp(-x))
