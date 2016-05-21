import numpy as np


def rss(self, X, y, theta, reg_term=0):
    training_set_size = len(y)
    hypothesis = X.dot(theta)
    reg_sum = reg_term * sum(theta[1:] ** 2)
    return (sum((hypothesis - y) ** 2) + reg_sum) / (2 * training_set_size)


def rss_gradient(self, X, y, theta, reg_term=0):
    training_set_size = len(y)
    hypothesis = X.dot(theta)
    gradient = np.array(X.transpose().dot(hypothesis - y)) / training_set_size
    gradient[1:] = gradient[1:] + reg_term * theta[1:] / training_set_size
    return gradient
