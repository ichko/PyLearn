import numpy as np
from .initial_parameters import ones
from .cost import rss_gradient


class LinearRegression:

    def __init__(self):
        self.learning_rate = 0.01
        self.regularization_param = 2

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        training_set_size = len(y)
        theta = ones(training_set_size)
        gradient = rss_gradient(X, y, theta, self.regularization_param)

        return gradient
