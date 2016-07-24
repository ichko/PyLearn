from math import exp
import numpy as np

from .trainable_model import TrainableModel


class LinearRegression(TrainableModel):

    def hypothesis(self, X, theta):
        return X.dot(theta)


class LogisticRegression(TrainableModel):

    def __init__(self, *args):
        super().__init__(*args)
        self.boundary_threshold = 0

    def sigmoid(self, x):
        return np.vectorize(lambda x: 1 / (1 + exp(-x)))(x)

    def hypothesis(self, X, theta):
        return self.sigmoid(X.dot(theta))

    def fit(self, *args):
        predict = super(LogisticRegression, self).fit(*args)
        self.unthresholded = predict
        return lambda inp: 1 if predict(inp) > 0 else 0
