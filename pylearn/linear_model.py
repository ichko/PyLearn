from math import exp
import numpy as np

from .trainable_model import TrainableModel


class LinearRegression(TrainableModel):

    def hypothesis(self, X, params):
        return X.dot(params)


class LogisticRegression(TrainableModel):

    def __init__(self, *args):
        super().__init__(*args)
        self.boundary_threshold = 0

    def sigmoid(self, x):
        return np.vectorize(lambda x: 1 / (1 + exp(-x)))(x)

    def hypothesis(self, X, params):
        return self.sigmoid(X.dot(params))

    def fit(self, *args):
        predict = super(LogisticRegression, self).fit(*args)
        self.predict = predictor
        return lambda inp, params=self.params: 1 if predict(
            inp, params) > self.boundary_threshold else 0
