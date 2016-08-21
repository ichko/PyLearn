from math import exp
import numpy as np

from .regression_model import RegressionModel
from .math import sigmoid


class LinearRegression(RegressionModel):

    def _hypothesis(self, X, params):
        return X.dot(params)


class LogisticRegression(RegressionModel):

    def __init__(self, *args):
        super().__init__(*args)
        self.boundary_threshold = 0

    def _hypothesis(self, X, params):
        return sigmoid(X.dot(params))

    def fit(self, *args):
        predict = super(LogisticRegression, self).fit(*args)
        self.predict = predict
        return lambda inp, params=self.params: 1 if predict(
            inp, params) > self.boundary_threshold else 0
