"""Module containing implementation of linear models.

"""


from math import exp

import numpy as np

from .regression_model import RegressionModel
from .math import sigmoid


class LinearRegression(RegressionModel):
    """Class implementing simple weighted _hypothesis and linearly
    fitting the data.

    """

    def _hypothesis(self, X, params):
        """Method applying sum of products between X (the input data) and
        the parameters of the model.

        """
        return X.dot(params)


class LogisticRegression(RegressionModel):
    """Class implementing simple weighted _hypothesis and applying the sigmoid
    function over the result resulting in linear binary classification.

    """

    def __init__(self, *args):
        super().__init__(*args)

        # Setting the decision boundary threshold
        self.boundary_threshold = 0.5

    def _hypothesis(self, X, params):
        """Method applying sum of products between X (the input data) and
        the parameters of the model and then applying the sigmoid function.

        """
        return sigmoid(X.dot(params))

    def fit(self, *args):
        """Method running the fitter and then setting the
        threshold predict function.

        """
        predict = super(LogisticRegression, self).fit(*args)
        self.predict = predict
        return lambda inp, params=self.params: 1 if predict(
            inp, params) > self.boundary_threshold else 0
