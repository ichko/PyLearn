"""Module containing implementation of optimization (minimization of the error)
of model using gradient descent.

"""


import numpy as np

from .preprocess import InputData, InitialParameters, FeatureScaling
from .cost import sum_squares


def _one(*_):
    return 1


class RegressionModel:

    def __init__(self, normalize_descent=True, log_statistics=True):
        """Setting initial values of the parameters of the model."""
        self.max_iterations = 1000
        self.learning_rate = 0.5
        self.regularization_term = 0
        self.train_threshold = 0.01
        self.params = []
        self.log_statistics = log_statistics
        self.normalize_descent = normalize_descent

        self.feature_scale = _one
        self.predict = _one
        self.cost = _one
        self.derivative = _one

        self.gradient_log = []
        self.cost_log = []
        self.params_log = []
        self.initial_params_function = InitialParameters.random

    def fit(self, X, y):
        """Method normalizing the input data and executing gradient descent
        over the cost function of the parameters of the model.

        """
        self.feature_scale, _ = FeatureScaling.get_mean_normalize(X)
        X, y = InputData.normalize(self.feature_scale(X), y)

        self.cost, self.derivative = sum_squares(
            self._hypothesis, self.regularization_term)
        self.params = self.initial_params_function(len(X[0]))
        self._train(X, y)

        def predictor(inp, params=self.params):
            return sum(x * t for x, t in zip(
                [1] + list(self.feature_scale(inp)), params))

        self.predict = predictor

        return predictor

    def _train(self, X, y):
        """Implementation of gradient descent over the input data with respect
        to the derivative of the cost function.
        The cost is defined as the sum of the squares of the difference
        between the results generated with the hypothesis function
        and the actual data.

        """
        last_derivative = self.derivative(self.params, X, y)
        iteration = self.max_iterations
        current_error = sum(map(abs, last_derivative))
        while iteration and current_error > self.train_threshold:
            if self.log_statistics:
                self.initiate_snapshot(current_error, X, y)

            if self.normalize_descent:
                last_derivative = last_derivative / np.linalg.norm(
                    last_derivative)

            self.params = self.params - self.learning_rate * last_derivative
            last_derivative = self.derivative(self.params, X, y)
            current_error = sum(map(abs, last_derivative))
            iteration -= 1

    def initiate_snapshot(self, current_error, X, y):
        """Method collecting statistical data during the execution
        of gradient descent.

        """
        self.gradient_log.append(current_error)
        self.cost_log.append(self.cost(self.params, X, y))
        self.params_log.append(self.params)
