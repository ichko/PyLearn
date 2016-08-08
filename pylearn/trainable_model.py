import numpy as np

from .preprocess import InputData, InitialParameters, FeatureScaling
from .cost import rss


def one(*_):
    return 1


class TrainableModel:

    def __init__(self, normalize_descent=True, log_statiscics=True):
        self.max_iterations = 1000
        self.learning_rate = 0.5
        self.regularization_term = 0
        self.train_threshold = 0.01
        self.params = []
        self.log_statiscics = log_statiscics
        self.normalize_descent = normalize_descent

        self.feature_scale = one
        self.predict = one
        self.cost = one
        self.derivative = one

        self.gradient_log = []
        self.cost_log = []
        self.params_log = []

    def fit(self, X, y):
        self.feature_scale, _ = FeatureScaling.get_mean_normalize(X)
        X, y = InputData.normalize(self.feature_scale(X), y)

        self.cost, self.derivative = rss(
            self.hypothesis, self.regularization_term)
        self.params = InitialParameters.random(len(X[0]))
        self.train(X, y)

        def predictor(inp, params=self.params):
            return sum(x * t for x, t in zip(
                [1] + list(self.feature_scale(inp)), params))

        self.predict = predictor

        return predictor

    # Gradient descent
    def train(self, X, y):
        last_derivative = self.derivative(self.params, X, y)
        iteration = self.max_iterations
        current_error = sum(map(abs, last_derivative))
        while iteration and current_error > self.train_threshold:
            if self.log_statiscics:
                self.initiate_snapshot(current_error, X, y)

            if self.normalize_descent:
                last_derivative = last_derivative / np.linalg.norm(
                    last_derivative)

            self.params = self.params - self.learning_rate * last_derivative
            last_derivative = self.derivative(self.params, X, y)
            current_error = sum(map(abs, last_derivative))
            iteration -= 1

    def initiate_snapshot(self, current_error, X, y):
        self.gradient_log.append(current_error)
        self.cost_log.append(self.cost(self.params, X, y))
        self.params_log.append(self.params)
