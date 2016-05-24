from math import exp
import numpy as np
from .preprocess import InputData, InitialParameters, FeatureScaling
from .cost import rss
from .trainable_model import TrainableModel


class LinearRegression(TrainableModel):

    def hypothesis(self, X, theta):
        return X.dot(theta)

    def fit(self, X, y):
        X, y = InputData.normalize(X, y)
        cost, derivative = rss(X, y, self.hypothesis, self.regularization_term)
        theta = InitialParameters.ones(len(X[0]))
        theta = self.train(derivative, theta)

        def predictor(inp):
            return sum(x * t for x, t in zip([1] + inp, theta))

        self.predict = predictor
        return theta


class LogisticRegression(TrainableModel):

    def sigmoid(self):
        return np.vectorize(lambda x: 1 / (1 + exp(-x)))

    def hypothesis(self, X, theta):
        return self.sigmoid()(X.dot(theta))

    def fit(self, X, y):
        normalizer, reverse = FeatureScaling.get_mean_normalize(X)
        X, y = InputData.normalize(normalizer(X), y)

        cost, derivative = rss(X, y, self.hypothesis, self.regularization_term)
        theta = InitialParameters.ones(len(X[0]))
        theta = self.train(derivative, theta)

        def predictor(inp):
            result = sum(x * t for x, t
                         in zip([1] + list(normalizer(inp)), theta))
            return result

        self.predict = predictor
        return theta, normalizer, reverse
