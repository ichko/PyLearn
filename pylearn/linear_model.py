from math import exp
import numpy as np
from .preprocess import InputData, InitialParameters
from .cost import rss
from .trainable_model import TrainableModel


class LinearRegression(TrainableModel):

    def hypothesis(self, X, theta):
        return X.dot(theta)

    def fit(self, X, y):
        X, y = InputData.normalize(X, y)
        cost, derivative = rss(X, y, self.hypothesis, self.regularization_term)
        theta = InitialParameters.ones(len(X[0]))
        return self.train(derivative, theta)


class LogisticRegression(TrainableModel):

    def sigmoid(x):
        return np.vectorize(lambda x: 1 / (1 + exp(-x)))

    def hypothesis(self, X, theta):
        return self.sigmoid(X.dot(theta))

    def fit(self, X, y):
        X, y = InputData.normalize(X, y)
        cost, derivative = rss(X, y, self.hypothesis, self.regularization_term)
        theta = InitialParameters.ones(len(X[0]))
        return self.train(derivative, theta)
