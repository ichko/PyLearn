import numpy as np
import random as rnd


class InitialParameters:

    @staticmethod
    def ones(size):
        return np.array([1] * size)

    @staticmethod
    def random(size, min=0, max=1):
        return np.array([rnd.random() * (max - min) + min for _ in range(size)])


class FeatureScaling:

    @staticmethod
    def normalize(x):
        return x / np.amax(x)

    @staticmethod
    def get_mean_normalize(X):
        X = np.array(X).transpose()
        average = np.array([np.average(row) for row in X])
        max = np.array([np.amax(row) for row in X])

        def normalize(x):
            return (np.array(x) - average) / max

        def reverse_normalize(x):
            return np.array(x) * max + average

        return normalize, reverse_normalize


class InputData:

    @staticmethod
    def add_constant_term(X, term=1):
        return np.array([[term] + list(row) for row in X])

    @classmethod
    def normalize(cls, X, y):
        return np.array(cls.add_constant_term(X)), np.array(y)
