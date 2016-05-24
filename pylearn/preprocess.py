import numpy as np


class InitialParameters:

    @staticmethod
    def ones(size):
        return np.array([1] * size)


class FeatureScaling:

    @staticmethod
    def normalize(x):
        return x / np.amax(x)

    @staticmethod
    def get_mean_normalize(x):
        average = np.average(x)
        print(average)
        max = np.amax(x)

        def normalize(x):
            return (x - average) / max

        def reverse_normalize(x):
            return x * max + average

        return normalize, reverse_normalize


class InputData:

    @staticmethod
    def add_constant_term(X, term=1):
        return [[term] + list(row) for row in X]

    @classmethod
    def normalize(cls, X, y):
        return np.array(cls.add_constant_term(X)), np.array(y)
