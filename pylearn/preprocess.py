import numpy as np


class InitialParameters:

    @staticmethod
    def ones(size):
        return np.array([1] * size)


class InputData:

    @staticmethod
    def add_constant_term(X, term=1):
        return [[term] + row for row in X]

    @classmethod
    def normalize(cls, X, y):
        return np.array(cls.add_constant_term(X)), np.array(y)
