"""For preprocessing input data and parameters (feature scaling).

"""


import numpy as np
import random as rnd


class InitialParameters:
    """Class containing static methods for setting the initial state
    of the parameters of model.

    """

    @staticmethod
    def ones(size):
        """Returns numpy vector with ones."""
        return np.array([1] * size)

    @staticmethod
    def random(size, min=0, max=1):
        """Returns numpy vector random values between min and max."""
        return np.array([rnd.random() * (max - min) + min
                         for _ in range(size)])


class FeatureScaling:

    @staticmethod
    def normalize(x):
        """Dividing the values of the vector by the maximal value."""
        return x / np.amax(x)

    @staticmethod
    def get_mean_normalize(X):
        """Method returning functions implementing mean normalization
        (and the reverse of this normalization) to dataset of values.

        """
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
        """Adding constant term in the first column of matrix."""
        return np.array([[term] + list(row) for row in X])

    @classmethod
    def normalize(cls, X, y):
        """Adding constant term in the first column of matrix and wrapping
        the input data sets in numpy arrays.

        """
        return np.array(cls.add_constant_term(X)), np.array(y)

    @staticmethod
    def image_matrix_normalizer(images, labels, labels_list, pixel_scalar=50):
        """Method dividing the pixel values of the image by a constant term and
        mapping the output values to binary vectors.

        """
        return ([[x / pixel_scalar for x in x_row] for x_row in images],
                [[1 if y == i else 0 for i in labels_list] for y in labels])
