import unittest
from .unittest_ext import lists_close
from pylearn import linear_model


class LinearModel(unittest.TestCase):

    def test_simple_linear_regression(self):
        data, result = [[0], [2]], [2, 4]
        lr = linear_model.LinearRegression()
        lr.learning_rate = 0.01
        params = lr.fit(data, result)
        self.assertTrue(lists_close(params, [2, 1], 0.1))


if __name__ == '__main__':
    unittest.main()
