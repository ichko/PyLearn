from .unittest_extended import TestCase

import numpy as np

from pylearn import cost


def fake_cost_hypothesis(X, params):
    return X.dot(params)

fake_X_data = np.array([
    [1, 2, 3],
    [4, 5, 4],
    [7, 5, 6],
])

fake_y_data = np.array([1, 2, 3])


class TestCost(TestCase):

    def test_sum_squares_reg_term(self):
        error, derivative = cost.sum_squares(fake_cost_hypothesis, 2)

        params = np.array([8, 3, 4])

        error_actual = error(params, fake_X_data, fake_y_data)
        derivative_actual = derivative(params, fake_X_data, fake_y_data)

        self.assertAlmostEqual(2143.33, error_actual, 2)
        self.assertListsAlmostEqual([304.33333333, 273.66666667, 293],
                                    derivative_actual)

    def test_sum_squares(self):
        error, derivative = cost.sum_squares(fake_cost_hypothesis, 0)

        params = np.array([8, 3, 4])

        error_actual = error(params, fake_X_data, fake_y_data)
        derivative_actual = derivative(params, fake_X_data, fake_y_data)

        self.assertEqual(2135, error_actual)
        self.assertListsAlmostEqual([304.33333333, 271.66666667, 290.33333333],
                                    derivative_actual)

    def test_sum_squares_derivative(self):
        derivative = cost.sum_squares_derivative(fake_cost_hypothesis, 0)

        params_1 = np.array([1, 1, 1])
        params_2 = np.array([7, 2, 4])
        params_3 = np.array([4, 8, 8])

        actual_1 = derivative(params_1, fake_X_data, fake_y_data)
        actual_2 = derivative(params_2, fake_X_data, fake_y_data)
        actual_3 = derivative(params_3, fake_X_data, fake_y_data)

        self.assertListsAlmostEqual([51.33333333, 46.66666667, 49.66666667],
                                    actual_1)
        self.assertListsAlmostEqual([263.33333333, 234.66666667, 251.33333333],
                                    actual_2)
        self.assertListsAlmostEqual([392.66666667, 360.33333333, 383.66666667],
                                    actual_3)

    def test_sum_squares_cost(self):
        error = cost.sum_squares_cost(fake_cost_hypothesis, 0)

        params_1 = np.array([1, 2, 2])
        params_2 = np.array([7, 2, 4])
        params_3 = np.array([4, 8, 8])

        actual_1 = error(params_1, fake_X_data, fake_y_data)
        actual_2 = error(params_2, fake_X_data, fake_y_data)
        actual_3 = error(params_3, fake_X_data, fake_y_data)

        self.assertEqual(196, actual_1)
        self.assertEqual(1598, actual_2)
        self.assertEqual(3669, actual_3)


if __name__ == '__main__':
    unittest.main()
