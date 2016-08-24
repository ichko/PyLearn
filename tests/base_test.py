import numpy as np
import unittest


class BaseTest(unittest.TestCase):

    fake_X_data = np.array([
        [1, 2, 3],
        [4, 5, 4],
        [7, 5, 6],
    ])

    fake_y_data = np.array([1, 2, 3])

    fake_X_data_complex = np.array([
        [1, 2, -3, -4],
        [4, -5, 4, -6],
        [-7, 5, 6, -2],
        [-4, 2, 3, -8],
    ])

    fake_y_data_complex = np.array([1, 1, -6, 3])

    fake_y_binary_data = np.array([1, 0, 0, 1])

    def assertListsAlmostEqual(self, list_a, list_b, eps=0.01):
        result = all(abs(x - y) < eps for x, y in zip(list_a, list_b))
        if not result:
            print('list_a =', list_a, '\nlist_b =', list_b)

        self.assertTrue(result)
