from .base_test import BaseTest

from pylearn import math


class TestMath(BaseTest):

    def test_sigmoid(self):
        actual_1 = math.sigmoid(100)
        actual_2 = math.sigmoid(-100)
        actual_3 = math.sigmoid(0)

        self.assertAlmostEqual(1, actual_1)
        self.assertAlmostEqual(0, actual_2)
        self.assertAlmostEqual(0.5, actual_3)

    def test_sigmoid_prime(self):
        actual_1 = math.sigmoid_prime(100)
        actual_2 = math.sigmoid_prime(-100)
        actual_3 = math.sigmoid_prime(0)

        self.assertAlmostEqual(0, actual_1)
        self.assertAlmostEqual(0, actual_2)
        self.assertAlmostEqual(0.25, actual_3)


if __name__ == '__main__':
    unittest.main()
