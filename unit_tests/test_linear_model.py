import unittest
from pylearn import linear_model


class SirTests(unittest.TestCase):

    def test_linearRegression(self):
        data, result = ((0), (2)), (2, 4)
        lr = linear_model.LinearRegression()
        params = lr.fit(data, result)

        self.assertEqual(len(params), 2)
        self.assertListEqual(params, [2, 1])


if __name__ == '__main__':
    unittest.main()
