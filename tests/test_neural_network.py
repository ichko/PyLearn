from .base_test import BaseTest

from pylearn.neural_network import NeuralNetwork
from pylearn.preprocess import InitialParameters


class TestNeuralNetwork(BaseTest):

    def setUp(self):
        self.net_3d = NeuralNetwork([3, 5, 5, 3], 5,
                                    InitialParameters.ones_matrix)
        self.net_4d = NeuralNetwork([4, 10, 8, 4], 11,
                                    InitialParameters.ones_matrix)
        self.net_3d.max_iteration = 30
        self.net_3d.batch_size = 1
        self.net_3d.batch_preprocess = lambda x: x
        self.net_4d.max_iteration = 30
        self.net_4d.batch_size = 1
        self.net_4d.batch_preprocess = lambda x: x

        self.fake_3d_X_data = [
            [1, 2, 3],
            [1, 2, 3],
            [-1, -2, -3],
            [-1, -2, -3],
        ]

        self.fake_3d_y_data = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]

        self.fake_4d_X_data = [
            [1, 2, 3, 2],
            [1, 2, 3, 2],
            [1, 2, 3, 2],
            [1, 2, 3, 2],
            [-1, -2, -3, -2],
            [-1, -2, -3, -2],
            [-1, -2, -3, -2],
            [-1, -2, -3, -2],
        ]

        self.fake_4d_y_data = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ]

    def test_network_num_hits(self):
        predictor = self.net_4d.fit(self.fake_4d_X_data,
                                    self.fake_4d_y_data)

        actual = self.net_4d.test_network([
            [1, 2, 3, 4],
            [-1, -2, -3, -4]
        ], [
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        self.assertAlmostEqual(2, actual)

    def test_network_cost(self):
        predictor = self.net_4d.fit(self.fake_4d_X_data,
                                    self.fake_4d_y_data)

        actual_cost_1 = self.net_4d.batch_cost([[1, 2, 3, 4]], [[1, 0, 0, 0]])
        actual_cost_2 = self.net_4d.batch_cost([[-1, -2, -3, -4]],
                                               [[0, 0, 1, 0]])

        self.assertAlmostEqual(0, actual_cost_1, 2)
        self.assertAlmostEqual(0, actual_cost_2, 2)

    def test_complex_network_fit(self):
        predictor = self.net_4d.fit(self.fake_4d_X_data,
                                    self.fake_4d_y_data)

        actual_y_1 = predictor([1, 2, 3, 4])
        actual_y_2 = predictor([-1, -2, -3, -4])

        self.assertListEqual([1, 0, 0, 0], list(actual_y_1))
        self.assertListEqual([0, 0, 1, 0], list(actual_y_2))

    def test_simple_network_fit(self):
        predictor = self.net_3d.fit(self.fake_3d_X_data,
                                    self.fake_3d_y_data)

        actual_y_1 = predictor([1, 2, 3])
        actual_y_2 = predictor([-1, -2, -3])

        self.assertListEqual([0, 1, 0], list(actual_y_1))
        self.assertListEqual([0, 0, 1], list(actual_y_2))


if __name__ == '__main__':
    unittest.main()
