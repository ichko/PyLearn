import numpy as np

from .base_test import BaseTest

from pylearn.preprocess import InitialParameters, FeatureScaling, InputData


class TestPreprocessing(BaseTest):

    def test_input_data(self):
        actual_1 = InputData.add_constant_term([[2, 3], [4, 5]])
        actual_2 = InputData.image_matrix_normalizer([[2, 3], [4, 5]], [1, 0],
                                                     [0, 1], 2)

        self.assertListEqual([[1, 2, 3], [1, 4, 5]],
                             list(list(a) for a in actual_1))

        self.assertListEqual([[1, 3 / 2], [2, 5 / 2]],
                             list(list(a) for a in actual_2[0]))
        self.assertListEqual([[0, 1], [1, 0]], list(actual_2[1]))

    def test_initial_parameters(self):
        actual_1 = InitialParameters.ones_matrix(2, 3)
        actual_2 = InitialParameters.ones(4)

        self.assertListEqual([[1, 1, 1], [1, 1, 1]],
                             list(list(a) for a in actual_1))
        self.assertListEqual([1, 1, 1, 1], list(actual_2))

    def test_feature_scaling(self):
        actual_1 = FeatureScaling.normalize([2, 3, 4])
        normal, reverse_normal = FeatureScaling.get_mean_normalize([
            [1, 2, 3],
            [2, 4, 7],
            [8, 1, 2],
        ])

        actual_2 = normal([1, 2, 3])
        actual_3 = reverse_normal(actual_2)

        self.assertListEqual([0.5, 0.75, 1], list(actual_1))
        self.assertListsAlmostEqual([-0.333, -0.083, -0.142], list(actual_2))
        self.assertListEqual([1, 2, 3], list(actual_3))


if __name__ == '__main__':
    unittest.main()
