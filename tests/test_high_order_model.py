from .base_test import BaseTest

from pylearn import high_order_model, preprocess, space_transform


class TestHighOrderModel(BaseTest):

    def setUp(self):
        ones = preprocess.InitialParameters.ones
        self.polynomial_model = high_order_model.PolynomialRegression()
        self.polynomial_model.initial_params_function = ones

        self.logistic_model = high_order_model.PolynomialLogisticRegression()
        self.logistic_model.initial_params_function = ones

        self.mapper_3d = space_transform.full_polynomial_mapper(3, 6)
        self.mapper_4d = space_transform.full_polynomial_mapper(4, 6)

    def test_complex_polynomial_logistic_fit(self):
        predictor = self.logistic_model.fit(self.fake_X_data_complex,
                                            self.fake_y_binary_data,
                                            self.mapper_4d)

        actual_y_1 = predictor([1, -7, 5, 1])
        actual_y_2 = predictor([2, 7, -5, 3])
        actual_y_3 = predictor([1, 2, -3, -4])

        self.assertEqual(0, actual_y_1)
        self.assertEqual(1, actual_y_2)
        self.assertEqual(1, actual_y_3)

    def test_simple_polynomial_fit(self):
        predictor = self.polynomial_model.fit(self.fake_X_data,
                                              self.fake_y_data,
                                              self.mapper_3d)

        actual_y_1 = predictor([1, -7, 5])
        actual_y_2 = predictor([2, 7, -5])
        actual_y_3 = predictor([3, -2, 2])

        self.assertAlmostEqual(-0.846, actual_y_1, 2)
        self.assertAlmostEqual(-3.807, actual_y_2, 2)
        self.assertAlmostEqual(-1.023, actual_y_3, 2)

    def test_complex_polynomial_fit(self):
        predictor = self.polynomial_model.fit(self.fake_X_data_complex,
                                              self.fake_y_data_complex,
                                              self.mapper_4d)

        actual_y_1 = predictor([1, -7, 5, 6])
        actual_y_2 = predictor([2, 7, -5, 3])
        actual_y_3 = predictor([3, -2, 2, 6])

        self.assertAlmostEqual(-54.345, actual_y_1, 2)
        self.assertAlmostEqual(107.364, actual_y_2, 2)
        self.assertAlmostEqual(-17.805, actual_y_3, 2)

if __name__ == '__main__':
    unittest.main()
