from .base_test import BaseTest

from pylearn import linear_model, preprocess


class TestLinearModel(BaseTest):

    def setUp(self):
        ones = preprocess.InitialParameters.ones
        self.linear_model = linear_model.LinearRegression()
        self.linear_model.initial_params_function = ones

        self.logistic_model = linear_model.LogisticRegression()
        self.logistic_model.initial_params_function = ones

    def test_complex_logistic_fit(self):
        predictor = self.logistic_model.fit(self.fake_X_data_complex,
                                            self.fake_y_binary_data)

        actual_y_1 = predictor([1, -7, 5, 1])
        actual_y_2 = predictor([2, 7, -5, 3])
        actual_y_3 = predictor([1, 2, -3, -4])

        self.assertEqual(0, actual_y_1)
        self.assertEqual(0, actual_y_2)
        self.assertEqual(1, actual_y_3)

    def test_simple_linear_fit(self):
        predictor = self.linear_model.fit(self.fake_X_data, self.fake_y_data)

        actual_y_1 = predictor([1, -7, 5])
        actual_y_2 = predictor([2, 7, -5])
        actual_y_3 = predictor([3, -2, 2])

        self.assertAlmostEqual(0.472, actual_y_1, 2)
        self.assertAlmostEqual(-0.57, actual_y_2, 2)
        self.assertAlmostEqual(0.527, actual_y_3, 2)

    def test_complex_linear_fit(self):
        predictor = self.linear_model.fit(self.fake_X_data_complex,
                                          self.fake_y_data_complex)

        actual_y_1 = predictor([1, -7, 5, 6])
        actual_y_2 = predictor([2, 7, -5, 3])
        actual_y_3 = predictor([3, -2, 2, 6])

        self.assertAlmostEqual(-17.205, actual_y_1, 2)
        self.assertAlmostEqual(-5.626, actual_y_2, 2)
        self.assertAlmostEqual(-13.946, actual_y_3, 2)

if __name__ == '__main__':
    unittest.main()
