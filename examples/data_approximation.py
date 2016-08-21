from pylearn import high_order_model, visualizer, data_loader
from pylearn.space_transform import full_polynomial_mapper


# Polynomial regression example
X, y = data_loader.parse_data('data/approx_data.txt')

model = high_order_model.PolynomialRegression()
model.learning_rate = 0.12
model.max_iterations = 500

mapper = full_polynomial_mapper(7, 2)
predict = model.fit(X, y, mapper)

visualizer.plot_1d_approximator_stats(model, X, y)
print('Done')
