from pylearn import high_order_model, linear_model, visualizer, data_loader
from pylearn.space_transform import full_polynomial_mapper


# Polynomial logistic regression example
X_lo, y_lo = data_loader.parse_data('data/logistic_non_linear.txt')
y_lo = [int(y) for y in y_lo]

model = high_order_model.PolynomialLogisticRegression()
model.learning_rate = 0.1
model.max_iterations = 400

mapper = full_polynomial_mapper(6, 2)
predict = model.fit(X_lo, y_lo, mapper)

visualizer.plot_2d_classifier_stats(model, X_lo, y_lo)


exit()

# Linear logistic regression example
X_lo, y_lo = data_loader.parse_data('data/logistic_linear.txt')
y_lo = [int(y) for y in y_lo]

model = linear_model.LogisticRegression()
predict = model.fit(X_lo, y_lo)

visualizer.plot_2d_classifier_stats(model, X_lo, y_lo)
