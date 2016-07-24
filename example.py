import matplotlib.pyplot as plt
import numpy as np
import example_data
from pylearn import high_order_model, linear_model

'''
# Logistic regression example
lr = linear_model.LogisticRegression()
predict = lr.fit(example_data.X_lo, example_data.y_lo)

colormap = np.array(['g', 'b', 'y'])
x_units = list(np.arange(30, 110, 5))
y_units = [i for z in [[i] * len(x_units) for i in x_units] for i in z]
grid = [[x, y] for x, y in zip(x_units * len(x_units), y_units)]

for x, y in grid:
    plt.scatter(x, y, marker='x', s=30, c=colormap[predict([x, y])])

for i, row in enumerate(example_data.X_lo):
    plt.scatter(row[0], row[1], s=50, c=colormap[example_data.y_lo[i]])

plt.show()

# Linear regression example
lr = linear_model.LinearRegression()
predict = lr.fit(example_data.X_re, example_data.y_re)

test_data = [-100, 100]
test_results = [predict([x]) for x in test_data]

plt.plot(test_data, test_results)
plt.plot(example_data.X_re, example_data.y_re, 'ro')
plt.show()
'''

example_data.X_re = [[-x[0] - 100] for x in example_data.X_re]
example_data.y_re = [y - 100 for y in example_data.y_re]

# Polynomial regression example
mapper = [lambda x: x ** 4]
lr = high_order_model.PolynomialRegression()
predict = lr.fit(example_data.X_re, example_data.y_re, mapper)

test_data = list(np.arange(-150, -50, 5))
test_results = [predict([x]) for x in test_data]

plt.plot(test_data, test_results)
plt.plot(example_data.X_re, example_data.y_re, 'ro')
plt.show()
