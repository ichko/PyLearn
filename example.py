import matplotlib.pyplot as plt
import numpy as np

import example_data
from pylearn import high_order_model


# Logistic regression example
mapper = [lambda x, _: x[0] * x[1], lambda x, _: x[1]]
lr = high_order_model.PolynomialLogisticRegression()
predict = lr.fit(example_data.X_lo, example_data.y_lo, mapper)

x = y = np.arange(30, 110, 5)
X, Y = np.meshgrid(x, y)
Z = [list(zip(rowX, rowY)) for rowX, rowY in zip(X, Y)]
Z = [[lr.unthresholded(list(data)) for data in row] for row in Z]

levels = [-3, -1, -0.5, 0, 0.5, 1, 3]
plt.contourf(X, Y, Z, cmap=plt.cm.seismic, levels=levels)

colormap = np.array(['cyan', 'red'])
for i, row in enumerate(example_data.X_lo):
    plt.scatter(row[0], row[1], s=50, c=colormap[example_data.y_lo[i]])

plt.show()


'''
# Linear regression example
lr = linear_model.LinearRegression()
predict = lr.fit(example_data.X_re, example_data.y_re)

test_data = [-100, 100]
test_results = [predict([x]) for x in test_data]

plt.plot(test_data, test_results)
plt.plot(example_data.X_re, example_data.y_re, 'ro')
plt.show()


example_data.X_re = [[-x[0] - 100] for x in example_data.X_re]
example_data.y_re = [y - 100 for y in example_data.y_re]

# Polynomial regression example
lr = high_order_model.PolynomialRegression()
predict = lr.fit(example_data.X_re, example_data.y_re, [lambda x, _: x ** 4])

test_data = list(np.arange(-150, -50, 5))
test_results = [predict([x]) for x in test_data]

plt.plot(test_data, test_results)
plt.plot(example_data.X_re, example_data.y_re, 'ro')
plt.show()
'''
