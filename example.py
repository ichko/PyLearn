import matplotlib.pyplot as plt
import numpy as np
import example_data
from pylearn import linear_model


# Logistic regression example
X = [[r for r in x] for x in example_data.lr_X]
y = example_data.lr_y
lr = linear_model.LogisticRegression()
lr.max_iterations = 400
lr.learning_rate = 1
params, normalize, reverse = lr.fit(X, y)
print('params =', params)

colormap = np.array(['g', 'b', 'y'])
for i, row in enumerate(example_data.lr_X):
    plt.scatter(row[0], row[1], s=50, c=colormap[y[i]])

x_units = list(np.arange(30, 110, 5))
y_units = [i for z in [[i] * len(x_units) for i in x_units] for i in z]
grid = [[x, y] for x, y in zip(x_units * len(x_units), y_units)]

for x, y in grid:
    pred = lr.predict([x, y])
    plt.scatter(x, y, marker='x', s=30,
                c=colormap[1 if pred > 0 else 0])

a = -params[1] / params[2]
d = -params[0] / params[2]
plt.plot([30, 100], [reverse(a * normalize(30) + d),
                     reverse(a * normalize(100) + d)])

# Linear regression example
X = [[i] for i in [-15.9368, -29.1530,  36.1895, 37.4922,
     -48.0588, -8.9415, 15.3078, -34.7063, 1.3892,
     -44.3838, 7.0135, 22.7627]]
y = [2.1343, 1.1733, 34.3591, 36.8380, 2.8090, 2.1211, 14.7103,
     2.6142, 3.7402, 3.7317, 7.6277, 22.7524]

lr = linear_model.LinearRegression()
lr.max_iterations = 1500
lr.learning_rate = 0.002
params = lr.fit(X, y)
print('params =', params)

new_data = [-50, 40]
plt.plot(new_data, [lr.predict([x]) for x in new_data])
plt.plot(X, y, 'ro')
plt.show()
