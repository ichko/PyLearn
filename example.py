import matplotlib.pyplot as plt
from pylearn import linear_model


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
