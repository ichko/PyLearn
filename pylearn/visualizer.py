"""Module using matplotlib to provide easy interface for visualising
data, model performance, hypothesis and decision boundaries.

"""


import matplotlib.pyplot as plt
import numpy as np


def show_plots():
    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.show()


def new_window():
    """Start new figure."""
    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.figure()


def quit_figure(event):
    """Close window (figure)."""
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def draw_log(log):
    """Draw line of values (log) to the current figure."""
    plt.plot([i for i in range(len(log))], [i for i in log])


# Visualising approximators

def get_1d_x_line(X_data, dim_id=0):
    """Get ordered list of values between the min and the max of certain dimension
    (column) of the X_data matrix.

    """
    x_max, x_min, y_max, y_min = get_data_stats(X_data, [0], [dim_id])
    return np.arange(x_min, x_max, (x_max - x_min) / 50)


def draw_1d_approximator_params_log(model, X_data, dim_id=0):
    """Draw the lines of the approximation of the model during the
    optimization iteration.

    """
    x_line = get_1d_x_line(X_data, dim_id)
    for params in model.params_log[::int(len(model.params_log) / 20)]:
        plt.plot(x_line, [model.predict([x], params) for x in x_line],
                 ls='dotted', c='gray')


def draw_1d_predictor(model, X_data, dim_id=0):
    """Draw the hypothesis line of the model."""
    x_line = get_1d_x_line(X_data, dim_id)
    plt.plot(x_line, [model.predict([x]) for x in x_line], linewidth=3)


def plot_1d_approximator_stats(model, X_data, y_data, dim_id=0):
    """Plot the predictor, input data and the cost function log
    of approximation model. Show the plot.

    """
    # draw_1d_approximator_params_log(model, X_data, dim_id)
    draw_1d_predictor(model, X_data, dim_id)
    draw_1d_data(X_data, y_data, dim_id)
    new_window()
    draw_log(model.cost_log)

    show_plots()


def get_data_stats(X_data, y_data, dimensions=None):
    """Get the min and max value of X_data matrix and of y_data vector."""
    dimensions = dimensions if dimensions is not None else range(len(y_data))
    x_max = max([max([x_row[d] for d in dimensions]) for x_row in X_data])
    x_min = min([min([x_row[d] for d in dimensions]) for x_row in X_data])
    return x_max, x_min, max(y_data), min(y_data)


def draw_1d_data(X_data, y_data, dim_id=0, point_size=50):
    """Draw certain dimension of input data."""
    for i, row in enumerate(X_data):
        plt.scatter(row[dim_id], y_data[i], marker='o', s=point_size, c='red')


# Visualising classifiers

def plot_2d_classifier_stats(model, X_data, y_data, dim_id_1=0, dim_id_2=1,):
    """Draw classifier gradient log, decision boundary and data points in
    two given dimensions.

    """
    draw_log(model.gradient_log)
    new_window()
    f_manager = plt.get_current_fig_manager()
    draw_2d_decision_boundary(model, X_data, dim_id_1, dim_id_2)
    draw_2d_classifier_data(X_data, y_data, dim_id_1, dim_id_2)

    show_plots()


def draw_2d_classifier_data(X_data, y_data, dim_id_1=0,
                            dim_id_2=1, point_size=50):
    """Draw binary classifier input data."""
    colormap = ['cyan', 'red']
    for i, row in enumerate(X_data):
        plt.scatter(row[dim_id_1], row[dim_id_2],
                    s=point_size, c=colormap[y_data[i]])


def draw_2d_decision_boundary(model, X_data, dim_id_1=0, dim_id_2=1):
    """Draw the decision boundary of binary classifier."""
    x_max, x_min, _, __ = get_data_stats(X_data, [0], [dim_id_1, dim_id_2])
    x_avr = x_max - x_min
    padding = x_avr * 0.15

    x = y = np.arange(x_min - padding, x_max + padding, x_avr / 25)
    X_mesh, Y_mesh = np.meshgrid(x, y)
    Z = [list(zip(X_row, Y_row)) for X_row, Y_row in zip(X_mesh, Y_mesh)]
    Z_mesh = [[model.predict(list(data)) for data in row] for row in Z]
    levels = [-5000, -0.05, 0, 0.015, 5000]

    plt.contourf(X_mesh, Y_mesh, Z_mesh, cmap=plt.cm.winter, levels=levels)
