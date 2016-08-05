import matplotlib.pyplot as plt
import numpy as np


def quit_figure(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def plot_2d_classifier_stats(model, X_data, y_data, dim_id_1=0, dim_id_2=1,):
    f_manager = plt.get_current_fig_manager()
    f_manager.window.move(650, 0)
    draw_error_log(model)

    new_window()
    f_manager = plt.get_current_fig_manager()
    f_manager.window.move(0, 0)
    draw_2d_decision_boundary(model, X_data, dim_id_1, dim_id_2)
    draw_2d_data(X_data, y_data, dim_id_1, dim_id_2)

    show_plots()


def draw_2d_data(X_data, y_data, dim_id_1=0, dim_id_2=1, point_size=50):
    colormap = ['cyan', 'red']
    for i, row in enumerate(X_data):
        plt.scatter(row[dim_id_1], row[dim_id_2],
                    s=point_size, c=colormap[y_data[i]])


def draw_error_log(model):
    plt.plot([i for i in range(len(model.error_log))],
             [i for i in model.error_log])


def draw_2d_decision_boundary(model, X_data, dim_id_1=0, dim_id_2=1):
    x_max = max([max(x_row[dim_id_1], x_row[dim_id_2]) for x_row in X_data])
    x_min = min([min(x_row[dim_id_1], x_row[dim_id_2]) for x_row in X_data])
    x_avr = x_max - x_min
    padding = x_avr * 0.15

    x = y = np.arange(x_min - padding, x_max + padding, x_avr / 25)
    X_mesh, Y_mesh = np.meshgrid(x, y)
    Z = [list(zip(X_row, Y_row)) for X_row, Y_row in zip(X_mesh, Y_mesh)]
    Z_mesh = [[model.unthresholded(list(data)) for data in row] for row in Z]
    levels = [-5000, -0.05, 0, 0.015, 5000]

    plt.contourf(X_mesh, Y_mesh, Z_mesh, cmap=plt.cm.winter, levels=levels)


def show_plots():
    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.show()


def new_window():
    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.figure()
