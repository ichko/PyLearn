"""Module for loading data from files into numpy matrices.

"""


def open_file(file_path):
    return open(file_path)


def pull_data_matrix(file_path, col_delimiter):
    """Function for loading data from file into matrix.
    New lines corresponds to new rows and the delimiter for columns
    is given with col_delimiter (string).

    """
    with open_file(file_path) as f:
        lines = f.readlines()
        return [[float(item.strip()) for item in line.split(col_delimiter)]
                for line in lines]


def parse_data(file_path, col_delimiter=','):
    """Function calling pull_data_matrix and returning tuple of
    the matrix without the last column and the last column.

    """
    matrix = pull_data_matrix(file_path, col_delimiter)
    return [row[:-1] for row in matrix], [row[-1] for row in matrix]
