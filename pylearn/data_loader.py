def pull_data_matrix(file_path, col_delimiter):
    with open(file_path) as f:
        lines = f.readlines()
        return [[float(item.strip()) for item in line.split(col_delimiter)]
                for line in lines]


def parse_data(file_path, col_delimiter=','):
    matrix = pull_data_matrix(file_path, col_delimiter)
    return [row[:-1] for row in matrix], [row[-1] for row in matrix]
