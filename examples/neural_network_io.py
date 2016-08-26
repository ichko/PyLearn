import numpy as np
import json


def _network_to_json(network):
    weights = list([list(l) for l in l_weights]
                   for l_weights in network.weights)
    biases = list(list(b) for b in network.biases)

    return json.dumps({"weights": weights, "biases": biases})


def _json_to_network(json_input):
    python_data = json.loads(json_input)
    biases = [np.array(b) for b in python_data['biases']]
    weights = [np.array(w) for w in python_data['weights']]

    return biases, weights


def json_export(network, path):
    output_string = _network_to_json(network)
    with open(path, 'w') as file:
        file.write(output_string)


def json_import(network, path):
    with open(path, 'r') as file:
        json_input = file.read()
        biases, weights = _json_to_network(json_input)

        network.biases = biases
        network.weights = weights
