from itertools import tee

from numpy import exp, random, dot, array


def pairwise(*iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def sigmoid(x):
    return 2 / (1 + exp(-x)) - 1


def sigmoid_derivative(x):
    return 2 * x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_layers, output_nodes):
        # random.seed(1)
        self.weights = [random.random((a, b)) for a, b in pairwise(input_nodes, *hidden_layers, output_nodes)]

    def train(self, train_input, desired_output):
        values = sigmoid(train_input)

        for layer in self.weights:
            print(values)
            values = sigmoid(dot(values, layer))

        return values


if __name__ == "__main__":
    neural_network = NeuralNetwork(4, (4, 3), 4)
    print(*neural_network.weights)
    print(neural_network.train(array([1, 2, 1, 3]), [1, 2, 3, 4]))
