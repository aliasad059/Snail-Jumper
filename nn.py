import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes, activation_function="Sigmoid"):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.layer_sizes = layer_sizes
        self.network_size = len(layer_sizes)
        self.activation_function = activation_function
        self.weights = {}
        self.biases = {}

        for i in range(self.network_size - 1):
            self.weights[i] = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            self.biases[i] = np.zeros((1, layer_sizes[i + 1]))
        pass

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        if self.activation_function == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "ReLU":
            return np.maximum(0, x)
        else:
            raise Exception("Unknown activation function")

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        for i in range(self.network_size - 1):
            x = self.activation(np.dot(x, self.weights[i]) + self.biases[i])
        return x
