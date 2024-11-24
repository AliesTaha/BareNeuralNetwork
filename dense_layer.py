import numpy as np

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights_unscaled = np.random.randn(n_inputs, n_neurons)
        self.weights = self.weights_unscaled * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases