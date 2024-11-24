import numpy as np


# dense layer
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights_unscaled = np.random.randn(n_inputs, n_neurons)
        self.weights = self.weights_unscaled * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Parameters gradient
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Values gradient
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
