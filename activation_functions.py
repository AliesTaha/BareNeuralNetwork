import numpy as np


# relu activation
class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # basically, if the value of the input is <0, the relu function is 0, so the derivative is 0, otherwise it is 1
        # because the derivative of the relu function is 1 (the slope of the x wrt x is 1)
        self.dinputs[self.inputs <= 0] = 0


# softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        self.output_exponentiated = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.output_exponentiated / \
            np.sum(self.output_exponentiated, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
