import numpy as np

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        self.output_exponentiated = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.output_exponentiated / \
            np.sum(self.output_exponentiated, axis=1, keepdims=True)