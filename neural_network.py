import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights_unscaled = np.random.randn(n_inputs, n_neurons)
        self.weights = self.weights_unscaled * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights)+self.biases


class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        self.output_exponentiated = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True))

        # keepdims=true because I want a column vector of the sums, not a list.
        self.output = self.output_exponentiated / \
            np.sum(self.output_exponentiated, axis=1, keepdims=True)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if (len(y_true.shape) == 1):
            confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif (len(y_true.shape) == 2):
            y_hat = y_pred_clipped*y_true
            confidences = np.sum(y_hat, axis=1)
        loss = -np.log(confidences)
        return loss


X, y = spiral_data(samples=100, classes=3)
loss_function = Loss_Categorical_Cross_Entropy()

dense1 = Dense_Layer(2, 3)
dense1.forward(X)
activation1 = Activation_Relu()
activation1.forward(dense1.output)


dense2 = Dense_Layer(3, 3)
dense2.forward(activation1.output)
activation2 = Activation_Softmax()
activation2.forward(dense2.output)

total_loss = loss_function.calculate(activation2.output, y)
print(total_loss)

y_pred = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(y_pred == y)

print(total_loss)
print(accuracy)
