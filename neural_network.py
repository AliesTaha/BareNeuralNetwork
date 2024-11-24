import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


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
        self.sample_losses = self.forward(output, y)
        data_loss = np.mean(self.sample_losses)
        return data_loss


class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if (len(y_true.shape) == 1):  # this is just the true class
            confidences = y_pred_clipped[range(len(y_pred)), y_true]
        # this is one hot encoded. our predictions are also one hot encoded.
        elif (len(y_true.shape) == 2):
            y_hat = y_pred_clipped*y_true
            confidences = np.sum(y_hat, axis=1)
        print("confidences")
        print(confidences)
        loss = -np.log(confidences)
        return loss


X, y = spiral_data(samples=100, classes=3)
loss_function = Loss_Categorical_Cross_Entropy()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()

print(X)

dense1 = Dense_Layer(2, 3)
dense1.forward(X)
activation1 = Activation_Relu()
activation1.forward(dense1.output)
print(activation1.output)
print("--"*20)

dense2 = Dense_Layer(3, 3)
dense2.forward(activation1.output)
print(dense2.output)
print("--"*20)

activation2 = Activation_Softmax()
activation2.forward(dense2.output)

print(activation2.output)
print("--"*20)

print("This is y")
print(y)
total_loss = loss_function.calculate(activation2.output, y)
print("---"*20)
print("This is loss")
print(loss_function.sample_losses)


y_pred = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(y_pred == y)

print(total_loss)
print(accuracy)

dvalues = np.array([[1., 1., 1.]])

print('-'*20+'New section on back propagation'+'-'*20)
dvalues = np.array([
    [1., 1., 1.],
    [2., 2., 2.],
    [3., 3., 3.]
])
weights = np.array(
    [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]).T
dinputs = np.dot(dvalues, weights.T)
print(dinputs)

# Passed in gradient from the next layer
# for the purpose of this example we're going to use an array of incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# sum weights of given input
# and multiply by the passed in gradient for this neuron
dweights = np.dot(inputs.T, dvalues)
print(dweights)
'''
In general, the loss function should follow a specific pattern:
total_loss=np.mean(loss)
loss=-np.log(probabilities_predicted_for_correct_class)

probabilities_predicted_for_correct_class should be 
[0.1, 0.2, 0.3, 0.1, 0.2, 0.2] etc not

[[0,0,0.1],
[0,0,0.2],
[0,0,0.3],
[0,0,0.1],
[0,0,0.2],
[0,0,0.2]]

so we do
probabilities_predicted_for_correct_class=np.sum((y*y_hat), axis=1)

where y is one-hot encoded

so how do we get y_hat?

y_hat=np.exp(output)/np.sum(np.exp(output), axis=1, keepdims=True)
that is, we convert the output to a probability distribution by exponentiating and then normalizing, such that the sum of each row is 1.

finally, output is simple, it is just

layer_output1=np.dot(X, w)+b
output1=np.maximum(layer_output1, 0)

layer_output2=np.dot(output1, w)+b
output2=np.maximum(layer_output2, 0)

layer_output3=np.dot(output2, w)+b
output=layer_output3

'''
