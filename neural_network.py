import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense_layer import *
from activation_functions import *
from loss_functions import *
from optimizers import *
# Initialize the nnfs library, which sets the random seed and other configurations for reproducibility
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# 1st dense layer created with an activation relu function following it
dense1 = Dense_Layer(2, 3)
activation1 = Activation_Relu()

# 2nd dense layer created with an activation softmax function following it
# but instead of an activation softmax and a loss function, we will combine the 2
# to make one final output layer (loss+activation)
dense2 = Dense_Layer(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# forward pass 1
dense1.forward(X)
activation1.forward(dense1.output)

# forward pass 2
dense2.forward(activation1.output)

# forward pass (activation and loss function)
# output of 2nd layer-> returns loss
# stores activation output
loss = loss_activation.forward(dense2.output, y)

# See the first few samples of the output of the softmax layer
print(loss_activation.output[:5])
print('loss:', loss)

# See the predictions
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print(f'acc: {accuracy:.3f}')

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Now, we can adjust weights and biases, lowering loss.
# This is the job of the optimizer (adjusting weights and biases using gradients to decrease loss)
print("dweights and biases of dense1")
print(dense1.dweights)
print(dense1.dbiases)
print('-'*10)
print("dweights and biases of dense2")
print(dense2.dweights)
print(dense2.dbiases)
print('-'*20)


print("weights and biases of dense1")
print(dense1.weights)
print(dense1.biases)
print('-'*10)
print("weights and biases of dense2")
print(dense2.weights)
print(dense2.biases)
optimizer_sgd = Optimizer_SGD(learning_rate=1.0)
optimizer_sgd.update_params(dense1)
optimizer_sgd.update_params(dense2)

print('-'*20)
print("weights and biases of dense1 after optimizer")
print(dense1.weights)
print(dense1.biases)
print('-'*10)
print("weights and biases of dense2 after optimizer")
print(dense2.weights)
print(dense2.biases)
