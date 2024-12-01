# the goal here is to test the loss function and the activation function, and the combined activation and loss function
import numpy as np
import nnfs
from activation_functions import *
from loss_functions import *
nnfs.init()

# I'm just making up some values here, as I only want to test the backprop of the softmax and loss function
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
y_true = np.array([0, 1, 1])

softmax_and_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
activation = Activation_Softmax()
loss = Loss_Categorical_Cross_Entropy()
softmax_and_loss.backward(softmax_outputs, y_true)
dvalues = softmax_and_loss.dinputs

activation.output = softmax_outputs
loss.backward(activation.output, y_true)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs
print(np.round(dvalues, 2))
print(np.round(dvalues2, 2))
