import numpy as np
from activation_functions import *


# loss class
class Loss:
    def calculate(self, output, y):
        self.sample_losses = self.forward(output, y)
        self.data_loss = np.mean(self.sample_losses)
        return self.data_loss


# categorical cross entropy loss
class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            y_hat = y_pred_clipped * y_true
            confidences = np.sum(y_hat, axis=1)

        neg_log = -np.log(confidences)
        return neg_log

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# Softmax classifier combining softmax activation and categorical cross entropy loss
# Results in faster backward step y_hat - y_true or y_hat-1
class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.loss_val = self.loss.calculate(self.output, y_true)
        return self.loss_val

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples