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
        '''
        dvalues is the derivative of the loss function with respect to the output of the softmax layer,
        which is the input of the loss function, which is y_hat
        '''
        samples = len(dvalues)
        labels = len(dvalues[0])
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        '''
        if we assume the following is the output layer, then 
        ◯  [a_0 a_1 a_2]
        ◯  [b_0 b_1 b_2]
        ◯  [c_0 c_1 c_2]
        ◯  [d_0 d_1 d_2]
        ◯  [e_0 e_1 e_2]
        each column above represents the derivative of the overall cost with respect to the input of the softmax layer
        however, we will get it in the form of sample by sample, so 
        [a_0 b_0 c_0 d_0 e_0]
        [a_1 b_1 c_1 d_1 e_1]
        [a_2 b_2 c_2 d_2 e_2]
        
        and as such, the overall number of samples is just the length of this dvalues matrix, and 
        the number of columns is the number of classes, or labels
        '''
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            '''
            We could have also done the following to deal with the case where it is not one hot encoded. 
            tmp = np.zeros((samples, labels))
            tmp[range(samples), y_true] = 1/dvalues[range(samples), y_true]
            self.dinputs = tmp
            return
            '''
        '''
        look at the forward pass. It's disgusting. It deals with the case of one-hot encoding and not. 
        Here, I just say, there will BE one-hot encoding. If it is not one-hot encoded, then we will make it so. 
        np.eye(labels)-> Say there are 4 labels, so we get 
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        Now say y_true is saying [0,0,3,1]. Basically, the 0th label in the 0th sample, the 0th label in the 1st sample,
        the 3rd label in the 2nd sample, and the 1st label in the 3rd sample. 
        So my one hot encoded will select the appropriate rows, by selecting for [y_true] from the np.eye() matrix. 
        That is, we would get
        1 0 0 0 
        1 0 0 0 
        0 0 0 1
        0 1 0 0
        '''
        self.dinputs = -y_true / dvalues
        '''
        Now, we want to take the derivative of the cost with respect to the input of the softmax layer. It's just -y_true/y_hat, 
        but we then have to do this for every sample. 
        '''
        self.dinputs = self.dinputs / samples
        '''
        Optimizers sum all the gradients before multiplying them by learning rate. True, fair. The more samples, the more gradients, the bigger the sum. 
        But we want one learning rate for literally all the samples we have. 
        So we divide gradients by number of samples. The mean is given, and we are done. 
        '''


# Softmax classifier combining softmax activation and categorical cross entropy loss
# Results in faster backward step y_hat - y_true or y_hat-1
class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        # y_pred - y_true is the derivative of the loss function with respect to the input of the softmax layer
        # y_true is always 1. hence
        # dvalues here IS Y_HAT, THE OUTPUT OF THE LAST LAYER.
        self.dinputs[range(samples), y_true] -= 1
        # Note that we are dividing by samples here, because we want the mean of the gradients, so that
        # the optimizer can take a step of learning rate size.
        # and the learning rate doesn't have to be tuned for the number of samples (more samples, more gradients, need smaller learning rate)
        self.dinputs = self.dinputs/samples
