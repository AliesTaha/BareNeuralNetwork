import nnfs
from nnfs.datasets import spiral_data
from activation_functions import *
from loss_functions import *
from dense_layer import *

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
print(X.shape, y.shape)
# I get 300 samples, of 2 features each, and 300 y labels, [0 , 1 , 2 , ...etc]
# Create Dense layer with 2 input features and 64 output values
dense1 = Dense_Layer(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_Relu()
# Create second Dense layer with 64 input features (as we take output # of previous layer here) and 3 output values (output values)
dense2 = Dense_Layer(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
