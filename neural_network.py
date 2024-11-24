import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from dense_layer import Dense_Layer
from activation_functions import Activation_Relu, Activation_Softmax
from loss_functions import Loss_Categorical_Cross_Entropy

# Initialize the nnfs library, which sets the random seed and other configurations for reproducibility
nnfs.init()

X, y = spiral_data(samples=100, classes=3)
loss_function = Loss_Categorical_Cross_Entropy()

print(X)

dense1 = Dense_Layer(2, 3)
dense1.forward(X)
activation1 = Activation_Relu()
activation1.forward(dense1.output)
print(activation1.output)
print("--" * 20)

dense2 = Dense_Layer(3, 3)
dense2.forward(activation1.output)
print(dense2.output)
print("--" * 20)

activation2 = Activation_Softmax()
activation2.forward(dense2.output)

print(activation2.output)
print("--" * 20)

print("This is y")
print(y)
total_loss = loss_function.calculate(activation2.output, y)
print("---" * 20)
print("This is loss")
print(loss_function.sample_losses)

y_pred = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(y_pred == y)

print(total_loss)
print(accuracy)

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
