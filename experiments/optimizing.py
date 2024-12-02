import nnfs
from nnfs.datasets import spiral_data
from activation_functions import *
from loss_functions import *
from dense_layer import *
from optimizers import *

'''
# Playing around with decay rate
starting_learning_rate = 1
decay = 0.1
for step in range(20):
    learning_rate = starting_learning_rate * 1/(decay * step + 1)
    print(learning_rate)
'''

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

optimizer = Optimizer_SGD(learning_rate=1, decay=1e-3, momentum=0.9)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(f'epoch: {epoch}, ' +
              f'accuracy: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
