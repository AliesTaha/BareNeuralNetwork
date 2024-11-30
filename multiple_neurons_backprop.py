import numpy as np
dvalues = np.array([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]
])

weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]).T

inputs = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

biases = np.array([2, 3, 0.5])

z = np.dot(inputs, weights)+biases
relu = np.maximum(z, 0)

# This means we've done a complete forward pass. It is now time to do a complete backward pass
# We shall assume the values passed by the next layer during back propagation were all ones, that is
# drelu will simulate the derivative with respect to input values
drelu = relu.copy()
drelu[z <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, dvalues)
dz_dbiases = np.ones(weights.T.shape[0])
dbiases = np.dot(dz_dbiases, dvalues)

# or we could do
dbiases = np.sum(dvalues, axis=0, keepdims=True)

# as for the relu function, we first have to find outputs
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
drelu = dvalues.copy()
drelu[z < 0] = 0

# ----------------------------------------------------------------------------
# Full Pass example
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases  # Dense layer
relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation

# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0
print(drelu)

# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)

# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)

# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)
