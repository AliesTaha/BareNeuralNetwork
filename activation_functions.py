import numpy as np


# relu activation
class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # basically, if the value of the input is <0, the relu function is 0, so the derivative is 0, otherwise it is 1
        # because the derivative of the relu function is 1 (the slope of the x wrt x is 1)
        self.dinputs[self.inputs <= 0] = 0


# softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        self.output_exponentiated = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.output_exponentiated / \
            np.sum(self.output_exponentiated, axis=1, keepdims=True)

    def backward(self, dvalues):
        '''
        This raises a question — if sample-wise gradients are the Jacobian matrices, how do we perform the chain rule
        with the gradient back-propagated from the loss function, since it’s a vector for each sample? Also, what do we do with the fact that the previous layer, 
        which is the Dense layer, will expect the gradients to be a 2D array? Currently, we have a 3D array of the partial derivatives — a list of the Jacobian matrices.
        The derivative of the softmax function is 
        [
          | ∂S₁/∂z₁ | ∂S₂/∂z₁ | ∂S₃/∂z₁ |
          | ∂S₁/∂z₂ | ∂S₂/∂z₂ | ∂S₃/∂z₂ |
          | ∂S₁/∂z₃ | ∂S₂/∂z₃ | ∂S₃/∂z₃ |
        ]
        dvalues = [dc/dz1, dc/dz2, dc/dz3]
        '''
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues.T)


soft = Activation_Softmax()
soft.forward(np.array([
    [2, 1, 0.1]
]))
print(np.round(soft.output, decimals=1))
dvalues = np.array([[1.0, -1.0, 0.5]])
soft.backward(dvalues)
print(soft.dinputs)

print('-'*10)
z = np.array([2, 1, 1])
S = np.array([0.7, 0.2, 0.1])
dvalues = np.array([1.0, -1.0, 0.5])

J = np.diag(S) - np.dot(S.reshape(-1, 1), S.reshape(1, -1))
'''
J =
| ∂S₁/∂z₁ | ∂S₂/∂z₁ | ∂S₃/∂z₁ |
| ∂S₁/∂z₂ | ∂S₂/∂z₂ | ∂S₃/∂z₂ |
| ∂S₁/∂z₃ | ∂S₂/∂z₃ | ∂S₃/∂z₃ |
and 
dvalues = [dc/dz1, dc/dz2, dc/dz3]

J= 
[[ 0.21 -0.14 -0.07]
 [-0.14  0.16 -0.02]
 [-0.07 -0.02  0.09]]
 
dvalues = [1.0, -1.0, 0.5]

now, dc/dz=ds/dz* dc/ds
where dc/ds is dvalues
and ds/dz is J

dL/dz_1 = (dS_1/dz_1) * (dL/dS_1) + (dS_2/dz_1) * (dL/dS_2) + (dS_3/dz_1) * (dL/dS_3)

'''
dz_1 = (0.21 * 1.0) + (-0.14 * -1.0) + (-0.07 * 0.5)
dz_2 = (-0.14 * 1.0) + (0.16 * -1.0) + (-0.02 * 0.5)
dz_3 = (-0.07 * 1.0) + (-0.02 * -1.0) + (0.09 * 0.5)
# np.array([dz_1, dz_2, dz_3])==np.dot(J, dvalues.T)
