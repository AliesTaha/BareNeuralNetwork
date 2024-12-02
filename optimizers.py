import numpy as np
from activation_functions import *
from loss_functions import *


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.5):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0.
        self.momentum = momentum

    # This is called before the parameters are updated to update the learning rate by decay
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                1./(self.decay * self.iterations + 1.)

    def update_params(self, layer):
        # layer.dweights are the parameter gradients
        if self.momentum:
            # if momentum is used, we need to update the velocity
            # velocity = momentum * velocity - learning_rate * gradient
            # weights = weights + velocity

            # In the beginning, the weight_momentums will not exist, so we need to initialize it
            if not hasattr(layer, 'weight_velocity'):
                layer.weight_velocity = np.zeros_like(layer.weights)
                layer.bias_velocity = np.zeros_like(layer.biases)

            # Update weights
            layer.weight_velocity = self.momentum * layer.weight_velocity - \
                self.current_learning_rate * layer.dweights
            weights_updates = layer.weight_velocity

            # Update biases
            layer.bias_velocity = self.momentum * layer.bias_velocity - \
                self.current_learning_rate * layer.dbiases
            biases_updates = layer.bias_velocity
        else:
            weights_updates = -self.current_learning_rate * \
                layer.dweights
            biases_updates = -self.current_learning_rate * \
                layer.dbiases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def post_update_params(self):
        self.iterations += 1
