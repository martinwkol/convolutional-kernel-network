from copy import deepcopy
import numpy as np

class Optimizer:
    def __init__(self, network, loss_function):
        self._network = network
        self.loss_function = loss_function
        self.reset()

    
    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        if self._network is not network:
            self._network = network
            self.reset()


    def step(self, training_input, expected_output):
        # TODO: network is NULL except

        predicted = self._network.forward(training_input)
        self._loss_sum += self.loss_function.loss(predicted=predicted, expected=expected_output)
        loss_func_gradient = self.loss_function.gradient(self._network.last_output, expected_output)
        gradients = self._network.gradients(loss_func_gradient)
        
        if self._gradent_sum is not None:
            for j in range(len(gradients)):
                self._gradent_sum[j] += gradients[j]

        else:
            self._gradent_sum = gradients

        self._num_steps += 1


    def optim(self, learning_rate, regularization_parameter):
        # TODO: network is NULL except
        if self._num_steps == 0:
            return

        gradient_sum_multiplier = learning_rate / self._num_steps
        regularization_term = np.sum(self.network.output_weights * self.network.output_weights) * regularization_parameter / 2

        for j in range(len(self._gradent_sum) - 1):
            self.network.layers[j].gradient_descent(gradient_sum_multiplier * self._gradent_sum[j])
        
        self.network.output_weights *= 1 - learning_rate * regularization_parameter
        self.network.output_weights -= gradient_sum_multiplier * self._gradent_sum[-1] 

        loss = self._loss_sum / self._num_steps + regularization_term
        self.reset()

        return loss

    
    def reset(self):
        self._loss_sum = 0
        self._gradent_sum = None
        self._num_steps = 0