from copy import deepcopy
import numpy as np

class Optimizer:
    def __init__(self, loss_function, network):
        self.loss_function = loss_function
        self._network = None

        self.set_network(network)

    
    @property
    def network(self):
        return self._network

    def set_network(self, network):
        if self._network is not network:
            self._network = network
            self.loss_sum = 0
            self.gradent_sum = None
            self.num_steps = 0


    def step(self, training_input, expected_output):
        # TODO: network is NULL except

        pred = self._network.forward(training_input)

        self.loss_sum += self.loss_function.loss(predicted=pred, expected=expected_output)

        gradients = self._network.gradients(loss_func=self.loss_function, expected_output=expected_output)
        
        if self.gradent_sum is None:
            # TODO: ugly
            self.gradent_sum = [gradients[0], gradients[1]]
            return

        for j in range(len(gradients[0])):
            if gradients[0][j] is None or self.gradent_sum[0][j] is None:
                continue
            self.gradent_sum[0][j] += gradients[0][j]

        self.gradent_sum[1] += gradients[1]

        self.num_steps += 1


    def optim(self, learning_rate, regularization_parameter):
        # TODO: network is NULL except
        if self.num_steps == 0:
            return

        grad_sum_scalar = learning_rate / self.num_steps

        for j in range(len(self.gradent_sum[0])):
            if self.gradent_sum[0][j] is None:
                continue

            new_filter_matrix = self.network.layers[j].filter_matrix - \
                                grad_sum_scalar * self.gradent_sum[0][j] 
            norms = np.linalg.norm(new_filter_matrix, axis = 0)
            self.network._layers[j].update_filter_matrix(new_filter_matrix / norms)
        
        self.network.output_weights -= grad_sum_scalar * self.gradent_sum[1] 
        self.network.output_weights *= 1 - learning_rate * regularization_parameter

        loss = self.loss_sum 
        self.loss_sum = 0
        self.gradent_sum = None
        self.num_steps = 0

        return loss