from copy import deepcopy
import numpy as np

class Optimizer:
    def __init__(self, loss_function, network = None, output_activation_func = None, max_loss_increase = 2):
        self.loss_function = loss_function
        self.network = network
        # TODO: find better name
        self.output_activation_func = output_activation_func if output_activation_func is not None else lambda x: x

        self.loss_sum = 0
        self.gradent_sum = None

    
    def set_network(self, network):
        self.network = network
        self.loss_sum = 0
        self.gradent_sum = None


    def step(self, training_input, expected_output):
        # TODO: network is NULL except

        pred_raw = self.network.forward(training_input)
        pred = self.output_activation_func(pred_raw)

        self.loss_sum += self.loss_function.loss(predicted=pred, expected=expected_output)

        gradients = self.network.gradients(loss_func=self.loss_function, expected_output=expected_output)
        
        if self.gradent_sum is None:
            # TODO: ugly
            self.gradent_sum = [gradients[0], gradients[1]]
            return

        for j in range(len(gradients[0])):
            if gradients[0][j] is None or self.gradent_sum[0][j] is None:
                continue
            self.gradent_sum[0][j] += gradients[0][j]


        self.gradent_sum[1] += gradients[1]


    def optim(self, learning_rate):
        # TODO: network is NULL except

        for j in range(len(self.gradent_sum[0])):
            if self.gradent_sum[0][j] is None:
                continue

            new_filter_matrix = self.network.layers[j].filter_matrix - \
                                learning_rate * self.gradent_sum[0][j] 
            norms = np.linalg.norm(new_filter_matrix, axis = 0)
            self.network._layers[j].update_filter_matrix(new_filter_matrix / norms)
        
        self.network.output_weights -= learning_rate * self.gradent_sum[1]

        loss = self.loss_sum 
        self.loss_sum = 0
        self.gradent_sum = None

        return loss