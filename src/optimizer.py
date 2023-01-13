from copy import deepcopy
import numpy as np

class Optimizer:
    def __init__(self, network, learning_rate, loss_function, output_activation_func = None, max_loss_increase = 2):
        self.network = network
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        # TODO: find better name
        self.output_activation_func = output_activation_func if output_activation_func is not None else lambda x: x
        self.max_loss_increase = max_loss_increase

        self.best_network = network
        self.loss_history = []
        self.loss_increase_counter = 0
        self.min_loss = float('inf')

        self.loss_sum = 0
        self.gradent_sum = None
        self.current_batch_size = 0


    def step(self, training_input, expected_output):
        pred_raw = self.network.forward(training_input)
        pred = self.output_activation_func(pred_raw)

        self.current_batch_size += 1
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

        #print(self.gradent_sum[1].shape)
        #print(gradients[1].shape)
        self.gradent_sum[1] += gradients[1]



    def optim(self):
        if self.current_batch_size == 0:
            return

        adjusted_learning_rate = self.learning_rate * self.current_batch_size
        adjusted_loss = self.loss_sum / self.current_batch_size

        for j in range(len(self.gradent_sum[0])):
            if self.gradent_sum[0][j] is None:
                continue

            new_filter_matrix = self.network.layers[j].filter_matrix - \
                                adjusted_learning_rate * self.gradent_sum[0][j] 
            norms = np.linalg.norm(new_filter_matrix, axis = 0)
            self.network._layers[j].update_filter_matrix(new_filter_matrix / norms)
        
        self.network.output_weights -= adjusted_learning_rate * self.gradent_sum[1]

        if adjusted_loss < self.min_loss:
            self.min_loss = adjusted_loss
            self.loss_increase_counter = 0
            # TODO: more efficient copy
            self.best_network = deepcopy(self.network)

        elif adjusted_loss > self.min_loss:
            self.loss_increase_counter += 1

            if self.loss_increase_counter >= self.max_loss_increase:
                self.learning_rate /= 2
                # TODO: more efficient copy
                self.network = deepcopy(self.best_network)
                self.loss_increase_counter = 0


        self.loss_history.append(adjusted_loss)
        
        self.loss_sum = 0
        self.gradent_sum = None
        self.current_batch_size = 0

        return adjusted_loss