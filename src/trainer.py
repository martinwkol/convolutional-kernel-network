import numpy as np
from copy import deepcopy

class Trainer:
    def __init__(self, network, optimizer, learning_rate, regularization_parameter, batch_size, train_input, train_output):
        self.network = network
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.batch_size = batch_size
        self.train_input = train_input
        self.train_output = train_output

        self.batch_counter = 0
        self.last_average_loss = float('inf')
        self.best_network = network

    def train(self, epochs):
        for _ in range(epochs):
            self.optimizer.set_network(network=self.network)
            
            permutation = np.random.permutation(len(self.train_input))
            loss_sum = 0
            optimized_data_counter = 0

            for index in permutation:
                self.optimizer.step(self.train_input[index], self.train_output[index])
                self.batch_counter += 1

                if self.batch_counter >= self.batch_size:
                    loss_sum += self.optimizer.optim(self.learning_rate, self.regularization_parameter)
                    optimized_data_counter += self.batch_counter
                    self.batch_counter = 0

            average_loss = loss_sum / optimized_data_counter
            if average_loss < self.last_average_loss:
                self.last_average_loss = average_loss
                self.best_network = deepcopy(self.network)

            elif average_loss > self.last_average_loss:
                self.network = deepcopy(self.best_network)
                self.learning_rate /= 2

