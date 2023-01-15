import numpy as np

class Trainer:
    def __init__(self, network, optimizer, learning_rate, batch_size, train_input, train_output):
        self.network = network
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_input = train_input
        self.train_output = train_output

        self.batch_counter = 0

    def train(self, epochs):
        self.optimizer.set_network(network=self.network)

        for _ in range(epochs):
            permutation = np.random.permutation(len(self.train_input))

            for index in permutation:
                self.optimizer.step(self.train_input[index], self.train_output[index])
                self.batch_counter += 1

                if self.batch_counter >= self.batch_size:
                    self.optimizer.optim(self.learning_rate)
                    self.batch_counter = 0
