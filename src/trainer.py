import numpy as np

class Trainer:
    def __init__(self, optimizer, batch_size, train_input, train_output):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.train_input = train_input
        self.train_output = train_output

        self.batch_counter = 0

    def train(self, epochs):
        for _ in range(epochs):
            permutation = np.random.permutation(len(self.train_input))

            for index in permutation:
                self.optimizer.step(self.train_input[index], self.train_output[index])
                self.batch_counter += 1

                if self.batch_counter >= self.batch_size:
                    self.optimizer.optim()
                    self.batch_counter = 0
