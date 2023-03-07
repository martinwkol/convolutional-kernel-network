import numpy as np
from copy import deepcopy

class Trainer:
    def __init__(self, optimizer, learning_rate, regularization_parameter, batch_size, train_input, train_output):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.batch_size = batch_size
        self.train_input = train_input
        self.train_output = train_output

        self.best_network = deepcopy(self.optimizer.network)

        self._best_average_loss_epoch = float('inf')
        self._average_loss_batch = []
        self._average_loss_epoch = []
        
        self._new_epoch()

    
    @property
    def batch(self):
        return len(self._average_loss_batch)

    @property
    def epoch(self):
        return len(self._average_loss_epoch)

    @property
    def best_average_loss_epoch(self):
        return self._best_average_loss_epoch

    @property
    def average_loss_batch(self):
        return self._average_loss_batch

    @property
    def average_loss_epoch(self):
        return self._average_loss_epoch

    @property
    def batch_counter(self):
        return self._batch_counter

    @property
    def epoch_counter(self):
        return self._permutation_index


    def next_image(self):
        index = self._permutation[self._permutation_index]
        self.optimizer.step(self.train_input[index], self.train_output[index])
        self._batch_counter += 1
        self._permutation_index += 1

        self._check_batch_epoch()


    def finish_batch(self):
        self.next_image()
        while self._batch_counter > 0:
            self.next_image()    


    def finish_epoch(self):
        self.next_image()
        while self._permutation_index > 0:
            self.next_image()
    
    
    def _check_batch_epoch(self):
        if self._batch_counter >= self.batch_size:
            loss_batch = self.optimizer.optim(self.learning_rate, self.regularization_parameter)
            self._average_loss_batch.append(loss_batch)
            self._loss_sum += loss_batch
            self._optimized_data_counter += self._batch_counter
            self._batch_counter = 0
        
            if self._permutation_index >= len(self._permutation):
                average_loss = self._loss_sum / self._optimized_data_counter
                self._average_loss_epoch.append(average_loss)
                if average_loss < self._best_average_loss_epoch:
                    self._best_average_loss_epoch = average_loss
                    self.best_network = deepcopy(self.optimizer.network)

                elif average_loss > self._best_average_loss_epoch:
                    self.optimizer.network = deepcopy(self.best_network)
                    self.learning_rate /= 2
                
                self._new_epoch()


    def _new_epoch(self):
        self._permutation = np.random.permutation(len(self.train_input))
        self._permutation = self._permutation[:len(self._permutation) - (len(self._permutation) % self.batch_size)]
        self._permutation_index = 0
        self._batch_counter = 0
        self._loss_sum = 0
        self._optimized_data_counter = 0

