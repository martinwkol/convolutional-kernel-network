import numpy as np
from copy import deepcopy
import pickle

class Trainer:
    def __init__(self, optimizer, learning_rate, regularization_parameter, batch_size, train_input = None, train_output = None):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.batch_size = batch_size
        self.set_training_data(train_input, train_output)

        self.best_network = deepcopy(self.optimizer.network)

        self._best_average_loss_epoch = float('inf')
        self._average_loss_batch = []
        self._average_loss_epoch = []
        self._learning_rates = []

    
    @property
    def batch(self):
        return len(self._average_loss_batch) + 1

    @property
    def epoch(self):
        return len(self._average_loss_epoch) + 1

    @property
    def epoch_size(self):
        return len(self._permutation) 

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


    def __getstate__(self):
        return { k: v for (k, v) in self.__dict__.items() if k not in ["_train_input", "_train_output"] }

    def __setstate__(self, d):
        self.__dict__ = d
        self._train_input = None
        self._train_output = None


    def set_training_data(self, train_input, train_output):
        assert len(train_input) == len(train_output)

        self._train_input = train_input
        self._train_output = train_output
        self._new_epoch()       


    def next_image(self):
        if self._train_input is None or self._train_output is None:
            raise TypeError("train_input or train_output is None")

        index = self._permutation[self._permutation_index]
        self.optimizer.step(self._train_input[index], self._train_output[index])
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
                self._learning_rates.append(self.learning_rate)
                if average_loss < self._best_average_loss_epoch:
                    self._best_average_loss_epoch = average_loss
                    self.best_network = deepcopy(self.optimizer.network)

                elif average_loss > self._best_average_loss_epoch:
                    self.optimizer.network = deepcopy(self.best_network)
                    self.learning_rate /= 2
                
                self._new_epoch()


    def _new_epoch(self):
        self._permutation = np.random.permutation(len(self._train_input))
        self._permutation = self._permutation[:len(self._permutation) - (len(self._permutation) % self.batch_size)]
        self._permutation_index = 0
        self._batch_counter = 0
        self._loss_sum = 0
        self._optimized_data_counter = 0

