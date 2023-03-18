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

        self.bestaverage_loss_epoch = float('inf')
        self.average_loss_batch = []
        self.average_loss_epoch = []
        self.learning_rates = []
    
    @property
    def batch(self):
        return len(self.average_loss_batch) + 1

    @property
    def epoch(self):
        return len(self.average_loss_epoch) + 1

    @property
    def epoch_size(self):
        return len(self.permutation) 

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_train_input']
        del state['_train_output']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
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

        index = self.permutation[self.epoch_counter]
        self.optimizer.step(self._train_input[index], self._train_output[index])
        self.batch_counter += 1
        self.epoch_counter += 1

        self._check_batch_epoch()

    def finish_batch(self):
        self.next_image()
        while self.batch_counter > 0:
            self.next_image()    

    def finish_epoch(self):
        self.next_image()
        while self.epoch_counter > 0:
            self.next_image()
    
    def _check_batch_epoch(self):
        # Check if enough images have been processed to complete a batch
        if self.batch_counter >= self.batch_size:
            loss_batch = self.optimizer.optim(self.learning_rate, self.regularization_parameter)
            self.average_loss_batch.append(loss_batch)
            self.loss_sum += loss_batch

            self.optimized_data_counter += self.batch_counter
            self.batch_counter = 0

            # Check if all images in the epoch have been processed
            if self.epoch_counter >= len(self.permutation):
                average_loss = self.loss_sum / self.optimized_data_counter
                self.average_loss_epoch.append(average_loss)
                self.learning_rates.append(self.learning_rate)

                # Check if the current epoch had the best average loss so far
                if average_loss < self.bestaverage_loss_epoch:
                    # If the current epoch had the best average loss so far, update the best average loss and copy the network
                    self.bestaverage_loss_epoch = average_loss
                    self.best_network = deepcopy(self.optimizer.network)
                elif average_loss > self.bestaverage_loss_epoch:
                    # Otherwise restore the best network and reduce the learning rate by half
                    self.optimizer.network = deepcopy(self.best_network)
                    self.learning_rate /= 2
                
                # Reset the counters for the batch and epoch, and start a new epoch
                self._new_epoch()

    def _new_epoch(self):
        # Shuffle the indices of the training data
        self.permutation = np.random.permutation(len(self._train_input))
        # Discard any indices that would result in an incomplete batch
        self.permutation = self.permutation[:len(self.permutation) - (len(self.permutation) % self.batch_size)]

        # Set counters to 0
        self.epoch_counter = 0
        self.batch_counter = 0
        self.loss_sum = 0
        self.optimized_data_counter = 0

