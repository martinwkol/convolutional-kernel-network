import numpy as np
from copy import deepcopy
import pickle
from optimizer import Optimizer
from network import Network

class Trainer:
    def __init__(self, optimizer, learning_rate, regularization_parameter, batch_size, train_images, train_labels):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.batch_size = batch_size
        self.train_images = train_images
        self.train_labels = train_labels
        self._new_epoch()

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

    def next_image(self):
        index = self.permutation[self.epoch_counter]
        self.optimizer.step(self.train_images[index], self.train_labels[index])
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
        self.permutation = np.random.permutation(len(self.train_images))
        # Discard any indices that would result in an incomplete batch
        self.permutation = self.permutation[:len(self.permutation) - (len(self.permutation) % self.batch_size)]

        # Set counters to 0
        self.epoch_counter = 0
        self.batch_counter = 0
        self.loss_sum = 0
        self.optimized_data_counter = 0

    def save_to_file(self, file):
        if isinstance(file, str):
            with open(file, "wb") as f:
                return self.save_to_file(f)
        
        self.optimizer.save_to_file(file)
        self.best_network.save_to_file(file)

        pickle.dump(
            (
                self.learning_rate,
                self.regularization_parameter,
                self.batch_size,

                self.bestaverage_loss_epoch,
                self.average_loss_batch,
                self.average_loss_epoch,
                self.learning_rates,

                self.permutation,

                self.epoch_counter,
                self.batch_counter,
                self.loss_sum,
                self.optimized_data_counter,
            ),
            file
        )

    @staticmethod
    def load_from_file(file, train_images, train_labels):
        if isinstance(file, str):
            with open(file, "rb") as f:
                return Trainer.load_from_file(f, train_images, train_labels)
        
        trainer = Trainer.__new__(Trainer)

        trainer.optimizer = Optimizer.load_from_file(file)
        trainer.best_network = Network.load_from_file(file)
        (
            trainer.learning_rate,
            trainer.regularization_parameter,
            trainer.batch_size,
        
            trainer.bestaverage_loss_epoch,
            trainer.average_loss_batch,
            trainer.average_loss_epoch,
            trainer.learning_rates,

            trainer.permutation,
            
            trainer.epoch_counter,
            trainer.batch_counter,
            trainer.loss_sum,
            trainer.optimized_data_counter,
        ) = pickle.load(file)

        trainer.train_images = train_images
        trainer.train_labels = train_labels

        return trainer