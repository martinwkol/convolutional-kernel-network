import numpy as np

import sys
import os
import struct
from array import array

class MNIST:
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.train_images, self.train_labels = MNIST.read_images_labels(training_images_filepath, training_labels_filepath)
        self.test_images, self.test_labels = MNIST.read_images_labels(test_images_filepath, test_labels_filepath)
        
    @staticmethod
    def read_images_labels(images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        images = []
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]) / 255
            img = img.reshape(28, 28)
            images[i] = img            
        
        return images, labels