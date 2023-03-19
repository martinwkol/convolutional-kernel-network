"""
Most of this code was written by ChatGPT
"""

import os
import urllib.request
import gzip
import numpy as np

class MNIST:
    def __init__(self, directory = 'mnist'):
        data_dir = MNIST.download_mnist(directory)

        with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as f:
            self.train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28) / 255
        with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            self.train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            self.test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28) / 255
        with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            self.test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    @staticmethod
    def download_mnist(directory):
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', directory)
        os.makedirs(data_dir, exist_ok=True)

        for file_name in files:
            file_path = os.path.join(data_dir, file_name)

            if not os.path.exists(file_path):
                url = base_url + file_name
                urllib.request.urlretrieve(url, file_path)

        return data_dir