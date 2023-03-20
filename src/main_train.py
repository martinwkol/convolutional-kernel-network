# Standard library imports
import os, sys
import argparse
import math

# Third-party library imports
import numpy as np
from mnist import MNIST
from matplotlib import pyplot as plt

# Local imports
import kernel
import layer_info as li
import network
import loss_function
import optimizer as op
import trainer as tr
from analysis import Analysis
from trainer import Trainer


def create_mnist_trainer(data, model_layers):
    net = network.Network(input_size=(28, 28), in_channels=1, layer_infos=model_layers, output_nodes=10)
    optimizer = op.Optimizer(network=net, loss_function=loss_function.SquareHingeLoss(margin=0.2))
    trainer = tr.Trainer(
        optimizer=optimizer, batch_size=128, learning_rate=2, regularization_parameter=1/60000,
        train_images=data.train_images, train_labels=data.train_labels
    )
    return trainer


def perform_test(trainer, test_images, test_labels, num_tests):
    num_tests = min(num_tests, len(test_images))
    print(f"Test of size {num_tests}\t\t")

    correct_preds = 0
    for j in range(num_tests):
        pred_enc = trainer.optimizer.network.forward(test_images[j])
        pred = np.argmax(pred_enc)
        if pred == test_labels[j]:
            correct_preds += 1

    accuracy = 100.0 * correct_preds / num_tests
    print(f"Accuracy: {correct_preds}/{num_tests} ({accuracy:.2f}%)\n")
    print('-' * 20)
    print()


def train_network(trainer, filepath, test_images, test_labels, epochs, num_tests, epochs_btw_tests):
    epoch_test_counter = 0

    while trainer.epoch <= epochs:
        print(f"Epoch: {trainer.epoch}")
        while True:
            print(f"[E{trainer.epoch}, {trainer.epoch_counter}]", end='\r')
            trainer.finish_batch()
            if trainer.epoch_counter == 0:
                break
        
        trainer.save_to_file(filepath)
        epoch_test_counter += 1
        if epoch_test_counter >= epochs_btw_tests:
            perform_test(trainer, test_images, test_labels, num_tests)
            epoch_test_counter = 0


def main():
    parser = argparse.ArgumentParser(description='Train a convolutional kernel network on the MNIST dataset')
    parser.add_argument('-f', help='Path to the file for the trainer', type=str, dest="filepath")
    parser.add_argument('-m', help='Path to the directory of the mnist dataset', type=str, dest="mnist_dir", default='mnist')
    parser.add_argument('-e', help='Number of epochs the networks is supposed to be trained for', type=int, dest="epochs", default=10)
    parser.add_argument('-nt', help="Number of tests to perform (<= 0 for all tests)", type=int, dest="num_tests", default=-1)
    parser.add_argument('-et', help="Number of epochs between tests (<= for no tests)", type=int, dest="epochs_btw_tests", default=1)
    parser.add_argument('--initial-test', help="Perform a test of the network before starting with training", action='store_true', dest='initial_test')
    args = parser.parse_args()

    filepath = args.filepath
    mnist_dir = args.mnist_dir
    epochs = args.epochs
    num_tests = args.num_tests if args.num_tests > 0 else math.inf
    epochs_btw_tests = args.epochs_btw_tests if args.epochs_btw_tests > 0 else math.inf
    initial_test = args.initial_test

    if not filepath:
        print("Error: No filepath for the trainer provided")
        exit()

    mnist = MNIST(directory=mnist_dir)

    if not os.path.isfile(filepath):
        trainer = create_mnist_trainer(data=mnist, model_layers=[
            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4))
        ])

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        trainer.save_to_file(filepath)

    else:
        trainer = Trainer.load_from_file(filepath, train_images=mnist.train_images, train_labels=mnist.train_labels)

    if initial_test:
        perform_test(
            trainer=trainer, 
            test_images=mnist.test_images, 
            test_labels=mnist.test_labels,
            num_tests=num_tests
        )

    train_network(
        trainer=trainer,
        filepath=filepath,
        test_images=mnist.test_images, 
        test_labels=mnist.test_labels, 
        epochs=epochs, 
        num_tests=num_tests, 
        epochs_btw_tests=epochs_btw_tests
    )

if __name__ == '__main__':
    main()