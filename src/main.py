# Standard library imports
import os
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


def create_mnist_trainer(data, model_layers):
    net = network.Network(input_size=(28, 28), in_channels=1, layer_infos=model_layers, output_nodes=10)
    optimizer = op.Optimizer(network=net, loss_function=loss_function.SquareHingeLoss(margin=0.2))
    trainer = tr.Trainer(
        optimizer=optimizer, batch_size=128, learning_rate=2, regularization_parameter=1/60000,
        train_input=data.train_images, train_output=data.train_labels
    )
    return trainer


def perform_test(trainer, test_images, test_labels, num_tests=math.inf):
    correct_preds = 0

    for j in range(min(num_tests, len(test_images))):
        pred_enc = trainer.optimizer.network.forward(test_images[j])
        pred = np.argmax(pred_enc)
        if pred == test_labels[j]:
            correct_preds += 1

    return correct_preds


def train_network(trainer, epochs, batches_per_test, test_images, test_labels, num_tests_batch=math.inf, num_tests_epoch=math.inf):
    batch_counter = 0
    num_tests_batch = min(num_tests_batch, len(test_images))
    num_tests_epoch = min(num_tests_epoch, len(test_images))

    for _ in range(epochs):
        print(f"Epoch: {trainer.epoch}")
        while True:
            print(f"[E{trainer.epoch}, {trainer.epoch_counter}]", end='\r')
            
            trainer.finish_batch()
            batch_counter += 1

            if batch_counter >= batches_per_test:
                print(f"Test of size {num_tests_batch}\t\t")
                correct_preds = perform_test(trainer, test_images, test_labels, num_tests_batch)
                accuracy = 100.0 * correct_preds / num_tests_batch
                print(f"Accuracy: {correct_preds}/{num_tests_batch} ({accuracy:.2f}%)\n")
                batch_counter = 0

            if trainer.epoch_counter == 0:
                print(f"Test of size {num_tests_epoch}\t\t")
                correct_preds = perform_test(trainer, test_images, test_labels, num_tests_epoch)
                accuracy = 100.0 * correct_preds / num_tests_epoch
                print(f"Accuracy: {correct_preds}/{num_tests_epoch} ({accuracy:.2f}%)\n")
                print('-' * 20)
                print()
                break



def create_analysis():
    mnist = MNIST(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    
    filepath = os.path.join(parent_directory, "analyses/Test.pick")

    trainer = create_mnist_trainer(data=mnist, model_layers=[
        li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        li.AvgPoolingInfo(pooling_size=(3, 3)),

        li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        li.AvgPoolingInfo(pooling_size=(3, 3)),

        li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        
    ])


    analysis = Analysis(trainer, mnist.test_images, mnist.test_labels, num_labels=10)
    for i in range(20):
        print("Epoch {}".format(analysis.trainer.epoch))
        print("Learning rate {}".format(analysis.trainer.learning_rate))
        analysis.perform_analysis(epochs=1, batches_per_test=5, num_test=300)
        analysis.save(filepath=filepath)
        print(str(analysis.test_results_epoch[-1]))
        print()
        print()


def main():
    mnist = MNIST('mnist_alt')

    trainer = create_mnist_trainer(data=mnist, model_layers=[
        li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        li.AvgPoolingInfo(pooling_size=(3, 3)),

        li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        li.AvgPoolingInfo(pooling_size=(3, 3)),

        li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        
    ])

    train_network(trainer=trainer, epochs=20, batches_per_test=100, test_images=mnist.test_images, test_labels=mnist.test_labels, num_tests_batch=1000)

if __name__ == '__main__':
    main()