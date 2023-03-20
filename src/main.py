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



def create_analysis(mnist, filepath, epochs, trainer, batches_per_test=100, num_tests_batch=1000, num_tests_epoch=math.inf):
    if os.path.exists(filepath):
        analysis = Analysis.load_from_file(filepath, mnist.train_images, mnist.train_labels, mnist.test_images, mnist.test_labels)
    else:
        analysis = Analysis(trainer, mnist.test_images, mnist.test_labels, num_labels=10)

    while analysis.trainer.epoch <= epochs:
        print("Epoch {}".format(analysis.trainer.epoch))
        print("Learning rate {}".format(analysis.trainer.learning_rate))
        analysis.perform_analysis(epochs=1, batches_per_test=batches_per_test, num_tests_batch=num_tests_batch)
        analysis.save_to_file(filepath)
        print(str(analysis.test_results_epoch[-1]))
        print()
        print()


def main():
    mnist = MNIST(directory='mnist')

    create_analysis(
        mnist=mnist, 
        filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../analyses/ana_3_3x3_layers_10_filters_3x3_pooling"),
        epochs=20,
        trainer=create_mnist_trainer(data=mnist, model_layers=[
            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4))
        ])
    )



    create_analysis(
        mnist=mnist, 
        filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../analyses/ana_3_5x5_layers_10_filters_3x3_pooling"),
        epochs=20,
        trainer=create_mnist_trainer(data=mnist, model_layers=[
            li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(5, 5), zero_padding='same', out_channels=10, dp_kernel=kernel.RadialBasisFunction(alpha=4))
        ])
    )


    create_analysis(
        mnist=mnist, 
        filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../analyses/ana_3_3x3_2_1x1_layers_5_filters__3x3_pooling__zp"),
        epochs=20,
        trainer=create_mnist_trainer(data=mnist, model_layers=[
            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=5, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),
            li.FilterInfo(filter_size=(1, 1), zero_padding='same', out_channels=5, dp_kernel=kernel.RadialBasisFunction(alpha=4)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=5, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),
            li.FilterInfo(filter_size=(1, 1), zero_padding='same', out_channels=5, dp_kernel=kernel.RadialBasisFunction(alpha=4)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=5, dp_kernel=kernel.RadialBasisFunction(alpha=4))
        ])
    )



    create_analysis(
        mnist=mnist, 
        filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../analyses/ana_2_3x3_layers_15_filters__3x3_pooling__no_zp"),
        epochs=20,
        trainer=create_mnist_trainer(data=mnist, model_layers=[
            li.FilterInfo(filter_size=(3, 3), zero_padding='none', out_channels=15, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
            li.AvgPoolingInfo(pooling_size=(3, 3)),

            li.FilterInfo(filter_size=(3, 3), zero_padding='none', out_channels=15, dp_kernel=kernel.RadialBasisFunction(alpha=4)),
        ])
    )


    #train_network(trainer=trainer, epochs=20, batches_per_test=100, test_images=mnist.test_images, test_labels=mnist.test_labels, num_tests_batch=1000)

if __name__ == '__main__':
    main()