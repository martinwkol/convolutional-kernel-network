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
from main_train import create_mnist_trainer


def create_analysis(mnist, filepath, epochs, trainer, batches_per_test=100, num_tests_batch=1000, num_tests_epoch=math.inf):
    if os.path.exists(filepath):
        analysis = Analysis.load_from_file(filepath, mnist.train_images, mnist.train_labels, mnist.test_images, mnist.test_labels)
    else:
        analysis = Analysis(trainer, mnist.test_images, mnist.test_labels, num_labels=10)

    while analysis.trainer.epoch <= epochs:
        print("Epoch {}".format(analysis.trainer.epoch))
        analysis.perform_analysis(epochs=1, batches_per_test=batches_per_test, num_tests_batch=num_tests_batch)
        analysis.save_to_file(filepath)
        print(str(analysis.test_results_epoch[-1]))
        print()
        print()


def main():
    mnist = MNIST(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mnist'))

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

if __name__ == '__main__':
    main()