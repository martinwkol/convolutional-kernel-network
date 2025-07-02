# End-to-End Kernel Learning with Supervised Convolutional Kernel Networks

This repository implements **Supervised Convolutional Kernel Networks (SCKNs)**, based on:

**Mairal, J. (2016). End-to-End Kernel Learning with Supervised Convolutional Kernel Networks.**  
[https://doi.org/10.48550/arXiv.1605.06265](https://doi.org/10.48550/arXiv.1605.06265)

## Overview

The goal of this project is to provide a modular, lightweight implementation of **Supervised Convolutional Kernel Networks (SCKNs)** for use in research and experimentation. It is implemented in pure `numpy` and avoids deep learning frameworks.

While the MNIST dataset is used as a default example, the core components are general and can be reused for other tasks and datasets.

## Installation

### Requirements

- Python 3.x
- `numpy`
- `matplotlib` (optional, for visualizations)

Install dependencies with:

```sh
pip install numpy matplotlib
````

## Usage

This repository is intended as a **general framework** for building and training convolutional kernel networks. You can use it to define custom architectures, datasets, loss functions, and training workflows.

A complete example is provided in [`src/train_mnist.py`](src/train_mnist.py), which trains and evaluates an SCKN on the MNIST dataset.

Below is a minimal outline for using the library programmatically.

### 1. Load or provide input data

You can use the built-in MNIST loader or prepare your own dataset in similar format.

```python
from mnist import MNIST
mnist = MNIST(directory='mnist')
train_images, train_labels = mnist.train_images, mnist.train_labels
test_images, test_labels = mnist.test_images, mnist.test_labels
```

### 2. Define your network architecture

```python
from layer_info import FilterInfo, AvgPoolingInfo
from kernel import RadialBasisFunction

model_layers = [
    FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=RadialBasisFunction(alpha=4)),
    AvgPoolingInfo(pooling_size=(3, 3)),
    FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=RadialBasisFunction(alpha=4)),
    AvgPoolingInfo(pooling_size=(3, 3)),
    FilterInfo(filter_size=(3, 3), zero_padding='same', out_channels=10, dp_kernel=RadialBasisFunction(alpha=4)),
]
```

### 3. Train and evaluate your network

```python
from network import Network
from loss_function import SquareHingeLoss
from optimizer import Optimizer
from trainer import Trainer

net = Network(input_size=(28, 28), in_channels=1, layer_infos=model_layers, output_nodes=10)
optimizer = Optimizer(network=net, loss_function=SquareHingeLoss(margin=0.2))
trainer = Trainer(
    optimizer=optimizer,
    batch_size=128,
    learning_rate=2.0,
    regularization_parameter=1 / 60000,
    train_images=train_images,
    train_labels=train_labels
)

# Training loop
for _ in range(10):
   trainer.finish_epoch()

# Evaluate accuracy
import numpy as np
correct = sum(
    np.argmax(trainer.optimizer.network.forward(img)) == label
    for img, label in zip(test_images, test_labels)
)
accuracy = 100.0 * correct / len(test_images)
print(f"Test Accuracy: {accuracy:.2f}%")
```

### 4. Save and load training checkpoints

```python
trainer.save_to_file("models/my_trainer.pkl")

# Later
from trainer import Trainer
trainer = Trainer.load_from_file("models/my_trainer.pkl", train_images, train_labels)
```

## Exploring the Theory

The file [`Mathematical background and experimental results.ipynb`](./Mathematical%20background%20and%20experimental%20results.ipynb) includes:

* A detailed explanation of the underlying mathematical concepts.
* Sample visualizations and results.

To run:

```sh
jupyter notebook "Mathematical background and experimental results.ipynb"
```

## Repository Structure

```
.
├── analyses
├── Mathematical background and experimental results.ipynb
├── Mathematical background and experimental results.pdf
├── mnist
├── README.md
├── src
│   ├── analysis.py
│   ├── create_analyses.py
│   ├── filter_layer.py
│   ├── gradient_calculation_info.py
│   ├── kernel.py
│   ├── layer_base.py
│   ├── layer_info.py
│   ├── loss_function.py
│   ├── mnist.py
│   ├── network.py
│   ├── optimizer.py
│   ├── pooling_layer.py
│   ├── trainer.py
│   └── train_mnist.py        # Example usage
└── test
    ├── layer_test.py
    └── pooling_layer_test.py
```

## Acknowledgments

* **Julien Mairal** for the original research paper.
* The MNIST dataset for serving as a testbed for experiments.