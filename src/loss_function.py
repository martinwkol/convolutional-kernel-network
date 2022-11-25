import numpy as np

class LossFunction:
    def __init__(self, loss, gradient):
        self._loss = loss
        self._gradient = gradient

    def loss(self, predicted, expected):
        return self._loss(predicted, expected)

    def gradient(self, predicted, expected):
        return self._gradient(predicted, expected)

mse = LossFunction(
    loss=lambda predicted, expected: 1 / predicted.shape[0] * np.square(predicted - expected).sum(), 
    gradient=lambda predicted, expected: 2 * (predicted - expected) / predicted.shape[0]
)