import numpy as np

class LossFunction:
    def __init__(self, loss, gradient):
        self._loss = loss
        self._gradient = gradient

    def loss(self, predicted, expected):
        return self._loss(predicted, expected)

    def gradient(self, predicted, expected):
        return self._gradient(predicted, expected)

def get_square_hinge_loss(margin):
    def loss(predicted, expected):
        loss_array_with_expected = np.maximum(margin + predicted - predicted[expected], 0)
        loss_array = loss_array_with_expected[np.arange(len(predicted)) != expected]
        return loss_array.sum() ** 2 / len(loss_array)

    def gradient(predicted, expected):
        loss_array_with_expected = np.maximum(margin + predicted - predicted[expected], 0)
        loss_array = loss_array_with_expected[np.arange(len(predicted)) != expected]

        grad = (loss_array_with_expected > 0).astype(float)
        grad[expected] = -np.count_nonzero(loss_array)
        grad *= 2 * loss_array.sum()  

        return grad / len(loss_array)

    return LossFunction(
        loss=loss, 
        gradient=gradient
    )

mse = LossFunction(
    loss=lambda predicted, expected: 1 / predicted.shape[0] * np.square(predicted - expected).sum(), 
    gradient=lambda predicted, expected: 2 * (predicted - expected) / predicted.shape[0]
)