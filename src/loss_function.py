import numpy as np

class LossFunction:
    def loss(self, predicted, expected):
        raise NotImplementedError()

    def gradient(self, predicted, expected):
        raise NotImplementedError()

class SquareHingeLoss(LossFunction):
    def __init__(self, margin):
        self.margin = margin
    
    def loss(self, predicted, expected):
        loss_array_with_expected = np.maximum(self.margin + predicted - predicted[expected], 0)
        loss_array = loss_array_with_expected[np.arange(len(predicted)) != expected]
        return loss_array.sum() ** 2 / len(loss_array)
    
    def gradient(self, predicted, expected):
        loss_array_with_expected = np.maximum(self.margin + predicted - predicted[expected], 0)
        loss_array = loss_array_with_expected[np.arange(len(predicted)) != expected]

        grad = (loss_array_with_expected > 0).astype(float)
        grad[expected] = -np.count_nonzero(loss_array)
        grad *= 2 * loss_array.sum()
        
        return grad / len(loss_array)

class MeanSquaredError(LossFunction):
    def __init__(self):
        pass

    def loss(self, predicted, expected):
        return 1 / predicted.shape[0] * np.square(predicted - expected).sum()
    
    def gradient(self, predicted, expected):
        return 2 * (predicted - expected) / predicted.shape[0]