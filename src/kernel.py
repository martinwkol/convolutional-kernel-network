import numpy as np

class dot_product_kernel:
    def __init__(self, function, derivative):
        self._function = function
        self._derivative = derivative

    @property
    def function(self):
        return self._function

    @property
    def derivative(self):
        return self._derivative

def get_rbf(alpha):
    return dot_product_kernel(
        lambda x: np.exp((x - 1) * alpha), 
        lambda x: alpha * np.exp((x - 1) * alpha)
    )