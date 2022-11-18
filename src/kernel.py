import numpy as np

class dot_product_kernel:
    def __init__(self, function, derivative):
        self._function = function
        self._derivative = derivative

    def func(self, x):
        return self._function(x)

    def deriv(self, x):
        return self._derivative(x)

def get_rbf(alpha):
    return dot_product_kernel(
        lambda x: np.exp((x - 1) * alpha), 
        lambda x: alpha * np.exp((x - 1) * alpha)
    )