import numpy as np

class DotProductKernel:
    def func(self, x):
        raise NotImplementedError()

    def deriv(self, x):
        raise NotImplementedError()

class RadialBasisFunction(DotProductKernel):
    def __init__(self, alpha):
        self.alpha = alpha

    def func(self, x):
        return np.exp((x - 1) * self.alpha)
    
    def deriv(self, x):
        return self.alpha * np.exp((x - 1) * self.alpha)
        