import numpy as np

class layer:
    def __init__(self, patch_size, num_channels, pooling_factor, dp_kernel, filter):
        self._patch_size = patch_size
        self._num_channels = num_channels
        self._pooling_factor = pooling_factor
        self._dp_kernel = dp_kernel
        self._filter = filter

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_channels(self):
        return self._num_channels
    
    @property
    def pooling_factor(self):
        return self._pooling_factor

    @property
    def dp_kernel(self):
        return self._dp_kernel

    @property
    def filter(self):
        return self._filter

    