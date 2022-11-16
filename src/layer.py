import numpy as np

class layer:
    def __init__(self, input_size, num_channels, patch_size, pooling_factor, dp_kernel, filter):
        self._input_size = input_size
        self._num_channels = num_channels
        self._patch_size = patch_size
        self._pooling_factor = pooling_factor
        self._dp_kernel = dp_kernel
        self._filter = filter

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def pooling_factor(self):
        return self._pooling_factor

    @property
    def dp_kernel(self):
        return self._dp_kernel

    @property
    def filter(self):
        return self._filter

    def extract_patches(self, input):
        (channels, pixels) = input.shape
        x_diff = (self._patch_size[0] - 1) // 2
        y_diff = (self._patch_size[1] - 1) // 2
        patch_mx = np.empty((channels * self._patch_size[0] * self._patch_size[1], pixels))
        for y in range(self._input_size[1]):
            for x in range(self._input_size[0]):
                patch_pixel = x + y * self._input_size[0]
                row = 0
                for py in range(y - y_diff, y + y_diff + 1):
                    for px in range(x - x_diff, x + x_diff + 1):
                        input_pixel = px + py * self._input_size[0]
                        if 0 <= px < self._input_size[0] and 0 <= py < self._input_size[1]:
                            for channel in range(self._num_channels):
                                patch_mx[row][patch_pixel] = input[channel][input_pixel]
                                row += 1
                        else:
                            for _ in range(self._num_channels):
                                patch_mx[row][patch_pixel] = 0 # zero padding
                                row += 1
        return patch_mx
                            

    def forward(self, input):
        pass

    