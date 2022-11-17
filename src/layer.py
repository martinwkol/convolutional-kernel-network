import numpy as np

class layer:
    def __init__(self, input_size, num_channels, filter_size, filter_matrix, pooling_factor, dp_kernel, zero_padding = (0, 0)):
        self._input_size = input_size
        self._num_channels = num_channels
        self._filter_size = filter_size
        self._filter_matrix = filter_matrix
        self._pooling_factor = pooling_factor
        self._dp_kernel = dp_kernel
        self._zero_padding = zero_padding

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def filter_matrix(self):
        return self._filter_matrix

    @property
    def pooling_factor(self):
        return self._pooling_factor

    @property
    def dp_kernel(self):
        return self._dp_kernel

    def extract_patches(self, input):
        x_filter_radius = (self._filter_size[0] - 1) // 2
        y_filter_radius = (self._filter_size[1] - 1) // 2
        
        input = np.reshape(input, (self._num_channels, self._input_size[0], self._input_size[1]))

        if self._zero_padding[0] > 0 or self._zero_padding[1] > 0:
            input = np.pad(input, ((0, 0), (self._zero_padding[0], self._zero_padding[0]), (self._zero_padding[1], self._zero_padding[1])))

        x_diff_range = range(x_filter_radius * 2 + 1)
        y_diff_range = range(y_filter_radius * 2 + 1)
        
        patch_mx_size = (self._num_channels * self._filter_size[0] * self._filter_size[1], 
                        input.shape[1] - x_filter_radius * 2, 
                        input.shape[2] - y_filter_radius * 2)
        patch_mx = np.empty(patch_mx_size)

        for patch_mx_x in range(patch_mx_size[1]):
            for patch_mx_y in range(patch_mx_size[2]):
                channel_offset = 0

                for x_diff in x_diff_range:
                    for y_diff in y_diff_range:
                        input_x = patch_mx_x + x_diff
                        input_y = patch_mx_y + y_diff

                        for channel in range(self._num_channels):
                            patch_mx[channel_offset + channel][patch_mx_x][patch_mx_y] = input[channel][input_x][input_y]
                        
                        channel_offset += self._num_channels

        patch_mx = np.reshape(patch_mx, (patch_mx_size[0], patch_mx_size[1] * patch_mx_size[2]))
        return patch_mx                  


    def forward(self, input):
        pass

    