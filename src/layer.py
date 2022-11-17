import numpy as np

class layer:
    def __init__(self, input_size, num_channels, patch_size, pooling_factor, dp_kernel, filter, use_zero_padding = False):
        self._input_size = input_size
        self._num_channels = num_channels
        self._patch_size = patch_size
        self._pooling_factor = pooling_factor
        self._dp_kernel = dp_kernel
        self._filter = filter
        self._use_zero_padding = use_zero_padding

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
        x_patch_radius = (self._patch_size[0] - 1) // 2
        y_patch_radius = (self._patch_size[1] - 1) // 2
        
        input = np.reshape(input, (self._num_channels, self._input_size[0], self._input_size[1]))

        if self._use_zero_padding:
            input = np.pad(input, ((0, 0), (x_patch_radius, x_patch_radius), (y_patch_radius, y_patch_radius)))

        x_diff_range = range(x_patch_radius * 2 + 1)
        y_diff_range = range(y_patch_radius * 2 + 1)
        
        patch_mx_size = (self._num_channels * self._patch_size[0] * self._patch_size[1], 
                        input.shape[1] - x_patch_radius * 2, 
                        input.shape[2] - y_patch_radius * 2)
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

    