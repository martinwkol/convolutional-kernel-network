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
        (channels, pixels) = input.shape

        x_patch_radius = (self._patch_size[0] - 1) // 2
        y_patch_radius = (self._patch_size[1] - 1) // 2

        x_diff_range = range( -x_patch_radius, x_patch_radius + 1 )
        y_diff_range = range( -y_patch_radius, y_patch_radius + 1 )

        x_offset = 0 if self._use_zero_padding else x_patch_radius
        y_offset = 0 if self._use_zero_padding else y_patch_radius
        
        patch_mx_size = (self._input_size[0] - x_offset * 2, self._input_size[1] - y_offset * 2)
        patch_mx = np.zeros((channels * self._patch_size[0] * self._patch_size[1], patch_mx_size[0] * patch_mx_size[1]))

        current_patch_mx_row = 0
        for y_diff in y_diff_range:
            for x_diff in x_diff_range:
                if self._use_zero_padding:
                    patch_mx_x_range = range(max(0, -x_diff), patch_mx_size[0] + min(0, -x_diff))
                    patch_mx_y_range = range(max(0, -y_diff), patch_mx_size[1] + min(0, -y_diff))
                else:
                    patch_mx_x_range = range(patch_mx_size[0])
                    patch_mx_y_range = range(patch_mx_size[1])

                for patch_mx_y in patch_mx_y_range:
                    input_y = patch_mx_y + y_offset + y_diff

                    patch_mx_pixel_row = patch_mx_y * patch_mx_size[0]
                    input_pixel_row = input_y * self._input_size[0]

                    for patch_mx_x in patch_mx_x_range:
                        input_x = patch_mx_x + x_offset + x_diff 

                        patch_mx_pixel = patch_mx_x + patch_mx_pixel_row
                        input_pixel = input_x + input_pixel_row
                        
                        for channel in range(channels):
                            patch_mx[current_patch_mx_row + channel][patch_mx_pixel] = input[channel][input_pixel]

                current_patch_mx_row += channels

        return patch_mx                            


    def forward(self, input):
        pass

    