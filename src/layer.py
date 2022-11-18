import numpy as np
import scipy

class layer:
    def __init__(self, input_size, num_channels, filter_size, filter_matrix, pooling_factor, dp_kernel, zero_padding = (0, 0)):
        self._input_size = input_size
        self._num_channels = num_channels
        self._filter_size = filter_size
        self._pooling_factor = pooling_factor
        self._dp_kernel = dp_kernel
        self._zero_padding = zero_padding

        self.update_filter_matrix(filter_matrix)

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

    @property
    def zero_padding(self):
        return self._zero_padding

    def update_filter_matrix(self, filter_matrix):
        assert filter_matrix.shape[0] == self._filter_size[0] * self._filter_size[1]

        # Z
        self._filter_matrix = filter_matrix
        # Z^T * Z
        m = np.matmul(filter_matrix.transpose(), filter_matrix)
        # k(Z^T * Z)
        m = self._dp_kernel.func(m)

        # TODO: catch error if this fails
        evalues, evectors = np.linalg.eigh(m)
        evalues_n1_4 = np.power(evalues, -1/4)
        evalues_n1_2 = evalues_n1_4 * evalues_n1_4
        evalues_n3_4 = evalues_n1_2 * evalues_n1_4
        
        # A = k(Z^T * Z) ^ -1/2
        self._A = np.matmul(evectors * evalues_n1_2, evectors.transpose())
        # A^1/2
        self._A_1_2 = np.matmul(evectors * evalues_n1_4, evectors.transpose())
        # A^3/2
        self._A_3_2 = np.matmul(evectors * evalues_n3_4, evectors.transpose())


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


    def extract_patches_adj(self, mx):
        x_filter_radius = (self._filter_size[0] - 1) // 2
        y_filter_radius = (self._filter_size[1] - 1) // 2
        
        mx = np.reshape(mx, (self._num_channels * self._filter_size[0] * self._filter_size[1],
                            self._input_size[0] + self._zero_padding[0] * 2 - x_filter_radius * 2,
                            self._input_size[1] + self._zero_padding[1] * 2 - y_filter_radius * 2))

        result_mx = np.zeros((self._num_channels, 
                            self._input_size[0] + self._zero_padding[0] * 2, 
                            self._input_size[1] + self._zero_padding[1] * 2))

        x_diff_range = range(x_filter_radius * 2 + 1)
        y_diff_range = range(y_filter_radius * 2 + 1)

        for patch_mx_x in range(mx.shape[1]):
            for patch_mx_y in range(mx.shape[2]):
                channel_offset = 0

                for x_diff in x_diff_range:
                    for y_diff in y_diff_range:
                        result_x = patch_mx_x + x_diff
                        result_y = patch_mx_y + y_diff

                        for channel in range(self._num_channels):
                            result_mx[channel][result_x][result_y] += mx[channel_offset + channel][patch_mx_x][patch_mx_y]
                        
                        channel_offset += self._num_channels

        if self._zero_padding[0] > 0 or self._zero_padding[1] > 0:
            without_padding = np.empty((self._num_channels, self._input_size[0], self._input_size[1]))
            for x in range(self._input_size[0]):
                for y in range(self._input_size[1]):
                    for channel in range(self._num_channels):
                        without_padding[channel][x][y] = result_mx[channel][x + self._zero_padding[0]][y + self._zero_padding[1]]
            result_mx = without_padding

        result_mx = np.reshape(result_mx, (self._num_channels, self._input_size[0] * self._input_size[1]))
        return result_mx


    def forward(self, input):
        patches_mx = self.extract_patches(input)
        norms = np.linalg.norm(patches_mx, axis = 0)
        normed_patches_mx = patches_mx * (1 / norms)
        # k(Z^T * E(input) * S^-1)
        kerneled = self._dp_kernel.func(np.matmul(self._filter_matrix.transpose(), normed_patches_mx))
        # A * k(Z^T * E(input) * S^-1) * S
        self._before_pooling = np.matmul(self._A, kerneled) * norms
        # TODO: add pooling
        return self._before_pooling

    