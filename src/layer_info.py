from filter_layer import FilterLayer
from pooling_layer import PoolingLayer
import numpy as np

class LayerInfoBase:
    def build(self, input_size, in_channels):
        raise NotImplementedError()

class FilterInfo(LayerInfoBase):
    def __init__(self, filter_size, out_channels, dp_kernel, zero_padding = 'same', filter_matrix = None):
        self.filter_size = filter_size
        self.out_channels = out_channels
        self.dp_kernel = dp_kernel
        self.zero_padding = zero_padding
        self.filter_matrix = filter_matrix

    def build(self, input_size, in_channels):
        def create_random_filter_matrix():
            filter_vector_length = in_channels * self.filter_size[0] * self.filter_size[1]
            return np.random.rand(filter_vector_length, self.out_channels)

        def norm_filter_matrix(filter_matrix):
            norms = np.linalg.norm(filter_matrix, axis = 0)
            return filter_matrix / norms

        def zero_padding_str2tuple(zero_padding):
            # TODO: is this correct for zero_padding == 'same'?
            return \
                (0, 0) if zero_padding == 'none' else \
                (self.filter_size[0] // 2, self.filter_size[1] // 2) if zero_padding == 'same' else \
                zero_padding

        filter_matrix = self.filter_matrix if self.filter_matrix is not None else create_random_filter_matrix()
        filter_matrix = norm_filter_matrix(filter_matrix)

        zero_padding = zero_padding_str2tuple(self.zero_padding)

        return FilterLayer(
            input_size=input_size, 
            in_channels=in_channels,
            filter_size=self.filter_size,
            filter_matrix=filter_matrix,
            dp_kernel=self.dp_kernel,
            zero_padding=zero_padding
        )


        
class AvgPoolingInfo(LayerInfoBase):
    def __init__(self, pooling_size):
        self.pooling_size = pooling_size

    def build(self, input_size, in_channels):
        return PoolingLayer(
            input_size=input_size,
            in_channels=in_channels,
            pooling_size=self.pooling_size
        )

