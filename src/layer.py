class Filter:
    def __init__(self, filter_size, out_channels, dp_kernel, zero_padding = 'same', filter_matrix = None):
        self.filter_size = filter_size
        self.out_channels = out_channels
        self.dp_kernel = dp_kernel
        self.zero_padding = zero_padding
        self.filter_matrix = filter_matrix
        
class AvgPooling:
    def __init__(self, filter_size):
        self.filter_size = filter_size

