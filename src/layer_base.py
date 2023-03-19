class LayerBase:
    def __init__(self, input_size, output_size, in_channels, out_channels):
        self.input_size = input_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input):
        raise NotImplementedError()

    def compute_gradient(self, gradient_calculation_info):
        raise NotImplementedError()
    
    def gradient_descent(self, descent):
        raise NotImplementedError()