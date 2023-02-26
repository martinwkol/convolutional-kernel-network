class LayerBase:
    def build(self, input_size, in_channels):
        raise NotImplementedError()

class IntLayerBase:
    def __init__(self, input_size, output_size, in_channels, out_channels):
        self._input_size = input_size
        self._output_size = output_size
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def compute_gradient(self, gradient_calculation_info):
        raise NotImplementedError()