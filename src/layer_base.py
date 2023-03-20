import pickle

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

    @staticmethod
    def load_from_file(file):
        if isinstance(file, str):
            with open(file, "rb") as f:
                return LayerBase.load_from_file(f)
        
        init_derived_layers()

        derived_layer_name = file.readline().decode().strip()
        derived_layer = LayerBase.derived_layers[derived_layer_name]

        if derived_layer is None:
            raise ValueError(f"Unknown derived layer: {derived_layer_name}")

        return derived_layer._derived_load_from_file(file)
    
    @classmethod
    def register_derived(cls, derived_layer):
        if not issubclass(derived_layer, LayerBase):
            raise ValueError("Derived layer must be a subclass of LayerBase")
        cls.derived_layers[derived_layer.__name__] = derived_layer

    derived_layers = {}

def init_derived_layers():
    from filter_layer import FilterLayer
    from pooling_layer import PoolingLayer

    LayerBase.register_derived(FilterLayer)
    LayerBase.register_derived(PoolingLayer)