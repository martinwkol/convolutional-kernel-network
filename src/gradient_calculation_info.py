class GradientCalculationInfo:
    def __init__(self, next_filter_layer_input, next_U, next_U_upscaled, layer_number):
        self.next_filter_layer_input = next_filter_layer_input
        self.next_U = next_U
        self.next_U_upscaled = next_U_upscaled
        self.layer_number = layer_number