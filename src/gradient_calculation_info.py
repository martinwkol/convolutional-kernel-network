class GradientCalculationInfo:
    def __init__(self, last_output_after_pooling, U, U_upscaled, layer_number):
        self.last_output_after_pooling = last_output_after_pooling
        self.U = U
        self.U_upscaled = U_upscaled
        self.layer_number = layer_number