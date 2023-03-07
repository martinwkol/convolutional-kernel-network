class GradientCalculationInfo:
    def __init__(self, last_output_after_pooling, U, U_upscaled, layer_number):
        # last_output_after_pooling = I_j = M_j P_j   (M_j convolutional layer output, P_j pooling matrix)
        self.last_output_after_pooling = last_output_after_pooling

        # input matrix for the functions g_j(U), h_j(U)
        self.U = U

        # U_upscaled = U P^T    (P pooling matrix)
        self.U_upscaled = U_upscaled
        
        self.layer_number = layer_number