import numpy as np
from layer_base import LayerBase
from gradient_calculation_info import GradientCalculationInfo
import itertools

class FilterLayer(LayerBase):
    def __init__(self, input_size, in_channels, filter_size, filter_matrix, dp_kernel, zero_padding = (0, 0)):
        super().__init__(
            input_size=input_size, 
            output_size=(
                input_size[0] + zero_padding[0] * 2 - (filter_size[0] - 1),
                input_size[1] + zero_padding[1] * 2 - (filter_size[1] - 1)
            ), 
            in_channels=in_channels, 
            out_channels=filter_matrix.shape[1]
        )

        self.filter_size = filter_size
        self.dp_kernel = dp_kernel
        self.zero_padding = zero_padding
        
        self.filter_matrix = filter_matrix

        self.last_input = None
        self.last_output = None

        # E(input)
        self._E_input = None

        # S (diagonal elements)
        self._S_diag = None

         # S^-1 (diagonal elements)
        self._S_n1_diag = None

        # Z^T E(input) S^-1
        self._Z_T__E_input__S_n1 = None

    @property
    def filter_matrix(self):
        return self._filter_matrix
    
    @filter_matrix.setter
    def filter_matrix(self, filter_matrix):
        assert filter_matrix.shape[0] == self.filter_size[0] * self.filter_size[1] * self.in_channels

        # Calculate Z^T Z, k'(Z^T Z), and k(Z^T Z) + eI
        self._filter_matrix = filter_matrix
        self._Z_T__Z = filter_matrix.transpose() @ filter_matrix
        self._k_d_Z_T__Z = self.dp_kernel.deriv(self._Z_T__Z)
        k_Z_T__Z__eI = self.dp_kernel.func(self._Z_T__Z) + np.diag(np.full(self._Z_T__Z.shape[0], 0.001))

        # Calculate A, A^(1/2), and A^(3/2)
        evalues, evectors = np.linalg.eigh(k_Z_T__Z__eI)
        evalues_n1_4 = np.power(evalues, -1/4)
        evalues_n1_2 = evalues_n1_4 * evalues_n1_4
        evalues_n3_4 = evalues_n1_2 * evalues_n1_4
        self._A = (evectors * evalues_n1_2) @ evectors.transpose()
        self._A_1_2 = (evectors * evalues_n1_4) @ evectors.transpose()
        self._A_3_2 = (evectors * evalues_n3_4) @ evectors.transpose()

    
    def forward(self, input):
        self.last_input = input

        # E(input)
        self._E_input = self._extract_patches(input)

        # S (diagonal elements)
        self._S_diag = np.linalg.norm(self._E_input, axis = 0)
        # avoid 0 entries so that we are able to invert S
        self._S_diag += np.full(len(self._S_diag), 0.00001)

        # S^-1 (diagonal elements)
        self._S_n1_diag = 1 / self._S_diag

        # Z^T E(input) S^-1
        self._Z_T__E_input__S_n1 = self._filter_matrix.transpose() @ (self._E_input * self._S_n1_diag)

        # k(Z^T E(input) S^-1)
        kerneled = self.dp_kernel.func(self._Z_T__E_input__S_n1)

        # M = A k(Z^T E(input) S^-1) S
        self.last_output = (self._A @ kerneled) * self._S_diag

        return self.last_output

    
    def compute_gradient(self, gradient_calculation_info):
        gci = gradient_calculation_info
        B = self._calculate_B(gci.U_upscaled)
        C = self._calculate_C(gci.U, gci.last_output_after_pooling)
        gradient = self._g(B, C)

        if gci.layer_number == 0:
            return gradient, None

        U = self._h(gci.U_upscaled, B)
        new_info = GradientCalculationInfo(
            last_output_after_pooling=self.last_input, 
            U=U, 
            U_upscaled=U,
            layer_number=gci.layer_number-1
        )
        return gradient, new_info


    def gradient_descent(self, descent):
        new_filter_matrix = self._filter_matrix - descent
        norms = np.linalg.norm(new_filter_matrix, axis = 0)
        self.filter_matrix = new_filter_matrix / norms

    
    def _calculate_B(self, U_upscaled):
        # U_upscaled = U P^T
        # B = k'(Z^T E(input) S^-1) * (A U P^T)
        return self.dp_kernel.deriv(self._Z_T__E_input__S_n1) * (self._A @ U_upscaled)


    def _calculate_C(self, U, next_filter_layer_input):
        # C = A^1/2 output U^T A^3/2
        return (self._A_1_2 @ next_filter_layer_input) @ (U.transpose() @ self._A_3_2)


    def _g(self, B, C):
        # g(U) = E(input) B^T - 1/2 Z (k'(Z^T Z) * (C + C^T))

        # E(input) B^T
        E_input__B_T = self._E_input @ B.transpose()

        # k'(Z^T Z) * (C + C^T)
        k_d_Z_T__Z__mul__C_plus_C_T = self._k_d_Z_T__Z * (C + C.transpose())

        # E(input) B^T - 1/2 Z (k'(Z^T Z) * (C + C^T))
        g_U = E_input__B_T - (1/2 * self.filter_matrix) @ k_d_Z_T__Z__mul__C_plus_C_T

        return g_U


    def _h(self, U_upscaled, B):
        # U_upscaled = U P^T
        # X = S^-2 * (M^T U P^T - E(input)^T Z B))    (X is a diagonal matrix)
        # h(U) = E_adj( Z B + E(input) X )

        # Z B
        Z_B = self.filter_matrix @ B

        # M^T U P^T         (diagonal elements)
        M_T__U__P_T__diag = np.einsum('ij,ji->i', self.last_output.transpose(), U_upscaled)

        # E(input)^T Z B    (diagonal elements)
        E_input_T__Z__B__diag = np.einsum('ij,ji->i', self._E_input.transpose(), Z_B)

        # X = S^-2 * (M^T U P^T - E(input)^T Z B))      (diagonal elements)
        X_diag = self._S_n1_diag * self._S_n1_diag * (M_T__U__P_T__diag - E_input_T__Z__B__diag)
        
        # h(U) = E_adj( Z B + E(input) X )
        h_U = self._extract_patches_adj(Z_B + self._E_input * X_diag)

        return h_U


    def _extract_patches(self, input):
        # Reshape input into a 3D matrix with shape (in_channels, input_size[0], input_size[1])
        input = np.reshape(input, (self.in_channels, self.input_size[0], self.input_size[1]))

        # Add zero padding to the edges of the input if necessary
        if self.zero_padding[0] > 0 or self.zero_padding[1] > 0:
            input = np.pad(input, ((0, 0), (self.zero_padding[0], self.zero_padding[0]), 
                                (self.zero_padding[1], self.zero_padding[1])))

        # Calculate the size of the output patch matrix
        patch_mx_size = (self.in_channels * self.filter_size[0] * self.filter_size[1], 
                        input.shape[1] - (self.filter_size[0] - 1), 
                        input.shape[2] - (self.filter_size[1] - 1))
        # Create an empty patch matrix with the calculated size
        patch_mx = np.empty(patch_mx_size)

        # Loop over each pair (x_offset, y_offset) and save the values from the input matrix in the corresponding channels 
        # in the patch matrix
        channel_offset = 0
        for x_offset in range(self.filter_size[0]):
            for y_offset in range(self.filter_size[1]):
                # Extract patches from the input matrix
                patch_mx[channel_offset : channel_offset + self.in_channels, :, :] = \
                    input[:, x_offset : patch_mx_size[1] + x_offset, y_offset : patch_mx_size[2] + y_offset]
                channel_offset += self.in_channels

        # Reshape the patch matrix into a 2D matrix with shape (in_channels * filter_size[0] * filter_size[1], num_patches)
        patch_mx = np.reshape(patch_mx, (patch_mx_size[0], patch_mx_size[1] * patch_mx_size[2]))

        return patch_mx
    

    def _extract_patches_adj(self, mx):        
        mx = mx.reshape(-1, self.output_size[0], self.output_size[1])

        # Initialize a zero-filled array with the size of the original input with zero-padding
        adj_patched = np.zeros((self.in_channels, self.input_size[0] + self.zero_padding[0] * 2, self.input_size[1] + self.zero_padding[1] * 2))
        
        # Sum all extracted patches to their original position
        channel_offset = 0
        for x_diff, y_diff in itertools.product(range(self.filter_size[0]), range(self.filter_size[1])):
            start_channel = (x_diff * self.filter_size[1] + y_diff) * self.in_channels
            end_channel = start_channel + self.in_channels

            adj_patched[:, 
                x_diff:self.output_size[0] + x_diff,
                y_diff:self.output_size[1] + y_diff
            ] += mx[start_channel:end_channel, :, :]

        # If the input had zero padding, remove it from the result
        if self.zero_padding[0] > 0 or self.zero_padding[1] > 0:
            start_x = self.zero_padding[0]
            end_x = self.input_size[0] + self.zero_padding[0]
            start_y = self.zero_padding[1]
            end_y = self.input_size[1] + self.zero_padding[1]

            adj_patched = adj_patched[:, start_x:end_x, start_y:end_y]

        adj_patched = adj_patched.reshape(-1, self.input_size[0] * self.input_size[1])
        return adj_patched

    