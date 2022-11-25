import numpy as np

class int_filter_layer:
    def __init__(self, input_size, num_channels, filter_size, filter_matrix, dp_kernel, zero_padding = (0, 0)):
        self._input_size = input_size
        self._num_channels = num_channels
        self._filter_size = filter_size
        self._dp_kernel = dp_kernel
        self._zero_padding = zero_padding

        self.update_filter_matrix(filter_matrix)

        self._output_size = \
                (self._input_size[0] + self._zero_padding[0] * 2 - (self.filter_size[0] - 1),
                self._input_size[1] + self._zero_padding[1] * 2 - (self.filter_size[1] - 1))

        self._last_input = None
        self._last_output = None
        self._E_input = None
        self._S_diag = None
        self._S_n1_diag = None
        self._Z_T__E_input__S_n1 = None

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

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
    def dp_kernel(self):
        return self._dp_kernel

    @property
    def zero_padding(self):
        return self._zero_padding

    @property
    def last_output(self): 
        return self._last_output


    def update_filter_matrix(self, filter_matrix):
        assert filter_matrix.shape[0] == self._filter_size[0] * self._filter_size[1] * self._num_channels

        # Z
        self._filter_matrix = filter_matrix
        # Z^T * Z
        self._Z_T__Z = np.matmul(filter_matrix.transpose(), filter_matrix)
        # k(Z^T * Z) + eI
        m = self._dp_kernel.func(self._Z_T__Z) + np.diag(np.full(self._Z_T__Z.shape[0], 0.001))

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
        input = np.reshape(input, (self._num_channels, self._input_size[0], self._input_size[1]))

        if self._zero_padding[0] > 0 or self._zero_padding[1] > 0:
            input = np.pad(input, ( (0, 0), (self._zero_padding[0], self._zero_padding[0]), 
                                    (self._zero_padding[1], self._zero_padding[1])))
       
        patch_mx_size = (self._num_channels * self._filter_size[0] * self._filter_size[1], 
                        input.shape[1] - (self._filter_size[0] - 1), 
                        input.shape[2] - (self._filter_size[1] - 1))
        patch_mx = np.empty(patch_mx_size)

        channel_offset = 0
        for x_diff in range(self._filter_size[0]):
            for y_diff in range(self._filter_size[1]):
                patch_mx[channel_offset : channel_offset + self._num_channels, :, :] = \
                    input[:, x_diff : patch_mx_size[1] + x_diff, y_diff : patch_mx_size[2] + y_diff]
                channel_offset += self._num_channels

        patch_mx = np.reshape(patch_mx, (patch_mx_size[0], patch_mx_size[1] * patch_mx_size[2]))
        return patch_mx
    

    def extract_patches_adj(self, mx):        
        mx = np.reshape(mx, (self._num_channels * self._filter_size[0] * self._filter_size[1],
                            self._input_size[0] + self._zero_padding[0] * 2 - (self._filter_size[0] - 1),
                            self._input_size[1] + self._zero_padding[1] * 2 - (self._filter_size[1] - 1)))

        result_mx = np.zeros((self._num_channels, 
                            self._input_size[0] + self._zero_padding[0] * 2, 
                            self._input_size[1] + self._zero_padding[1] * 2))

        channel_offset = 0
        for x_diff in range(self._filter_size[0]):
            for y_diff in range(self._filter_size[1]):
                result_mx[:, x_diff : mx.shape[1] + x_diff, y_diff : mx.shape[2] + y_diff] += \
                    mx[channel_offset : channel_offset + self._num_channels][:][:]
                
                channel_offset += self._num_channels

        if self._zero_padding[0] > 0 or self._zero_padding[1] > 0:
            without_padding = np.empty((self._num_channels, self._input_size[0], self._input_size[1]))
            without_padding =  \
                result_mx[:, self._zero_padding[0] : self._input_size[0] + self._zero_padding[0], 
                            self._zero_padding[1] : self._input_size[1] + self._zero_padding[1]]
            result_mx = without_padding

        result_mx = np.reshape(result_mx, (self._num_channels, self._input_size[0] * self._input_size[1]))
        return result_mx


    def calculate_B(self, U_upscaled):
        # B = k'(Z^T E(input) S^-1) * (A U P^T)
        # TODO: add pooling
        return self._dp_kernel.deriv(self._Z_T__E_input__S_n1) * np.matmul(self._A, U_upscaled)


    def calculate_C(self, U, output_after_pooling):
        # C = A^1/2 output U^T A^3/2
        return np.matmul( np.matmul(self._A_1_2, output_after_pooling), np.matmul(U.transpose(), self._A_3_2) )


    def g(self, B, C):
        # g(U) = E(input) B^T - 1/2 Z (k'(Z^T Z) * (C + C^T))

        # E(input) B^T
        E_input__B_T = np.matmul(self._E_input, B.transpose())

        # k'(Z^T Z) * (C + C^T)
        k_d_Z_T__Z__mul__C_plus_C_T = self._dp_kernel.deriv(self._Z_T__Z) * (C + C.transpose())

        # E(input) B^T - 1/2 Z (k'(Z^T Z) * (C + C^T))
        g_U = E_input__B_T - 1/2 * np.matmul(self.filter_matrix, k_d_Z_T__Z__mul__C_plus_C_T)

        return g_U


    def h(self, U_upscaled, B):
        # U_upscaled = U P^T
        # X = S^-2 * (M^T U P^T - E(input)^T Z B))    (X is a diagonal matrix)
        # h(U) = E_adj( Z B + E(input) X )

        # Z B
        Z_B = np.matmul(self.filter_matrix, B)

        # M^T U P^T         (diagonal elements)
        M_T__U__P_T__diag = np.einsum('ij,ji->i', self._last_output.transpose(), U_upscaled)

        # E(input)^T Z B    (diagonal elements)
        E_input_T__Z__B__diag = np.einsum('ij,ji->i', self._E_input.transpose(), Z_B)

        # X = S^-2 * (M^T U P^T - E(input)^T Z B))      (diagonal elements)
        X_diag = self._S_n1_diag * self._S_n1_diag * (M_T__U__P_T__diag - E_input_T__Z__B__diag)
        
        # h(U) = E_adj( Z B + E(input) X )
        h_U = self.extract_patches_adj(Z_B + self._E_input * X_diag)

        return h_U


    def forward(self, input):
        # output = A k(Z^T E(input) S^-1) S P
        self._last_input = input

        # E(input)
        self._E_input = self.extract_patches(input)

        # S (diagonal elements)
        self._S_diag = np.linalg.norm(self._E_input, axis = 0)
        # no 0 values
        self._S_diag += np.full(len(self._S_diag), 0.00001)

        # S^-1 (diagonal elements)
        self._S_n1_diag = 1 / self._S_diag

        # Z^T E(input) S^-1
        self._Z_T__E_input__S_n1 = np.matmul(self._filter_matrix.transpose(), self._E_input * self._S_n1_diag)

        # k(Z^T E(input) S^-1)
        kerneled = self._dp_kernel.func(self._Z_T__E_input__S_n1)

        # A k(Z^T E(input) S^-1) S = M
        self._last_output = np.matmul(self._A, kerneled) * self._S_diag

        # A k(Z^T E(input) S^-1) S P
        #self._last_output = self.avg_pooling(self._before_pooling)

        return self._last_output

    