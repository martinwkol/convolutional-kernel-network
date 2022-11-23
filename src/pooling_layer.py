import numpy as np

class pooling_layer:
    def __init__(self, input_size, pooling_size):
        self._input_size = input_size
        self._pooling_size = pooling_size
        self._output_size = (input_size[0] // pooling_size[0], input_size[1] // pooling_size[1])

    
    def forward(self, U):
        return self._avg_pooling(U)

    def backward(self, U):
        return self._avg_pooling_t(U)


    def _avg_pooling(self, U):
        assert U.shape[1] == self._input_size[0] * self._input_size[1]
        U_3d = np.reshape(U, (U.shape[0], self._input_size[0], self._input_size[1]))

        pooled_size = (U.shape[0], self._output_size[0], self._output_size[1])
        pooled = np.zeros(pooled_size)

        for x in range(self._pooling_size[0]):
            for y in range(self._pooling_size[1]):
                pooled += U_3d[:,   x : x + pooled_size[1] * self._pooling_size[0] : self._pooling_size[0], 
                                    y : y + pooled_size[2] * self._pooling_size[1] : self._pooling_size[1]]

        pooled /= self._pooling_size[0] * self._pooling_size[1]
        return pooled.reshape((pooled_size[0], pooled_size[1] * pooled_size[2]))


    def _avg_pooling_t(self, U):
        assert U.shape[1] == self._output_size[0] * self._output_size[1]
        U_3d = np.reshape(U, (U.shape[0], self._output_size[0], self._output_size[1]))

        upscaled_size = (U.shape[0], self._input_size[0], self._input_size[1])
        upscaled = np.empty(upscaled_size)

        for x in range(self._pooling_size[0]):
            for y in range(self._pooling_size[1]):
                upscaled[:, x : x + self._output_size[0] * self._pooling_size[0] : self._pooling_size[0], 
                            y : y + self._output_size[1] * self._pooling_size[1] : self._pooling_size[1]] = U_3d
        
        upscaled[:, self._output_size[0] * self._pooling_size[0]::, :] = 0
        upscaled[:, :, self._output_size[1] * self._pooling_size[1]::] = 0

        upscaled /= self._pooling_size[0] * self._pooling_size[1]
        return upscaled.reshape((upscaled_size[0], upscaled_size[1] * upscaled_size[2]))