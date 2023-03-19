import numpy as np
from layer_base import LayerBase
from gradient_calculation_info import GradientCalculationInfo

class PoolingLayer(LayerBase):
    def __init__(self, input_size, in_channels, pooling_size):
        super().__init__(
            input_size=input_size, 
            output_size=(input_size[0] // pooling_size[0], input_size[1] // pooling_size[1]), 
            in_channels=in_channels, 
            out_channels=in_channels
        )

        self.pooling_size = pooling_size
        self.last_output = None
    
    def compute_gradient(self, gradient_calculation_info):
        new_info = GradientCalculationInfo(
            last_output_after_pooling=gradient_calculation_info.last_output_after_pooling, 
            U=gradient_calculation_info.U, 
            U_upscaled=self.backward(gradient_calculation_info.U_upscaled), # U P^T
            layer_number=gradient_calculation_info.layer_number - 1
        )

        return 0, new_info

    def gradient_descent(self, descent):
        pass

    def forward(self, input):
        self.last_output = self._avg_pooling(input)
        return self.last_output

    def backward(self, U):
        return self._avg_pooling_t(U)

    def _avg_pooling(self, U):
        assert U.shape[1] == self.input_size[0] * self.input_size[1]
        
        # Reshape U to a 3D tensor
        U_3d = U.reshape(self.out_channels, self.input_size[0], self.input_size[1])

        # Compute the strides of the tensor U_3d
        stride_channels = U_3d.strides[0]
        stride_x = U_3d.strides[1]
        stride_y = U_3d.strides[2]
        
        # Create a view of U_3d with the specified shape and strides
        pooling_view = np.lib.stride_tricks.as_strided(
            U_3d, 
            shape=(
                self.out_channels, 
                self.output_size[0], 
                self.output_size[1], 
                self.pooling_size[0], 
                self.pooling_size[1]
            ),
            strides=(
                stride_channels, 
                stride_x * self.pooling_size[0], 
                stride_y * self.pooling_size[1], 
                stride_x, 
                stride_y
            )
        )

        # Compute the average pooling over the last two dimensions of the view
        pooled = pooling_view.sum(axis=(3, 4)) / (self.pooling_size[0] * self.pooling_size[1])
        
        # Reshape pooled to a 2D tensor
        pooled = pooled.reshape(self.out_channels, -1)
        
        return pooled

    def _avg_pooling_t(self, U):
        assert U.shape[1] == self.output_size[0] * self.output_size[1]

        # Reshape U into a 3D tensor
        U_3d = U.reshape(-1, self.output_size[0], self.output_size[1])

        # Upsample the 3D tensor by repeating values along the pooling dimensions
        upscaled = np.repeat(np.repeat(U_3d, self.pooling_size[0], axis=1), self.pooling_size[1], axis=2)

        # Pad the tensor with zeros if the input dimensions are not multiples of the pooling size
        missing_x = self.input_size[0] - upscaled.shape[1]
        missing_y = self.input_size[1] - upscaled.shape[2]
        if missing_x > 0 or missing_y > 0:
            upscaled = np.pad(upscaled, ((0, 0), (0, missing_x), (0, missing_y)))

        # Normalize the values by the pooling size and reshape the tensor
        upscaled /= self.pooling_size[0] * self.pooling_size[1]
        upscaled = upscaled.reshape(-1, self.input_size[0] * self.input_size[1])

        # Return the upscaled tensor
        return upscaled