import numpy as np
import layer as lr
from internal_filter_layer import IntFilterLayer
from internal_pooling_layer import IntPoolingLayer
from gradient_calculation_info import GradientCalculationInfo

class Network:
    def __init__(self, input_size, in_channels, layers, output_nodes, output_weights = None):
        self._layers = []
        for layer in layers:
            new_layer = layer.build(input_size, in_channels)
            input_size = new_layer.output_size
            in_channels = new_layer.out_channels

            self._layers.append(new_layer)

        self.output_weights = output_weights
        if self.output_weights is None:
            mu, sigma = 0, 1 / np.sqrt(in_channels * input_size[0] * input_size[1])
            size = (output_nodes, in_channels, input_size[0] * input_size[1])
            self.output_weights = np.random.normal(mu, sigma, size)

        self._last_output = None

    @property
    def input_size(self):
        return self._layers[0].input_size
    
    @property
    def output_size(self):
        return self.output_weights.shape[0]

    @property
    def layers(self):
        return self._layers
    
    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
            
        self._last_output = np.einsum('jk,ijk->i', x, self.output_weights)
        return self._last_output

    def gradients(self, loss_func, expected_output):
        # layers + one for output_weights
        gradients = [None] * (len(self._layers) + 1)

        loss_func_gradient = loss_func.gradient(self._last_output, expected_output)
        gradients[-1] = np.einsum('i,jk->ijk', loss_func_gradient, self._layers[len(self._layers) - 1].last_output)

        next_U = np.einsum('k,kij->ij', loss_func_gradient, self.output_weights)
        gci = GradientCalculationInfo(
            next_filter_layer_input=self._layers[len(self._layers) - 1].last_output,
            next_U=next_U,
            next_U_upscaled=next_U,
            layer_number=len(self._layers)-1
        )
        
        for i in reversed(range(len(self._layers))):
            gradients[i], gci = self._layers[i].compute_gradient(gci)
        
        return gradients