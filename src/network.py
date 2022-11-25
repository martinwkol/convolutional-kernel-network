import numpy as np
from internal_filter_layer import IntFilterLayer
from internal_pooling_layer import IntPoolingLayer

class Network:
    def __init__(self, layers, output_weights):
        self._layers = layers
        self._output_weights = output_weights

        self._last_output = None
    
    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
            
        self._last_output = np.einsum('jk,ijk->i', x, self._output_weights)
        return self._last_output

    def gradients(self, loss_func, expected_output):
        loss_func_gradient = loss_func.gradient(self._last_output, expected_output)
        output_weights_grad = np.einsum('i,jk->ijk', loss_func_gradient, self._layers[len(self._layers) - 1].last_output)

        layer_gradients = [None] * len(self._layers)
        # TODO: find an even better variable name
        grad_after_pooling = np.einsum('k,kij->ij', loss_func_gradient, self._output_weights)
        grad_upscaled = grad_after_pooling
        output_after_pooling = self._layers[len(self._layers) - 1].last_output
        
        for i in reversed(range(len(self._layers))):
            if isinstance(self._layers[i], IntFilterLayer):
                B = self._layers[i].calculate_B(grad_upscaled)
                C = self._layers[i].calculate_C(grad_after_pooling, output_after_pooling)
                layer_gradients[i] = self._layers[i].g(B, C)

                if i > 0:
                    grad_after_pooling = self._layers[i].h(grad_upscaled, B)
                    grad_upscaled = grad_after_pooling
                    output_after_pooling = self._layers[i - 1].last_output
            
            elif isinstance(self._layers[i], IntPoolingLayer):
                grad_upscaled = self._layers[i].backward(grad_upscaled)
        
        return layer_gradients, output_weights_grad