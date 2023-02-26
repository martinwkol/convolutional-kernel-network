import numpy as np
import layer as lr
from internal_filter_layer import IntFilterLayer
from internal_pooling_layer import IntPoolingLayer

class Network:
    @staticmethod
    def create_random(input_size, in_channels, layers, output_nodes):
        int_layers = []
        for layer in layers:
            new_layer = layer.build(input_size, in_channels)
            input_size = new_layer.output_size
            in_channels = new_layer.out_channels

            int_layers.append(new_layer)
        
        output_weights = np.random.normal(0, 1 / np.sqrt(in_channels * input_size[0] * input_size[1]), 
                                        size=(output_nodes, in_channels, input_size[0] * input_size[1]))
        
        return Network(int_layers, output_weights)


    def __init__(self, int_layers, output_weights):
        self._layers = int_layers
        self.output_weights = output_weights

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
        loss_func_gradient = loss_func.gradient(self._last_output, expected_output)
        output_weights_grad = np.einsum('i,jk->ijk', loss_func_gradient, self._layers[len(self._layers) - 1].last_output)

        layer_gradients = [None] * len(self._layers)
        # TODO: find an even better variable name
        grad_after_pooling = np.einsum('k,kij->ij', loss_func_gradient, self.output_weights)
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