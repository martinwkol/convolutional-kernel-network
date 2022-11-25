import numpy as np
import layer as lr
from internal_filter_layer import IntFilterLayer
from internal_pooling_layer import IntPoolingLayer

class Network:
    @staticmethod
    def create_random(input_size, num_channels, layers, output_nodes):
        int_layers = []
        for layer in layers:
            if isinstance(layer, lr.Filter):
                # TODO: check input for errors
                filter_matrix = layer.filter_matrix
                if filter_matrix is None:
                    filter_matrix_size = (num_channels * layer.filter_size[0] * layer.filter_size[1], layer.out_channels)
                    filter_matrix = np.random.rand(filter_matrix_size[0], filter_matrix_size[1])
                    norms = np.linalg.norm(filter_matrix, axis = 0)
                    filter_matrix /= norms

                zero_padding = layer.zero_padding
                if isinstance(zero_padding, str):
                    if layer.zero_padding == 'same':
                        zero_padding = (layer.filter_size[0] // 2, layer.filter_size[1] // 2)
                    elif layer.zero_padding == 'none':
                        zero_padding = (0, 0)

                new_layer = IntFilterLayer(
                    input_size=input_size, 
                    in_channels=num_channels,
                    filter_size=layer.filter_size,
                    filter_matrix=filter_matrix,
                    dp_kernel=layer.dp_kernel,
                    zero_padding=zero_padding
                )
                int_layers.append(new_layer)

                input_size = new_layer.output_size
                num_channels = new_layer.out_channels

            elif isinstance(layer, lr.AvgPooling):
                new_layer = IntPoolingLayer(
                    input_size=input_size,
                    pooling_size=layer.filter_size
                )
                int_layers.append(new_layer)

                input_size = new_layer.output_size

            else:
                raise TypeError("layer must be Filter or AvgPooling, not {}".format(type(layer)))
        
        output_weights = np.random.normal(0, 1 / np.sqrt(num_channels * input_size[0] * input_size[1]), 
                                        size=(output_nodes, num_channels, input_size[0] * input_size[1]))
        
        return Network(int_layers, output_weights)


    def __init__(self, int_layers, output_weights):
        self._layers = int_layers
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