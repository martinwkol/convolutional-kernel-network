import numpy as np

class network:
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
        good_variable_name = np.einsum('k,kij->ij', loss_func_gradient, self._output_weights)
        for i in reversed(range(len(self._layers))):
            B = self._layers[i].calculate_B(good_variable_name)
            C = self._layers[i].calculate_C(good_variable_name)
            layer_gradients[i] = self._layers[i].g(good_variable_name, B, C)
            if i > 0:
                good_variable_name = self._layers[i].h(good_variable_name, B)
        
        return layer_gradients, output_weights_grad