import numpy as np
from gradient_calculation_info import GradientCalculationInfo
from layer_base import LayerBase
import pickle

class Network:
    def __init__(self, input_size, in_channels, layer_infos, output_nodes, output_weights = None):
        self.layers = []

        # Build the layers sequentially
        for layer_info in layer_infos:
            layer = layer_info.build(input_size, in_channels)
            self.layers.append(layer)
            input_size = layer.output_size
            in_channels = layer.out_channels

        # Initialize the output weights if not given
        if output_weights is None:
            mu, sigma = 0, 1 / np.sqrt(in_channels * input_size[0] * input_size[1])
            output_weights = np.random.normal(mu, sigma, (output_nodes, in_channels, input_size[0] * input_size[1]))

        self.output_weights = output_weights
        self.last_input = None
        self.last_output = None        

    @property
    def input_size(self):
        return self.layers[0].input_size
    
    @property
    def output_size(self):
        return self.output_weights.shape[0]
    
    def forward(self, x):
        self.last_input = x

        for layer in self.layers:
            x = layer.forward(x)

        self.last_output = np.einsum('jk,ijk->i', x, self.output_weights)
        return self.last_output

    def compute_gradients(self, loss_func_gradient):
        """Compute the gradients for all filter layers and the output layer"""

        # Initialize gradients
        num_layers = len(self.layers)
        gradients = [None] * (num_layers + 1)

        # Compute gradient for output_weights
        last_output = self.layers[num_layers - 1].last_output
        gradients[-1] = np.einsum('i,jk->ijk', loss_func_gradient, last_output)

        # Compute gradient for all other layers
        U = np.einsum('k,kij->ij', loss_func_gradient, self.output_weights)
        gci = GradientCalculationInfo(last_output_after_pooling=last_output,
                                      U=U,
                                      U_upscaled=U,
                                      layer_number=num_layers-1)
        for i in reversed(range(len(self.layers))):
            gradients[i], gci = self.layers[i].compute_gradient(gci)
        
        return gradients

    def save_to_file(self, file):
        if isinstance(file, str):
            with open(file, "wb") as f:
                return self.save_to_file(f)
        
        pickle.dump(len(self.layers), file)
        for layer in self.layers:
            layer.save_to_file(file)
        
        pickle.dump(
            (
                self.output_weights,
                self.last_input,
                self.last_output
            ), 
            file
        )

    @staticmethod
    def load_from_file(file):
        if isinstance(file, str):
            with open(file, "rb") as f:
                return Network.load_from_file(f)
        
        layers = []
        num_layers = pickle.load(file)
        for _ in range(num_layers):
            layers.append(LayerBase.load_from_file(file))

        (
            output_weights,
            last_input,
            last_output
        ) = pickle.load(file)

        network = Network.__new__(Network)
        network.layers = layers
        network.output_weights = output_weights
        network.last_input = last_input
        network.last_output = last_output
        return network