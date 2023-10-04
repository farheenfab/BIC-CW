import numpy as np

class MultiLayerANN:
    def __init__(self, layer_sizes, activation_functions):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(size, 1) for size in layer_sizes[1:]]

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def forward_propagate(self, x):
        activations = [x]
        weighted_inputs = []

        for i in range(self.num_layers - 1):
            weighted_input = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            weighted_inputs.append(weighted_input)

            if self.activation_functions[i] == 'logistic':
                activation = self.sigmoid(weighted_input)
            elif self.activation_functions[i] == 'relu':
                activation = self.relu(weighted_input)
            elif self.activation_functions[i] == 'tanh':
                activation = self.tanh(weighted_input)

            activations.append(activation)

        return activations, weighted_inputs

# Example usage
layer_sizes = [10, 20, 2]  # Configure the number of neurons in each layer
activation_functions = ['logistic', 'relu', 'logistic']  # Specify the activation functions for each layer

# Create the multi-layer ANN
ann = MultiLayerANN(layer_sizes, activation_functions)

# Perform forward propagation for a sample input
sample_input = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])  # Sample input
activations, weighted_inputs = ann.forward_propagate(sample_input)

# Print the activations and weighted inputs for each layer
for i in range(len(activations)):
    print(f"Layer {i + 1} - Activations: {activations[i].flatten()}")
    print(f"Weighted Inputs: {weighted_inputs[i].flatten()}")
