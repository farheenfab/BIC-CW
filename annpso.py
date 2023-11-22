import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

class Activation:
    def __init__(self, activation):
        self.activation = activation
        
    def evaluate(self, x):
        if self.activation == "Logistic":
            return Sigmoid.evaluate(x)
        elif self.activation == "Hyperbolic tangent":
            return tanh.evaluate(x)
        elif self.activation == "ReLU":
            return ReLU.evaluate(x)

    def derivate(self, x):
        if self.activation == "Logistic":
            return Sigmoid.derivative(x)
        elif self.activation == "Hyperbolic tangent":
            return tanh.derivative(x)
        elif self.activation == "ReLU":
            return ReLU.derivative(x)

# sigmoid activation – sub class of activation
class Sigmoid (Activation):
    def evaluate(x):
        return 1 / (1 + np.exp(-x))
    
    # derivative of f = sigma * (1- sigma)
    def derivative(x):
        f = 1 / (1 + np.exp(-x))
        return f * (1 - f)

# tanh activation – sub class of activation
class tanh(Activation):
    def evaluate(x):
        return np.tanh(x)

    # derivative of tanh(x) = sech^2 (x) = 1- tanh^2 (x)
    def derivative(x):
        f = np.tanh(x)
        return 1 - f**2

# ReLU activation – sub class of activation
class ReLU(Activation):
    def evaluate(x):
        return np.maximum(0, x)

    # derivative of max(0, x) = 0 if x < 0, 1 if x > 0 because slope is uniform which is the derivative of the ReLU Graph(1).
    def derivative(x):
        return np.where(x > 0, 1, 0)

# abstract Loss class
# provides evaluate and derivative methods class Loss:
class Loss:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y # actual y labels
        self.t = t # predicted y labels

    def evaluate(self): 
        if self.x == "MSE":
            return MSE.evaluate(self.y,self.t)
        elif self.x == "Binary Cross Entropy":
            return Binary_cross_entropy.evaluate(self.y,self.t)
        elif self.x == "Hinge":
            return Hinge.evaluate(self.y,self.t)

    def derivate(self): 
        if self.x == "MSE":
            return MSE.derivative(self.y,self.t)
        elif self.x == "Binary Cross Entropy":
            return Binary_cross_entropy.derivative(self.y,self.t)
        elif self.x == "Hinge":
            return Hinge.derivative(self.y,self.t)
        
# MSE (Mean Squared Error) Loss – subclass of Loss
class MSE(Loss):
    def evaluate(y, t):
        return (1/2)*((t-y)**2) #1/2*(t-y)**2
    
    def derivative(y, t): 
        return t-y

#Binary cross entropy Loss – subclass of Loss 
class Binary_cross_entropy(Loss):
    def evaluate(y,t):
        y_pred = np.clip(y, 1e-7, 1 - 1e-7)
        term0 = (1-t) * np.log(1- y_pred + 1e-7) 
        term1 = t * np.log(y_pred + 1e-7) 
        return -(term0 + term1)
    
    def derivative(y,t):
        return t/y + (1-t) /(1-y)
    
#Hinge Loss – subclass of Loss 
class Hinge(Loss):
    def evaluate(y,t):
        return max(0, 1-t*y)

    def derivative(y,t): 
        if 1 - t * y > 0:
            return -t
        else:
            return 0

class Layer:
    def __init__(self, nb_inputs, nb_nodes, activation):
        self.nb_nodes = nb_nodes
        size = (nb_nodes, nb_inputs)
        self.activation = activation
        # generating random weights for each neuron in the layer
        self.W = np.random.uniform(-1, 1, size)
        # generating random biases for each neuron in the layer
        self.B = np.random.uniform(-1, 1, nb_nodes)

    def forward(self, fin):
        fin_transposed = fin.T if fin.ndim > 1 else fin
        active = Activation(self.activation)
        B_reshaped = self.B.reshape(-1, 1)
        out = active.evaluate(np.dot(self.W, fin_transposed) + B_reshaped)
        return out.T
    
    def update(self, g, lamda):
        self.W = self.W - lamda * g[0]
        self.B = self.B - lamda * g[1]
        """
        Update weights and biases using gradient information and learning rate.

        Parameters:
        g (numpy.ndarray): Gradient of the loss with respect to the layer's output.
        learning_rate (float): Learning rate for the update.

        Returns:
        None
        """
        
#Network class encapsulates the list of layers and provides forward method
class Network:
    def __init__(self): #initialise the empty list of layers 
        self.layers = []
    
    def append(self, layer): #to append a layer to the network 
        self.layers.append(layer)

    # Forward pass through layers of the neural network, assumes input data has already been passed into the network and 
    # continues forward pass from last layer's output.
    # Used internally within network to pass data through the layers
    def forward(self, data_in): 
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def get_parameters(self):
        # Extract all weights and biases, flatten them, and concatenate into a single vector
        params = []
        for layer in self.layers:
            params.append(layer.W.flatten())
            params.append(layer.B.flatten())
        return np.concatenate(params)
    
    def set_parameters(self, flat_parameters):
        # Restore weights and biases from the flat vector
        pointer = 0  # This pointer will keep track of where we are in the flat_parameters vector
        for layer in self.layers:
            weight_shape = layer.W.shape
            bias_shape = layer.B.shape
            
            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)
            
            layer.W = flat_parameters[pointer:pointer + weight_size].reshape(weight_shape)
            pointer += weight_size
            
            layer.B = flat_parameters[pointer:pointer + bias_size].reshape(bias_shape)
            pointer += bias_size

#Create a network using the parameters provides by the user 
class ANNBuilder:
    def build(nb_layers, list_nb_nodes, list_functions): 
        ann = Network()
        for i in range(nb_layers):
            if i == 0:
                layer = Layer(4, list_nb_nodes[i], list_functions[i])
            else:
                layer = Layer(list_nb_nodes[i-1], list_nb_nodes[i], list_functions[i]) 
            ann.append(layer)
        return ann

class Particle:
    def __init__(self, layers, nodes, functions, loss_func):
        self.ann = ANNBuilder.build(layers, nodes, functions)
        self.fitness = -float('inf')  # Initialize with negative infinity
        self.best_fitness = -float('inf')  # Initialize with negative infinity
        self.best_ann_params = self.ann.get_parameters()
        self.v = np.zeros_like(self.best_ann_params)  # Initialize velocity as zeros
        self.informants = []  # List of other particles that are informants
        self.best_informant_params = self.best_ann_params
        self.loss_function = loss_func

    def evaluate_fitness(self, X, y):
            # Evaluate fitness using the loss class
            y_pred = []
            for sample in X:
                # Assuming the ANN output is a single value between 0 and 1
                output = self.ann.forward(sample)
                prediction = 1 if output.flatten()[-1] >= 0.5 else 0
                y_pred.append(prediction)
            
            # y_pred = y_pred.reshape(-1, 1) if y_pred.ndim > 1 else y_pred
            l = Loss(self.loss_function, y, y_pred)
            loss = l.evaluate()
            loss = np.mean(loss) if isinstance(loss, np.ndarray) else loss
            print("Loss = ",loss)
            fitness = 1 - loss  # Assuming accuracy is inversely proportional to the loss
            return fitness

def PSO(X, y, layers, nodes, functions, epochs=50, pop_size=100, alpha_w=0.5, beta=2, gamma=2, delta=2, err_crit=0.00001, num_informants=3, loss_func="MSE", progress_callback=None):
    
    def evaluate_ann(ann, X, y):
        # Assuming X is the feature matrix and y is the target vector (0 or 1)
        y_pred = []
        for sample in X:
            # Assuming the ANN output is a single value between 0 and 1
            output = ann.forward(sample)
            prediction = 1 if output.flatten()[-1] >= 0.5 else 0
            y_pred.append(prediction)

        y_pred = np.array(y_pred)
        accuracy = np.sum(np.equal(y, y_pred)) / len(y)
        return accuracy
    
    particles = [Particle(layers, nodes, functions, loss_func) for _ in range(pop_size)]

    for p in particles:
        p.informants = np.random.choice([particle for particle in particles if particle != p], size=num_informants, replace=False)

    # Initialize global best as the first particle's position
    gbest_fitness = -float('inf')
    gbest_params = particles[0].best_ann_params

    # progress_bar = tqdm(total=epochs, desc='Epochs', position=0, leave=True)
    for epoch in range(epochs):
        for p in particles:
            # Evaluate current fitness
            accuracy = evaluate_ann(p.ann, X, y)
            fitness = p.evaluate_fitness(X, y)
            
            # Update personal best if the current fitness is better
            if fitness > p.best_fitness:
                p.best_fitness = fitness
                p.best_ann_params = p.ann.get_parameters()

            # Update the best position found by informants
            for informant in p.informants:
                if informant.best_fitness > p.best_fitness:
                    p.best_informant_params = informant.best_ann_params
            
            # Update global best if the current fitness is better
            if fitness > gbest_fitness:
                gbest_fitness = fitness
                best_acc = accuracy
                gbest_params = p.ann.get_parameters()
            
            # Update velocity and position
            r1 = np.random.rand(p.v.shape[0])
            r2 = np.random.rand(p.v.shape[0])
            r3 = np.random.rand(p.v.shape[0])
            p.v = alpha_w * p.v + beta * r1 * (p.best_ann_params - p.ann.get_parameters()) + gamma * r2 * (p.best_informant_params - p.ann.get_parameters()) + delta * r3 * (gbest_params - p.ann.get_parameters())
            
            # Update position with new velocity
            new_position = p.ann.get_parameters() + p.v
            p.ann.set_parameters(new_position)
        # Check for early stopping
        if gbest_fitness >= 1 - err_crit:
            print("Stopping early due to meeting fitness criterion")
            break
        
        if progress_callback is not None:
            progress_callback(epoch + 1)

        # progress_bar.update(1)
        # print("Epoch = ", epoch+1)
    # progress_bar.close()
    print('\nParticle Swarm Optimisation finished')
    print('Best fitness achieved:', gbest_fitness)
    print('Best accuracy achieved:', best_acc)
    return gbest_fitness, best_acc, gbest_params # Return the best parameters found

# Load the dataset
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target

f = "data_banknote_authentication.txt"
data = pd.read_csv(f, header=None)

# Extract features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Set hyper-parameters
# layers = 3
# nodes = [3, 3, 5, 1]
# functions = ["Logistic", "Hyperbolic tangent", "ReLU", "Logistic"]
# epochs = 50

# Run PSO with specified epochs and learning rate
# best_params = PSO(X, y, layers, nodes, functions, epochs)
# print(best_params)

def set_up():
    # Getting number of hidden layers
    n_layers = int(input("Enter number of hidden layers: "))

    # Getting number of nodes per layer
    n_nodes = []
    for i in range(n_layers):
        node = int(input(f"Enter number of nodes for Layer {i+1}: "))    
        n_nodes.append(node)
    output_node = int(input("Enter number of nodes for Output Layer: "))
    n_nodes.append(output_node)

    # Getting Activation Functions
    functions = []
    print(f"Select one of the below listed activation functions for each of the layers: ")
    print("1 - Logistic")
    print("2 - Hyperbolic tangent")
    print("3 - ReLU")
    for i in range(len(n_nodes)):
        if i != len(n_nodes)-1:
            active_func = int(input(f"Enter option for Layer {i+1}: "))
            if active_func == 1:
                functions.append("Logistic")
            elif active_func == 2:
                functions.append("Hyperbolic tangent")
            elif active_func == 3:
                functions.append("ReLU")
            else:
                while active_func not in (1, 2, 3):
                    print("Wrong option")
                    active_func = int(input(f"Enter option for Layer {i+1}: "))
                    if active_func == 1:
                        functions.append("Logistic")
                    elif active_func == 2:
                        functions.append("Hyperbolic tangent")
                    elif active_func == 3:
                        functions.append("ReLU")
        else:
            active_func = int(input(f"Enter option for Output Layer {i+1}: "))
            if active_func == 1:
                functions.append("Logistic")
            elif active_func == 2:
                functions.append("Hyperbolic tangent")
            elif active_func == 3:
                functions.append("ReLU")
            else:
                while active_func not in (1, 2, 3):
                    print("Wrong option")
                    active_func = int(input(f"Enter option for Layer {i+1}: "))
                    if active_func == 1:
                        functions.append("Logistic")
                    elif active_func == 2:
                        functions.append("Hyperbolic tangent")
                    elif active_func == 3:
                        functions.append("ReLU")

    epochs_input = input("Enter number of epochs (default = 50): ")
    epochs = int(epochs_input) if epochs_input else 50

    pop_in = input("Enter population size (default = 100): ")
    pop_size = int(pop_in) if pop_in else 100

    alpha_w_in = input("Enter alpha value (default = 0.5): ")
    alpha_w = int(alpha_w_in) if alpha_w_in else 0.5

    beta_in = input("Enter beta value (default = 2): ")
    beta = int(beta_in) if beta_in else 2

    gamma_in = input("Enter gamma value (default = 2): ")
    gamma = int(gamma_in) if gamma_in else 2

    delta_in = input("Enter delta value (default = 2): ")
    delta = int(delta_in) if delta_in else 2

    err_crit_in = input("Enter error criterion (default = 0.00001): ")
    err_crit = float(err_crit_in) if err_crit_in else 0.00001

    num_informants_in = input("Enter the number of informants (default = 3): ")
    num_informants = int(num_informants_in) if num_informants_in else 3

    print(f"Select one of the below listed Loss Functions: ")
    print("1 - MSE")
    print("2 - Binary Cross Entropy")
    print("3 - Hinge")
    loss_func_in = input(f"Enter option for Loss Function (default = MSE): ")
    loss_func_in = int(loss_func_in) if loss_func_in else 1
    loss_func = "MSE"
    if loss_func_in == 1:
        loss_func = "MSE"
    elif loss_func_in == 2:
        loss_func = "Binary Cross Entropy"
    elif loss_func_in == 3:
        loss_func = "Hinge"
    else:
        while loss_func_in not in (1, 2, 3):
            print("Wrong option")
            loss_func_in = int(input(f"Enter option for Loss Function (default = MSE): "))
            loss_func = "MSE"
            if loss_func_in == 1:
                loss_func = "MSE"
            elif loss_func_in == 2:
                loss_func = "Binary Cross Entropy"
            elif loss_func_in == 3:
                loss_func = "Hinge"

    return(n_layers+1, n_nodes, functions, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_func)
 
layers, nodes, functions, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_func = set_up()
best_params, fitness, accuracy = PSO(X, y, layers, nodes, functions, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_func)
# print(best_params)
# weights = []
# bias = []
# for i in range(0, len(best_params), 2):
#     weights.append(best_params[i])

# for i in range(1, len(best_params), 2):
#     bias.append(best_params[i])

# print("The best parameters: ", best_params)
# print("The best parameters have been split into weights and biases below:")
# print("Weights: \n", weights)
# print("Bias: \n", bias)

# def format_best_params(best_params, nodes):
#     formatted_params = {}
#     pointer = 0

#     # Iterate through layers
#     for i in range(len(nodes) - 1):
#         # Number of nodes in current and next layer
#         current_layer_nodes = nodes[i]
#         next_layer_nodes = nodes[i+1]

#         # Calculate the size of weights and biases for current layer
#         weight_size = current_layer_nodes * next_layer_nodes
#         bias_size = next_layer_nodes

#         # Extract weights
#         weights = best_params[pointer:pointer + weight_size].reshape(next_layer_nodes, current_layer_nodes)
#         pointer += weight_size

#         # Extract biases
#         biases = best_params[pointer:pointer + bias_size]
#         pointer += bias_size

#         # Store in a dictionary
#         formatted_params[f'Layer {i+1}'] = {'Weights': weights, 'Biases': biases}

#     return formatted_params

# # Example usage
# formatted_best_params = format_best_params(best_params, nodes)
# for layer, params in formatted_best_params.items():
#     print(f"{layer} Parameters:")
#     print("Weights:\n", params['Weights'])
#     print("Biases:\n", params['Biases'], "\n")


# print(best_params)

# run experiment
#loss, accuracy = mini_batch(ann, data, classes, epochs, learning_rate, loss_func, batch_size) 

# plot, display results
# plt.plot(range(epochs), loss, label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()