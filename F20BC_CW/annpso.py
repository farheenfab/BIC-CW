import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

# Activation class - checks the activation function passed and then uses the evaluate method of the respective subclass function
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

    # Not used as backpropagation is not implemented
    def derivate(self, x):
        if self.activation == "Logistic":
            return Sigmoid.derivative(x)
        elif self.activation == "Hyperbolic tangent":
            return tanh.derivative(x)
        elif self.activation == "ReLU":
            return ReLU.derivative(x)

# Sigmoid Activation Function – sub class of Activation
class Sigmoid (Activation):
    def evaluate(x):
        return 1 / (1 + np.exp(-x))
    
    # Not used as backpropagation is not implemented
    def derivative(x):
        f = 1 / (1 + np.exp(-x))
        return f * (1 - f)

# tanh Activation Function – sub class of Activation
class tanh(Activation):
    def evaluate(x):
        return np.tanh(x)

    # Not used as backpropagation is not implemented
    def derivative(x):
        f = np.tanh(x)
        return 1 - f**2

# ReLU Activation Function – sub class of Activation
class ReLU(Activation):
    def evaluate(x):
        return np.maximum(0, x)
    
    # Not used as backpropagation is not implemented
    def derivative(x):
        return np.where(x > 0, 1, 0)

# Loss class - checks the activation function passed then uses the evaluate method of the respective subclass function
class Loss:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y # Actual y labels
        self.t = t # Predicted y labels

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
        return (1/2)*((t-y)**2)
    
    def derivative(y, t): 
        return t-y

# Binary Cross Entropy Loss – subclass of Loss 
class Binary_cross_entropy(Loss):
    def evaluate(y,t):
        y_pred = np.clip(y, 1e-7, 1 - 1e-7)
        term0 = (1-t) * np.log(1- y_pred + 1e-7) 
        term1 = t * np.log(y_pred + 1e-7) 
        return -(term0 + term1)
    
    def derivative(y,t):
        return t/y + (1-t) /(1-y)
    
# Hinge Loss – subclass of Loss 
class Hinge(Loss):
    def evaluate(y,t):
        return np.maximum(0, 1-t*y)

    def derivative(y,t): 
        if 1 - t * y > 0:
            return -t
        else:
            return 0

# Layer class - used to create a layer that will be used in the Network class
class Layer:
    # Takes input parameters and initializes them: 
    # > number of input nodes (nb_inputs)
    # > number of current layer nodes (nb_nodes)
    # > activation function
    def __init__(self, nb_inputs, nb_nodes, activation):
        self.nb_nodes = nb_nodes
        self.activation = activation
        # generating random weights for each neuron/node in the layer
        size = (nb_nodes, nb_inputs)
        self.W = np.random.uniform(-1, 1, size)
        # generating random biases for each neuron/node in the layer
        self.B = np.random.uniform(-1, 1, nb_nodes)

    # forward() takes data as input and moves one layer forward through the network using the weights, biases and activation function
    def forward(self, fin):
        fin_transposed = fin.T if fin.ndim > 1 else fin
        active = Activation(self.activation)
        B_reshaped = self.B.reshape(-1, 1)
        out = active.evaluate(np.dot(self.W, fin_transposed) + B_reshaped)
        return out.T
    
    # update() updates the weights and biases using gradient information and learning rate
    # As gradient descent was not used, update() was also not used anywhere
    def update(self, g, lamda):
        self.W = self.W - lamda * g[0]
        self.B = self.B - lamda * g[1]
        
# Network class - 
# > generates and initializes a list of layers
# > provides append method to append a layer into the layers list of the Network
# > provides forward method to move forward through the whole network
# > provides get_parameters method that extracts all weights and biases, flattens them and concatenates into a single vector
# > provides set_parameters method that restores weights and biases from the flat vector
class Network:
    def __init__(self): 
        self.layers = []
    
    def append(self, layer): 
        self.layers.append(layer)

    # Forward pass through layers of the neural network, assumes input data has already been passed into the network and 
    # continues forward pass from last layer's output.
    def forward(self, data_in): 
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    # Extracts all weights and biases, flattens them, and concatenates into a single vector
    # Reference : ChatGPT
    def get_parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.W.flatten())
            params.append(layer.B.flatten())
        return np.concatenate(params)
    
    # Restores the weights and biases from the flat vector
    # Reference : ChatGPT
    def set_parameters(self, flat_parameters):
        # Pointer keeps track of where we are in the flat_parameters vector
        pointer = 0  
        for layer in self.layers:
            weight_shape = layer.W.shape
            bias_shape = layer.B.shape
            
            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)
            
            layer.W = flat_parameters[pointer:pointer + weight_size].reshape(weight_shape)
            pointer += weight_size
            
            layer.B = flat_parameters[pointer:pointer + bias_size].reshape(bias_shape)
            pointer += bias_size

# ANNBuilder class - Creates a network using the parameters provides by the user 
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

# Particle class - Creates a particle object with the different informations needed including the informants information
class Particle:
    def __init__(self, layers, nodes, functions, loss_func):
        self.ann = ANNBuilder.build(layers, nodes, functions)       # Each particle has an ANN
        self.fitness = -float('inf')                                # Initialize with negative infinity
        self.best_fitness = -float('inf')                           # Initialize with negative infinity
        self.best_ann_params = self.ann.get_parameters()            # Stores best parameters captured by the ann
        self.v = np.zeros_like(self.best_ann_params)                # Initialize velocity as zeros
        self.informants = []                                        # List that stores other particles that are informants
        self.best_informant_params = self.best_ann_params           # Stores the best parameters captured by informants
        self.loss_function = loss_func                              # Stores loss function given by the user

    # Evaluates fitness of the particle using the loss function
    def evaluate_fitness(self, X, y):
            # If loss function is MSE or Hinge, for each sample we move forward through the ANN and the get the y_pred
            if self.loss_function == "MSE" or self.loss_function == "Hinge":
                y_pred = []
                for sample in X:
                    output = self.ann.forward(sample)
                    prediction = 1 if output.flatten()[-1] >= 0.5 else 0
                    y_pred.append(prediction)
            # Else, we move forward through the ANN using the whole X dataset and reshape the array if the dimensions of y_pred are greater than 1
            else:
                y_pred = self.ann.forward(X)
                y_pred = y_pred.reshape(-1, 1) if y_pred.ndim > 1 else y_pred
            # We then calculate the loss and then get the fitness but subtracting 1 - loss
            l = Loss(self.loss_function, y, y_pred)
            loss = l.evaluate()
            loss = np.mean(loss) if isinstance(loss, np.ndarray) else loss
            fitness = 1 - loss
            return fitness, loss

# Evalutes ANN and calculates the accuracy of the ANN
def evaluate_ann(ann, X, y):
        y_pred = []
        for sample in X:
            output = ann.forward(sample)
            prediction = 1 if output.flatten()[-1] >= 0.5 else 0
            y_pred.append(prediction)

        y_pred = np.array(y_pred)
        accuracy = np.sum(np.equal(y, y_pred)) / len(y)
        return accuracy

# PSO method - Implements the Particle Swarm Optimization algorithm using ANN (It considers each particle as an ANN)
# Takes input: X data, y label data, list of number of nodes, list of activation function
# for each layer, epochs, population size, alpha weight, beta, gamma, delta for updating 
# velocity, error criterion, number of informants, loss function, progess_callback used for the progress bar on the GUI
def PSO(X, y, layers, nodes, functions, epochs=50, pop_size=100, alpha_w=0.5, beta=2, gamma=2, delta=2, err_crit=0.00001, num_informants=3, loss_func="MSE", progress_callback=None):
    
    # Creates a Particle class object based on the number of population size and puts all the particles in a list
    particles = [Particle(layers, nodes, functions, loss_func) for _ in range(pop_size)]

    # We go through all the particles and choose random particles as informants (except the current particle) based on the number of informants specified
    for p in particles:
        p.informants = np.random.choice([particle for particle in particles if particle != p], size=num_informants, replace=False)

    # Initialize global best as the first particle's position
    gbest_fitness = -float('inf')
    gbest_params = particles[0].best_ann_params

    loss_per_epochs = []
    fitness_per_epoch = []

    # We run a for loop for each epoch
    for epoch in range(epochs):
        # Stores all the losses calculated per epoch temporarily
        temp_loss = []
        for p in particles:
            # Evaluate accuracy of the ann and the fitness of the particle
            accuracy = evaluate_ann(p.ann, X, y)
            fitness, loss = p.evaluate_fitness(X, y)
            temp_loss.append(loss)
            
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
            
            # Update velocity using the equation given in Course PPT
            r1 = np.random.rand(p.v.shape[0])
            r2 = np.random.rand(p.v.shape[0])
            r3 = np.random.rand(p.v.shape[0])
            p.v = alpha_w * p.v + beta * r1 * (p.best_ann_params - p.ann.get_parameters()) + gamma * r2 * (p.best_informant_params - p.ann.get_parameters()) + delta * r3 * (gbest_params - p.ann.get_parameters())
            
            # Update position with new velocity
            new_position = p.ann.get_parameters() + p.v
            p.ann.set_parameters(new_position)
        
        # We calculate the mean of temp_loss and append the value to loss_per_epochs
        loss_per_epochs.append(np.mean(temp_loss))
        # We append global the best fitness into fitness_per_epoch
        fitness_per_epoch.append(gbest_fitness)

        # Check for early stopping and stop if the fitness criterion is met
        if gbest_fitness >= 1 - err_crit:
            print("Stopping early due to meeting fitness criterion")
            break
        
        # Used for the progress bar in the GUI
        if progress_callback is not None:
            progress_callback(epoch + 1)

    print('\nParticle Swarm Optimisation finished')
    print('Best fitness achieved:', gbest_fitness)
    print('Best accuracy achieved:', best_acc)
    return gbest_fitness, best_acc, gbest_params, loss_per_epochs, fitness_per_epoch 

# Set up for running using terminal
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

    # Getting Activation Functions for each layer
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

    # Getting number of epochs
    epochs_input = input("Enter number of epochs (default = 50): ")
    epochs = int(epochs_input) if epochs_input else 50

    # Getting population size
    pop_in = input("Enter population size (default = 100): ")
    pop_size = int(pop_in) if pop_in else 100

    # Getting alpha weight value to be used in the update velocity equation
    alpha_w_in = input("Enter alpha value (default = 0.5): ")
    alpha_w = int(alpha_w_in) if alpha_w_in else 0.5

    # Getting beta value to be used in the update velocity equation
    beta_in = input("Enter beta value (default = 2): ")
    beta = int(beta_in) if beta_in else 2

    # Getting gamma value to be used in the update velocity equation
    gamma_in = input("Enter gamma value (default = 2): ")
    gamma = int(gamma_in) if gamma_in else 2

    # Getting delta value to be used in the update velocity equation
    delta_in = input("Enter delta value (default = 2): ")
    delta = int(delta_in) if delta_in else 2

    # Getting error criterion
    err_crit_in = input("Enter error criterion (default = 0.00001): ")
    err_crit = float(err_crit_in) if err_crit_in else 0.00001

    # Getting number of informants
    num_informants_in = input("Enter the number of informants (default = 3): ")
    num_informants = int(num_informants_in) if num_informants_in else 3

    # Getting the loss function
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
    
    # We return n_layers+1 because we also consider the output later
    return(n_layers+1, n_nodes, functions, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_func)

# Please change the location of the file and header value accordingly, if using terminal.
# After you enter the Loss function option, it will take some time based on the dataset. So please wait until it prints the output.
# Uncomment the below lines to run the PSO on Terminal

# f = "data_banknote_authentication.txt" # Plese replace with your file destination, if needed.
# data = pd.read_csv(f, header=None)

# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# layers, nodes, functions, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_func = set_up()
# fitness, accuracy, best_params, loss_per_epoch, fitness_per_epoch = PSO(X, y, layers, nodes, functions, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_func)
# print ("Best parameters = ", best_params)

# plt.plot(range(epochs), loss_per_epoch, label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Loss')
# plt.legend()
# plt.title("Mean Loss per Epoch")
# plt.show()

# plt.plot(range(epochs), fitness_per_epoch, label='Fitness')
# plt.xlabel('Epochs')
# plt.ylabel('Global Fitness')
# plt.legend()
# plt.title("Global Fitness per Epoch")
# plt.show()