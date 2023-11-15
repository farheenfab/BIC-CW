import numpy as np
import pandas as pd

# abstract activation class
# provide evaluate and derivative methods
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
        self.y = y
        self.t = t

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
l = 0
#Layer class providing forward method 
# class Layer:
#     def __init__(self, nb_inputs, nb_nodes, activation):
#         #declare attributes: nb_nodes, X_in, W, B, activation
#         global l
#         l = l+1
#         print("l = ", l)
#         self.nb_nodes = nb_nodes
#         size = (nb_nodes, nb_inputs)
#         self.activation = activation
#         print(nb_inputs)
#         print(nb_nodes)
#         # generating random weights for each neuron in the layer
#         self.W = np.random.uniform(-1, 1, nb_inputs)
#         #generates an array of specified shape, where each element is drawn from a standard normal distribution (also known as a Gaussian distribution) with a mean of 0 and a standard deviation of 1.
#         # generating random biases for each neuron in the layer
#         self.B = np.random.uniform(-1, 1, nb_inputs)
    
#     def forward(self, fin):
#         self.X_in = fin
#         active = Activation(self.activation)
#         out = active.evaluate(np.dot(self.W, self.X_in) + self.B)
#         return out
    
class Layer:
    def __init__(self, nb_inputs, nb_nodes, activation):
        global l
        l = l + 1
        print("l = ", l)
        self.nb_nodes = nb_nodes
        size = (nb_nodes, nb_inputs)
        self.activation = activation
        print(nb_inputs)
        print(nb_nodes)
        # generating random weights for each neuron in the layer
        self.W = np.random.uniform(-1, 1, size)
        # generating random biases for each neuron in the layer
        self.B = np.random.uniform(-1, 1, nb_nodes)

    def forward(self, fin):
        self.X_in = fin
        active = Activation(self.activation)
        out = active.evaluate(np.dot(self.W, self.X_in) + self.B)
        return out
    
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

# iter_max = 10000
# pop_size = 100
# dimensions = 2
# c1 = 2
# c2 = 2
# err_crit = 0.00001

class Particle:
    def __init__(self, layers, nodes, functions):
        self.ann = ANNBuilder.build(layers, nodes, functions)
        self.fitness = -float('inf')  # Initialize with negative infinity
        self.best_fitness = -float('inf')  # Initialize with negative infinity
        self.best_ann_params = self.ann.get_parameters()
        self.v = np.zeros_like(self.best_ann_params)  # Initialize velocity as zeros
        self.informants = []  # List of other particles that are informants
        self.best_informant_params = self.best_ann_params

def PSO(X, y, layers, nodes, functions, epochs, pop_size=100, c1=2, c2=2, err_crit=0.00001, w=0.5, num_informants=3):
    
    def evaluate_ann(ann, X, y):
        # Assuming X is the feature matrix and y is the target vector (0 or 1)
        y_pred = []
        for sample in X:
            # Assuming the ANN output is a single value between 0 and 1
            output = ann.forward(sample)
            y_pred.append(1 if output[-1] >= 0.5 else 0)

        y_pred = np.array(y_pred)
        accuracy = np.mean(y_pred == y)
        fitness = accuracy  # You can use other metrics as needed
        errorf6 = 1 - fitness
        return fitness, errorf6
    
    particles = [Particle(layers, nodes, functions) for _ in range(pop_size)]

    for p in particles:
        p.informants = np.random.choice([particle for particle in particles if particle != p], size=num_informants, replace=False)

    # Initialize global best as the first particle's position
    gbest_fitness = -float('inf')
    gbest_params = particles[0].best_ann_params
    
    for epoch in range(epochs):
        for p in particles:
            # Evaluate current fitness
            fitness = evaluate_ann(p.ann, X, y)[0]
            
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
                gbest_params = p.ann.get_parameters()
            
            # Update velocity and position
            r1 = np.random.rand(p.v.shape[0])
            r2 = np.random.rand(p.v.shape[0])
            p.v = w * p.v + c1 * r1 * (p.best_ann_params - p.ann.get_parameters()) + c2 * r2 * (p.best_informant_params - p.ann.get_parameters())
            
            # Update position with new velocity
            new_position = p.ann.get_parameters() + p.v
            p.ann.set_parameters(new_position)
            
        # Check for early stopping
        if gbest_fitness >= 1 - err_crit:
            print("Stopping early due to meeting fitness criterion")
            break
        
        print("Epoch = ", epoch+1)
        # Progress bar
        # if i % (iter_max // 10) == 0:
        #     print('.', end='')

    print('\nParticle Swarm Optimisation finished')
    print('Best fitness achieved:', gbest_fitness)
    return gbest_params  # Return the best parameters found

# Load the dataset
url = "data_banknote_authentication.txt"
data = pd.read_csv(url, header=None)

# Extract features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Set hyper-parameters
layers = 3
nodes = [3, 3, 5, 1]
functions = ["Logistic", "Hyperbolic tangent", "ReLU", "Logistic"]
epochs = 50

# Run PSO with specified epochs and learning rate
best_params = PSO(X, y, layers, nodes, functions, epochs)
print(best_params)
# class Particle:
#     def __init__(self, nb_layers, list_nb_nodes, list_functions):
#         self.ann = ANNBuilder.build(nb_layers, list_nb_nodes, list_functions)
#         self.fitness = 0.0
#         self.best_fitness = 0.0
#         self.best_ann = self.ann
#         self.v = 0.0

# #PSO
# def PSO(X, y, layers, nodes, functions, iter_max=10000, pop_size=100, dimensions=2, c1=2, c2=2, err_crit=0.00001, w = 0.5):
#     # class Particle:
#     #     # pass
#     #     def __init__(self, dimensions):
#     #         self.params = np.array([random() for _ in range(dimensions)])
#     #         self.fitness = 0.0
#     #         self.best = self.params
#     #         self.v = np.zeros(dimensions)

#     # class Particle:
#     #     def __init__(self, nb_layers, list_nb_nodes, list_functions):
#     #         self.ann = ANNBuilder.build(nb_layers, list_nb_nodes, list_functions)
#     #         self.fitness = 0.0
#     #         self.best = self.ann
#     #         self.v = 0.0

#     def f6(para):
#         '''Schaffer's F6 function'''
#         num = (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))**2) - 0.5
#         denom = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1])))**2
#         f6 =  0.5 - (num / denom)
#         errorf6 = 1 - f6
#         return f6, errorf6
    
#     def evaluate_ann(ann, X, y):
#         # Assuming X is the feature matrix and y is the target vector (0 or 1)
#         y_pred = []
#         for sample in X:
#             # Assuming the ANN output is a single value between 0 and 1
#             output = ann.forward(sample)
#             y_pred.append(1 if output[-1] >= 0.5 else 0)

#         y_pred = np.array(y_pred)
#         accuracy = np.mean(y_pred == y)
#         fitness = accuracy  # You can use other metrics as needed
#         errorf6 = 1 - fitness
#         return fitness, errorf6

#     particles = [Particle(layers, nodes, functions) for _ in range(pop_size)]

#     for p in particles:
#         p.informants = np.random.choice(particles, size=10, replace=False)

#     gbest = particles[0]
#     err = float('inf')
#     i = 0

#     max_velocity = 0.1 

#     while i < iter_max:
#         for p in particles:
#             fitness, _ = evaluate_ann(p.ann, X, y)
#             if fitness > p.fitness:
#                 p.fitness = fitness
#                 p.best_ann = p.ann
#                 p.best_fitness = fitness

#             if fitness > gbest.fitness:
#                 gbest = p
            
#             r1 = np.random.rand(dimensions)
#             r2 = np.random.rand(dimensions)

#             current_position = p.ann.get_parameters()
#             p_best_position = p.best_ann.get_parameters()
#             g_best_position = gbest.best_ann.get_parameters()
#             # Calculate particle's new velocity
#             v = w * v + c1 * r1 * (p_best_position - current_position) + c2 * r2 * (g_best_position - current_position)
#             # v = p.v + c1 * random() * (p.best_ann - p.ann) + c2 * random() * (gbest.best_ann - p.ann)
#             new_position = current_position + v
#             p.ann.set_parameters(new_position)
#             # Limit velocity to avoid excessive movement
#             v = np.clip(v, -max_velocity, max_velocity)
            
#             p.v = v

#             # Update particle's position
#             p.ann = p.ann + p.v

#         # i += 1
#         # if err < err_crit:
#         #     break

#         # Early stopping criterion
#         if abs(gbest.fitness - p.fitness) < err_crit:
#             break

#         # Progress bar (print '.' every 10% of iterations)
#         if i % (iter_max // 10) == 0:
#             print('.')

#     print ('\nParticle Swarm Optimisation\n')
#     print ('PARAMETERS\n','-'*9)
#     print ('Population size : ', pop_size)
#     print ('Dimensions      : ', dimensions)
#     print ('Error Criterion : ', err_crit)
#     print ('c1              : ', c1)
#     print ('c2              : ', c2)
#     print ('function        :  f6')

#     print ('RESULTS\n', '-'*7)
#     print ('gbest fitness   : ', gbest.fitness)
#     print ('gbest ann    : ', gbest.ann)
#     print ('iterations      : ', i+1)
#     return gbest
#     ## Uncomment to print particles
#     # for p in particles:
#     #     print('params: %s, fitness: %s, best: %s' % (p.params, p.fitness, p.best))


# run experiment
#loss, accuracy = mini_batch(ann, data, classes, epochs, learning_rate, loss_func, batch_size) 

# plot, display results
# plt.plot(range(epochs), loss, label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()