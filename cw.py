import numpy as np
import math
from numpy import array
from random import random
from math import sin, sqrt
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
# class Sigmoid (Activation):
#     def evaluate(x):
#         return 1 / (1 + math.exp(-x))
    
#     # derivative of f = sigma * (1- sigma)
#     def derivative(x):
#         f = 1 / (1 + math.exp(-x))
#         return f * (1 - f)

# # tanh activation – sub class of activation
# class tanh(Activation):
#     def evaluate(x):
#         return math.tanh(x)

#     # derivative of tanh(x) = sech^2 (x) = 1- tanh^2 (x)
#     def derivative(x):
#         f = math.tanh(x)
#         return 1 - f**2

# # ReLU activation – sub class of activation
# class ReLU(Activation):
#     def evaluate(x):
#         return np.maximum(0, x)

#     # derivative of max(0, x) = 0 if x < 0, 1 if x > 0 because slope is uniform which is the derivative of the ReLU Graph(1).
#     def derivative(x):
#         return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    def evaluate(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative(self, x):
        f = 1 / (1 + math.exp(-x))
        return f * (1 - f)

class tanh(Activation):
    def evaluate(self, x):
        return math.tanh(x)

    def derivative(self, x):
        f = math.tanh(x)
        return 1 - f**2

class ReLU(Activation):
    def evaluate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
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

#Layer class providing forward method 
class Layer:
    def __init__(self, nb_inputs, nb_nodes, activation):
        #declare attributes: nb_nodes, X_in, W, B, activation
        self.nb_nodes = nb_nodes
        size = (nb_nodes, nb_inputs)
        self.activation = activation
        # generating random weights for each neuron in the layer
        self.W = np.random.uniform(-1, 1, size)
        #generates an array of specified shape, where each element is drawn from a standard normal distribution (also known as a Gaussian distribution) with a mean of 0 and a standard deviation of 1.
        # generating random biases for each neuron in the layer
        self.B = np.random.uniform(-1, 1, size)
        self.activation = activation
    
    def forward(self, fin):
        self.X_in = fin
        active = Activation(self.activation)
        out = active.evaluate(np.dot(self.W.T, self.X_in) + self.B)
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

#Create a network using the parameters provides by the user 
class ANNBuilder:
    def build(nb_layers, list_nb_nodes, list_functions): 
        ann = Network()
        for i in range(nb_layers):
            layer = Layer(list_nb_nodes[i], list_nb_nodes[i+1], list_functions[i]) 
            ann.append(layer)
        return ann

iter_max = 10000
pop_size = 100
dimensions = 2
c1 = 2
c2 = 2
err_crit = 0.00001

class Particle:
    def __init__(self, nb_layers, list_nb_nodes, list_functions):
        self.ann = ANNBuilder.build(nb_layers, list_nb_nodes, list_functions)
        self.fitness = 0.0
        self.best_fitness = 0.0
        self.best_ann = self.ann
        self.v = 0.0
#PSO
def PSO(X, y, layers, nodes, functions, iter_max=10000, pop_size=100, dimensions=2, c1=2, c2=2, err_crit=0.00001):
    # class Particle:
    #     # pass
    #     def __init__(self, dimensions):
    #         self.params = np.array([random() for _ in range(dimensions)])
    #         self.fitness = 0.0
    #         self.best = self.params
    #         self.v = np.zeros(dimensions)

    # class Particle:
    #     def __init__(self, nb_layers, list_nb_nodes, list_functions):
    #         self.ann = ANNBuilder.build(nb_layers, list_nb_nodes, list_functions)
    #         self.fitness = 0.0
    #         self.best = self.ann
    #         self.v = 0.0

    def f6(para):
        '''Schaffer's F6 function'''
        num = (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))**2) - 0.5
        denom = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1])))**2
        f6 =  0.5 - (num / denom)
        errorf6 = 1 - f6
        return f6, errorf6
    
    def evaluate_ann(ann, X, y):
        # Assuming X is the feature matrix and y is the target vector (0 or 1)
        y_pred = []
        for sample in X:
            # Assuming the ANN output is a single value between 0 and 1
            output = ann.forward(sample)
            y_pred.append(1 if output >= 0.5 else 0)

        accuracy = np.sum(y_pred == y) / len(y)
        fitness = accuracy  # You can use other metrics as needed
        errorf6 = 1 - fitness
        return fitness, errorf6

    particles = [Particle(layers, nodes, functions) for _ in range(pop_size)]

    for p in particles:
        p.informants = np.random.choice(particles, size=10, replace=False)

    gbest = particles[0]
    err = float('inf')
    i = 0

    max_velocity = 0.1 

    while i < iter_max:
        for p in particles:
            fitness, _ = evaluate_ann(p.ann, X, y)
            if fitness > p.fitness:
                p.fitness = fitness
                p.best_ann = p.ann
                p.best_fitness = fitness

            if fitness > gbest.fitness:
                gbest = p

        # Calculate particle's new velocity
            v = p.v + c1 * random() * (p.best_ann - p.ann) + c2 * random() * (gbest.best_ann - p.ann)

            # Limit velocity to avoid excessive movement
            v = np.clip(v, -max_velocity, max_velocity)
            
            p.v = v

            # Update particle's position
            p.ann = p.ann + p.v

        # i += 1
        # if err < err_crit:
        #     break

        # Early stopping criterion
        if abs(gbest.fitness - p.fitness) < err_crit:
            break

        # Progress bar (print '.' every 10% of iterations)
        if i % (iter_max // 10) == 0:
            print('.')

    print ('\nParticle Swarm Optimisation\n')
    print ('PARAMETERS\n','-'*9)
    print ('Population size : ', pop_size)
    print ('Dimensions      : ', dimensions)
    print ('Error Criterion : ', err_crit)
    print ('c1              : ', c1)
    print ('c2              : ', c2)
    print ('function        :  f6')

    print ('RESULTS\n', '-'*7)
    print ('gbest fitness   : ', gbest.fitness)
    print ('gbest ann    : ', gbest.ann)
    print ('iterations      : ', i+1)
    return gbest
    ## Uncomment to print particles
    # for p in particles:
    #     print('params: %s, fitness: %s, best: %s' % (p.params, p.fitness, p.best))

# Load the dataset
url = "data_banknote_authentication.txt"
data = pd.read_csv(url, header=None)

# Extract features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#read and prepare your data x, y 
layers = 3
nodes = [3,3,5,2]
functions = ["Logistic", "Logistic", "Hyperbolic tangent", "ReLU"]

#read ANN params from user: layers, nodes, functions 
# ann = ANNBuilder.build(layers, nodes, functions)

best = PSO(X, y, layers, nodes, functions)
print(best.ann)
# read hyper-parameters: epochs, rate, batch_size, loss
# epochs = 6
# learning_rate = 0.5
# batch_size = 30
# loss_func = Loss.evaluate("MSE", )

# run experiment
#loss, accuracy = mini_batch(ann, data, classes, epochs, learning_rate, loss_func, batch_size) 

# plot, display results
# plt.plot(range(epochs), loss, label='Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()