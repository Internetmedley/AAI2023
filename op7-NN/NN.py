import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function to use on the weighted sum of a neuron.
            Parameters:
                    z (float):     A float
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Returns the sigmoid derivative of input z.
            Parameters:
                    z (float):     A float
    """
    return sigmoid(z) * (1 - sigmoid(z))

def one_hot_encode(y_labels):
    """
    Transforms the list of labels to become one-hot-encoded(https://en.wikipedia.org/wiki/One-hot) so it is easier to evaluate the model.
            Parameters:
                    y_labels (iris_targets):     A 2d-list iris_targets
                    
            Returns:
                    output (list[list[int]]):    A 2d-list of integers
    """
    classes = np.unique(y_labels)                                   #een set van alle mogelijke classes
    output = np.zeros(shape=(len(y_labels), len(classes)))          
    for i in range(len(y_labels)):
        output[i][np.where(classes == y_labels[i])] = 1
    return output


class Neuron:
    def __init__(self, n_weights, b=0):
        self.weights = [ np.random.random_sample() for _ in range(n_weights) ]      #makes a random value between 0.0 and 1.0
        self.bias = b
        self.input = 0
        self.output = 0
        self.delta = 0

    def activate(self, inputs):
        """
        Activates the neuron given input to calculate the weighted sum of the input, addinf the bias.
                Parameters:
                        inputs (list[list[float]]):     A 2d-list of floats
        """
        self.inputs = inputs

        weighted_sum = self.bias
        for i in range(len(inputs)):
            weighted_sum += self.weights[i] * inputs[i]
        self.output = sigmoid(weighted_sum)


class Layer_Dense:
    def __init__(self, n_neurons, n_weights):
        self.neurons = [ Neuron(n_weights) for _ in range(n_neurons) ]


    def forward(self, inputs):
        """
        Calls the activation function for each neuron in this layer given an input.
            Parameters:
                    inputs (list[list[float]]):     A 2d-list of floats
        """
        self.output = []
        self.inputs = inputs
        for neuron in self.neurons:
            neuron.activate(inputs)
            self.output.append(neuron.output)

    def update_weights_and_biases(self, lr):
        """
        Updates the weights and biases for each neuron in this layer given a learning rate.
            Parameters:
                    lr (float):     A float
        """                
        for neuron in self.neurons:
            for i in range(len(neuron.weights)):
                neuron.weights[i] += lr * neuron.delta * self.inputs[i]         #update weight
            neuron.bias += lr * neuron.delta                                    #update bias                  
    
class Output_Layer_Dense(Layer_Dense):              
    def backward(self, y_true):
        """
        Backward propagation on each neuron in this output layer given expected label to calculate their new deltas.
            Parameters:
                    y_true (list[int]):     A list of integers
        """
        error = y_true - self.output
        for i, neuron in enumerate(self.neurons):
            neuron.delta = sigmoid_derivative(neuron.input) * error[i]      #calculate new delta

class Hidden_Layer_Dense(Layer_Dense):              
    def backward(self, previous_layer : Layer_Dense):
        """
        Backward propagation on each neuron in this layer to recalculate their deltas given a layer that backwarded before this one.
            Parameters:
                    previous_layer (Layer_Dense):     A Layer_Dense superclass object
        """
        for i, neuron in enumerate(self.neurons):
            delta_weight_sum = 0
            for prev_layer_neuron in previous_layer.neurons:
                delta_weight_sum += prev_layer_neuron.delta * prev_layer_neuron.weights[i]
            neuron.delta = sigmoid_derivative(neuron.output) * delta_weight_sum             #calculate new delta
