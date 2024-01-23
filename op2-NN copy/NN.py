import numpy as np

def One_Hot_Encode(y_labels):
    classes = np.unique(y_labels)                                   #een set van alle mogelijke classes
    output = np.zeros(shape=(len(y_labels), len(classes)))          
    for i in range(len(y_labels)):
        output[i][np.where(classes == y_labels[i])] = 1
    return output


class Neuron:
    def __init__(self, w, b=0):
        self.weight = 1 * w
        self.bias = b

class Layer_Dense:
    def __init__(self, n_neurons):
        self.neurons = [Neuron(weight) for weight in np.random.randn(n_neurons)]

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.array([])
        
        for n in self.neurons:
            a = 0
            for x in inputs:
                a += x * n.weight
            a += n.bias

            self.output = np.append(self.output, a)
        
    def update_weights(self, weights):
        


        self.neurons = [Neuron(weight) for weight in weights]
        

    def backward(self, dvalues):
        pass


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs)) 

    def outputlayer_delta(self, target):
        self.delta_output = self.output - target * self.output * (1 - self.output)           # bereken de derivative van delta values met MSE

    def hiddenlayer_delta(self, delta_inputs):
        self.delta_hidden = (1 / (1 + np.exp(-self.inputs))) * (1 - (1 / (1 + np.exp(-self.inputs)))) * np.sum(delta_inputs * self.weights)




class Loss_MeanSquaredError:
    def calculate(self, y_predict, y_true):
        N = len(y_true)
        mse = np.sum(np.square(y_true - y_predict) / N)               
        return mse
