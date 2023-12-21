import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weight = 0.01 * np.random.randn()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases                #nog herscrijven naar eigen functie

    def backward(self, dvalues):
        # gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # make copy first since we modify original
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0 ] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))      # haal het maximum van alle inputs af van 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) 
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1, 1)
            #calculate jacobian matrix of the output 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs


        self.output = 1 / (1 + np.exp(-inputs)) 



class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)      #tegen divide by zero errors
        
        if len(y_true.shape) == 1:          #pass scalar values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:        #if one hot encoded 
            correct_confidences= np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

        def backward(self, dvalues, y_true):
            # nr of samples 
            samples = len(dvalues)

            # nr of labels in every sample, use the first to count them
            labels = len(dvalues[0])
            # if labels are sparse, turn them into a one-hot vector
            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]
            
            # calc gradient
            self.dinputs = -y_true / dvalues
            # normalize gradient
            self.dinputs = self.dinputs / samples
            


