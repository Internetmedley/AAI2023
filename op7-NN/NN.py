import numpy as np

def sigmoid(z):
    #z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def one_hot_encode(y_labels):
    classes = np.unique(y_labels)                                   #een set van alle mogelijke classes
    output = np.zeros(shape=(len(y_labels), len(classes)))          
    for i in range(len(y_labels)):
        output[i][np.where(classes == y_labels[i])] = 1
    return output


class Neuron:
    def __init__(self, n_weights, b=0):
        #self.weights = [ w for w in np.random.randn(n_weights) ]
        self.weights = [ np.random.random_sample() for _ in range(n_weights) ]      #makes a random value between 0.0 and 1.0
        self.bias = b
        self.delta = 0


class Layer_Dense:
    def __init__(self, n_neurons, n_weights):
        self.neurons = [ Neuron(n_weights) for _ in range(n_neurons) ]

    def forward(self, inputs):
        self.inputs = inputs
        #self.weighted = []
        self.output = []
        
        for i in range(len(self.neurons)):
            sum_values = 0.0
            for j in range(len(inputs)):
                sum_values += inputs[j] * self.neurons[i].weights[j]
                #print("sum_values: ", sum_values)
            self.output.append(sigmoid(sum_values + self.neurons[i].bias))
        print("self.output: ", self.output)


    def update_weights(self, lr):                #seems done
        for i in range(len(self.neurons)):
            #print("before: ", self.neurons[i].weights)
            for j in range(len(self.neurons[i].weights)):
                self.neurons[i].weights[j] += (lr * self.neurons[i].delta * self.inputs[j])
            #print("after: ", self.neurons[i].weights)
    
    def update_biases(self, lr):                        #seems done?
        for i in range(len(self.neurons)):     
            #print("neuron bias before: ", self.neurons[i].bias)
            #print("neuron delta before: ", self.neurons[i].delta)
            
            self.neurons[i].bias += (lr * self.neurons[i].delta)
            #print("neuron bias after:  ", self.neurons[i].bias)
            #print("neuron delta after: ", self.neurons[i].delta)

        
class Hidden_Layer_Dense(Layer_Dense):              #>>>> now seems to be done
    def backward(self, previous_layer : Layer_Dense):
        for i in range(len(self.neurons)):
            delta_sum = 0.0
            for j in range(len(previous_layer.neurons)):
                delta_sum += self.output[j] * previous_layer.neurons[j].delta
            self.neurons[i].delta = sigmoid_derivative(self.output[i] * delta_sum)
        #print("hidden layer back")


class Output_Layer_Dense(Layer_Dense):              #this one seems good
    def backward(self, y_true):
        error = (y_true - self.output)
        print("error: ", error)
        for i in range(len(self.neurons)):
            self.neurons[i].delta = sigmoid_derivative(self.output[i]) * error[i] 
            print("delta in backward:", self.neurons[i].delta)    
        #print("output layer back")






#--------------------------------------
# unused
# class Loss_MeanSquaredError:              
#     def calculate(self, y_predict, y_true):
#         N = len(y_true)
#         mse = np.sum(np.square(y_true - y_predict) / N)               
#         return mse
