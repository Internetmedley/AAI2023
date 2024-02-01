import numpy as np


class Model:
    """
    Class which can take multiple Layer_Dense objects in constructor to make a list of layers to use for forward 
    and backward propagation for use with classification.
    """
    def __init__(self, *args, **kwargs):
        self.hidden_layers = list(args[:-1])                        # up to last is reserved for hidden layers
        self.output_layer = args[-1]                                # last argument is reserved for output layer


    def forward(self, inputs):
        """
        Calls forward propagation on all layers for this model, where inputs is given to the first layer and their output is input for the next layer.
                Parameters:
                        inputs (list[list[float]]):    A 2d-list of floats
        """
        for layer in self.hidden_layers:
            layer.forward(inputs)
            inputs = layer.output
        self.output_layer.forward(inputs)
    
    def backward(self, inputs):
        """
        Calls backward propagation on all layers for this model, starting with the output layer first given the input. 
        The last layer is given to each subsequent hidden layer for the backward function. 
                Parameters:
                        inputs (list[list[float]]):    A 2d-list of floats
        """
        self.output_layer.backward(inputs)
        self.output_layer.update_weights_and_biases(self.lr)
        prev_layer = self.output_layer
        for layer in self.hidden_layers[::-1]:                      #[::-1] loops through in reverse, so it starts with last hidden layer
            layer.backward(prev_layer)
            layer.update_weights_and_biases(self.lr)
            prev_layer = layer
        
        
    def fit(self, X_train, x_labels, epochs, lr=0.01):
        """
        Trains the network given a training dataset and corresponding labels in onehot-encoded format by looping through the number of epochs and 
        basically adjusting the weights in the neurons in the layers using forward and backward-propagation. 
                Parameters:
                        inputs (list[list[float]]):     A 2d-list of floats
                        x_labels (list[list[int]]):     A 2d-list of integers 
                        epochs (int)                    An integer
                        lr (float)                      A float
        """
        self.epochs = epochs
        self.lr = lr
        for _ in range(epochs):
            for i in range(len(X_train)):
                self.forward(X_train[i])
                self.backward(x_labels[i])

    def predict(self, X_data):
        """
        Predicts what classes belong to each datapoint in X_data by using forward propagation only, then returns the predictions.
                Parameters:
                        X_data (list[list[float]]):     A 2d-list of floats
                        
                Returns:
                        predictions (list[list[float]]):    A 2d-list of output layer outputs
        """
        predictions = []
        for datapoint in X_data:
            self.forward(datapoint)
            predictions.append(self.output_layer.output)
        return np.array(predictions)


    def evaluate(self, y_pred, y_labels):
        """
        Evaluates the model's performance 
                Parameters:
                        y_pred (list[list[float]]):     A 2d-list of floats
                        y_labels (list[list[int]]):     A 2d-list of integers 

                Returns:
                        accuracy (float):   A float representing accuracy score
        """
        matches = np.count_nonzero(np.argmax(y_pred, axis=1) == np.argmax(y_labels, axis=1))
        accuracy = float(matches / len(y_labels)) * 100
        return accuracy