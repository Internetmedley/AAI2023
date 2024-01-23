from ucimlrepo import fetch_ucirepo 
import numpy as np
import NN as N

np.random.seed(0)           # set seed to 0 so the results are always the same

# fetch dataset 
iris = fetch_ucirepo(id=53) 

X = np.array(iris.data.features)
y = np.array(iris.data.targets)

y = N.One_Hot_Encode(y)             # encode the labels to one hot encoded, which we need for comparison with output of neural network

indices = np.arange(X.shape[0])

np.random.shuffle(indices)          # randomise the indices so that we can make a good train data set and validation dataset
train_size = 0.8                    # use 80% for training, the rest for validation
train_samples = int(train_size * X.shape[0])

# split the indices into training and testing sets
train_indices = indices[:train_samples]
test_indices = indices[train_samples:]
# data (as pandas dataframes) 
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


dense1 = N.Layer_Dense(4)             # 4 neurons with random weights, this is the hidden layer
activation1 = N.Activation_Sigmoid()

dense2 = N.Layer_Dense(3)             # 3 neurons with random weights, 3 is required, because 3 classes possible
activation2 = N.Activation_Sigmoid()

learning_rate = 0.001
for epoch in range(1):
    predictions = np.empty(y_train.shape)
    for i in range(len(X_train)):
        #print('\n')
        dense1.forward(X_train[i])
        #print(dense1.output)
        activation1.forward(dense1.output)
        #print(activation1.output)
        dense2.forward(activation1.output)
        #print(dense2.output)
        activation2.forward(dense2.output)
        #print(activation2.output)
        #predictions[i] = activation2.output
        #print("pred :", activation2.output)

        true =  y_train[i]
        pred = activation2.output
        error = pred - true
        #output layer back propagation:
        outputlayer_delta = pred * (1 - pred)           #sigmoid derivative, met delta gaan we de weights aanpassen
        #print("d: ", outputlayer_delta)
        outputlayer_weights = [neuron.weight for neuron in dense2.neurons]
        outputlayer_biases = [neuron.bias for neuron in dense2.neurons]
        print(outputlayer_biases)
        hiddenlayer_delta = (activation1.output * (1 - activation1.output)) * np.sum(outputlayer_delta * outputlayer_weights)
        print(hiddenlayer_delta)
        outputlayer_biases = outputlayer_biases + learning_rate * outputlayer_delta
        print(outputlayer_biases)
        

        






    # loss_fn = N.Loss_MeanSquaredError()
    # loss = loss_fn.calculate(predictions, y_train)    #verwacht en actual
    

    # print(predictions)
    # print(loss)
    # print(np.mean(predictions))
    # activation2.outputlayer_delta(y_train)          #backward voor de laatste laag

    # print("delta is: ", activation2.delta_output)


    # dense2.backward(activation2.delta_output)
    # activation1.backward(dense2.delta_output)
    # dense1.backward(activation1.delta_output)




    #print(1 / (1 + np.exp(-0.1)))
