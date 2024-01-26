from ucimlrepo import fetch_ucirepo 
import numpy as np
import NN as N

np.random.seed(0)           # set seed to 0 so the results are always the same

def dot_product(a, b):
    x = 0
    for i in range(len(a)):
        x += a[i] * b[i]
    return x

def dot_product2d(a, b):
    arr = [ dot_product(a[i], b) for i in range(len(a)) ]
    return arr
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
data_offset = 120               #80% of 150 is 120 for training, and the remaining 30 for testing
X = np.array(iris.data.features)
y = np.array(iris.data.targets)

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




dense1 = N.Layer_Dense(4, 4)            # create a layer with the 4 neurons and 4 inputs being petal length, width & sepal length, width 
activation1 = N.Activation_ReLU()       
dense2 = N.Layer_Dense(4, 3)            # hidden layer met 3 neurons
activation2 = N.Activation_ReLU()
dense3 = N.Layer_Dense(3, 3)            # laatse laag met 3 neurons voor de output layer voor iris-virginica, iris-versicolor en iris-setosa
activaton3 = N.Activation_Softmax()

loss_activation = N.Loss_CategoricalCrossEntropy()

# hou de beste weights en biases bij
lowest_loss = 999999                            # een initiele waarde
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_weights = dense3.weights.copy()
best_dense3_biases = dense3.biases.copy()


dense1.forward(X_train)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)

loss = loss_activation.forward(dense3.output, y_train)

print(loss_activation.output[:5])
print('loss: ', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_train.shape) == 2:
    y_train = np.argmax(y_train, axis=1)
accuracy = np.mean(predictions==y_train)

print('acc: ', accuracy)

loss_activation.backward(loss_activation.output, y_train)
dense3.backward(loss_activation.dinputs)
activation2.backward(dense3.dinputs)
dense2.backward(activation2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.weights)
print(dense1.biases)
print(dense2.weights)
print(dense2.biases)
print(dense3.weights)
print(dense3.biases)

