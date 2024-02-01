from ucimlrepo import fetch_ucirepo 
import numpy as np
from model import *
import NN as N

np.random.seed(0)                                       # set seed to 0 so the results are always the same

# fetch dataset 
iris = fetch_ucirepo(id=53) 

X = np.array(iris.data.features)
y = np.array(iris.data.targets)

# apply min-max normalisation
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))       

# encode the labels to one hot encoded
y = N.one_hot_encode(y)             

indices = np.arange(X.shape[0])
np.random.shuffle(indices)                              # randomise the indices so that we can make a good train data set and validation dataset
train_size = 0.8                                        # use 80% for training, the rest for validation
train_samples = int(train_size * X.shape[0])

# split the indices into training and testing sets
train_indices = indices[:train_samples]
test_indices = indices[train_samples:]
# re-assign data for training and testing
X_train, x_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# initializing the layers
dense1 = N.Hidden_Layer_Dense(4, X.shape[1])                # 4 neurons with 4 weights, this is the hidden layer
dense2 = N.Output_Layer_Dense(y.shape[1], 4)                # 3 neurons with 4 weights, 3 neurons is required, because 3 classes possible
my_model = Model(dense1, dense2)

# # uncomment this part to test the network with an extra hidden layer (or add more)
# dense1 = N.Hidden_Layer_Dense(4, X.shape[1])                # 4 neurons with 4 weights
# dense2 = N.Hidden_Layer_Dense(5, 4)                         # 5 neurons with 4 weights
# dense3 = N.Output_Layer_Dense(y.shape[1], 5)                # 3 neurons with 5 weights
# my_model = Model(dense1, dense2, dense3)

# start training the network
my_model.fit(X_train, y_train, epochs=500, lr=0.1)

predictions = my_model.predict(x_test)
print("Output of output layer: \n", predictions)

acc = my_model.evaluate(predictions, y_test)
print(f"accuracy: {acc:.2f}%, with learning rate: {my_model.lr} and epochs: {my_model.epochs}.")