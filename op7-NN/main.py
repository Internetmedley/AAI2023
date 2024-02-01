from ucimlrepo import fetch_ucirepo 
import numpy as np
from model import *
import NN as N

np.random.seed()           # set seed to 0 so the results are always the same

# fetch dataset 
iris = fetch_ucirepo(id=53) 

X = np.array(iris.data.features)
y = np.array(iris.data.targets)

# apply min-max normalisation before any processing
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))       

# encode the labels to one hot encoded, which we need for comparison with output of neural network
y = N.one_hot_encode(y)             

# for x_data, y_data in enumerate(zip(X,y)):
#     print(x_data, y_data)

indices = np.arange(X.shape[0])

np.random.shuffle(indices)          # randomise the indices so that we can make a good train data set and validation dataset
train_size = 0.8                    # use 80% for training, the rest for validation
train_samples = int(train_size * X.shape[0])

# split the indices into training and testing sets
train_indices = indices[:train_samples]
test_indices = indices[train_samples:]
# data (as pandas dataframes) 
X_train, y_train = X[train_indices], X[test_indices]
x_labels, y_labels = y[train_indices], y[test_indices]

# for y_data, label in enumerate(zip(y_train, y_labels)):
#     print(y_data, label)

# initialiseren van de layers 
dense1 = N.Hidden_Layer_Dense(4, X_train.shape[1])              # 4 neurons with 4 weights, this is the hidden layer
dense2 = N.Output_Layer_Dense(y_labels.shape[1], 4)             # 3 neurons with 4 weights, 3 neurons is required, because 3 classes possible
my_model = Model(X_train[0], dense1, dense2)

# start training the network
my_model.fit(X_train, x_labels, epochs=200, lr=0.1)
predictions = my_model.predict(y_train)
print(predictions)
acc = my_model.evaluate(predictions, y_labels)
print(f"accuracy: {acc:.2f}%, with learning rate: {my_model.lr} and epochs: {my_model.epochs}.")
