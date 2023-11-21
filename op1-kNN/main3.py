import numpy as np

class KNN:
    def __init__(self, k=66):
        """
        Initialize the KNN classifier.

        Parameters:
        - k: int, number of neighbors to consider (default is 3).
        """
        self.k = k
        self.predictions = []

    def fit(self, X_train, y_train):
        """
        Fit the KNN model with training data.

        Parameters:
        - X_train: array-like, shape (n_samples, n_features), training data features.
        - y_train: array-like, shape (n_samples,), training data labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test_data):
        """
        Predict labels for test data.

        Parameters:
        - X_test_data: array-like, shape (n_samples, n_features), test data features.

        Returns:
        - predictions: array-like, shape (n_samples,), predicted labels for the test data.
        """
        self.predictions = []
        for x_test in X_test_data:
            self.predictions.append(self._predict(x_test))
        return np.array(self.predictions)

    def _predict(self, val_data):
        """
        Predict a single label for the given data point.

        Parameters:
        - val_data: array-like, shape (n_features,), data point for which to predict the label.

        Returns:
        - most_common: the predicted label for the given data point.
        """
        # Compute distances between x and all examples in the training set
        #distances = [np.linalg.norm(val_data - x_train) for x_train in self.X_train]
        distances = np.linalg.norm(val_data - self.X_train, axis=1)

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[indice] for indice in k_indices]
        # In case of a tie, it selects the nearest neighbor
        most_common = max(set(k_neighbor_labels), key=k_neighbor_labels.count)
        return most_common

def main():
    data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
    d_labels = []
    for label in dates:
        if label < 20000301:
            d_labels.append('winter')
        elif 20000301 <= label < 20000601:
            d_labels.append('lente')
        elif 20000601 <= label < 20000901:
            d_labels.append('zomer')
        elif 20000901 <= label < 20001201:
            d_labels.append('herfst')
        else: # from 01-12 to end of year
            d_labels.append('winter')

    validata = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    validates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
    v_labels = []
    for label in dates:
        if label < 20010301:
            v_labels.append('winter')
        elif 20010301 <= label < 20010601:
            v_labels.append('lente')
        elif 20010601 <= label < 20010901:
            v_labels.append('zomer')
        elif 20010901 <= label < 20011201:
            v_labels.append('herfst')
        else: # from 01-12 to end of year
            v_labels.append('winter')

    weerclassifier = KNN()
    weerclassifier.fit(data, d_labels)
    bullshit = weerclassifier.predict(validata)
    print(len(bullshit))

    scores = []
    for p in range(len(bullshit)):
        if(bullshit[p] == v_labels[p]):
            scores.append(1)
        else:
            scores.append(0)
    print(np.mean(scores))

            

if __name__ == '__main__':
    main()