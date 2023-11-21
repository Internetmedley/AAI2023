import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

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

    

    # Generate some random data for demonstration purposes
    # np.random.seed(42)
    # X = np.random.rand(100, 2)  # 100 samples with 2 features
    # y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification based on the sum of features

    # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def knn_predict(X_train, y_train, X_test, k=3):
        predictions = []
        for x_test in X_test:
            distances = np.linalg.norm(X_train - x_test, axis=1)
            k_neighbors_indices = np.argsort(distances)[:k]
            print(k_neighbors_indices)
            k_neighbor_labels = [y_train[indice] for indice in k_neighbors_indices]
            most_common = max(set(k_neighbor_labels),key=k_neighbor_labels.count)
            
            print(most_common)
        print(np.array(predictions))

    # Set the value of k (number of neighbors)
    k_value = 66

    
    # Make predictions on the test set
    y_pred = knn_predict(validata, v_labels, data, k=k_value)

    # # Evaluate the accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy with k={k_value}: {accuracy:.4f}")


if __name__ == '__main__':
    main()
