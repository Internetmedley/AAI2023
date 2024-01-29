import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def calculate_distances(a, b):
    """
    Returns a a numpy array containing distance corresponding to the Euclidian distances between datapoints A and all neighbouring datapoints B.

            Parameters:
                    a (2d-array):       A numpy 2d-array
                    b (2d-array):       Another numpy 2d-array

            Returns:
                    distances (np-array): Numpy array containing distance
    """
    distances = np.sqrt(np.sum(np.square(a - b), axis=1))
    return distances


def predict_cluster_labels(cluster, x_labels):
    """
    Returns a list with the predicted labels given a cluster array and an array with labels.
            Parameters:
                    cluster (2d-array):     A numpy 2d-array
                    x_labels (2d-array):    Another numpy 2d-array

            Returns:
                    y_pred (list): List containing predicted labels
    """
    y_pred = []
    
    votes = { key : [] for key in np.unique(cluster) }          #initialize a dict with votes for each possible cluster number 

    for cluster_num, label in zip(cluster, x_labels):
        votes[cluster_num].append(label)
    for key, value in votes.items():
        if len(Counter(value)) != 0:
            votes[key] = max(Counter(value))                    #tally which label gets the most votes and assign it

    for cluster_num in cluster:
        y_pred.append(votes[cluster_num])
    return y_pred


def evaluate(y_predictions, y_labels):
    """
    Returns an accuracy score given a list of predictions and labels.
            Parameters:
                    y_predictions (np-array):   A numpy array
                    y_labels (np-array):        Another numpy array

            Returns:
                    accuracy (float): A float containing accuracy percentage score
    """
    matches = np.count_nonzero(y_labels == y_predictions) 
    accuracy = float(matches / len(y_labels)) * 100
    return accuracy


def cluster_K_means(centroids, iterations, X):
    """
    Returns an array containing the cluster numbers each datapoint is assigned to and the within cluster sum of this cluster. 
            Parameters:
                    centroids (2d-np-array):    A 2d-numpy array
                    iterations (int):       An integer
                    X (2d-np-array)             A 2d-np-array

            Returns:
                    y (np-array):                 A numpy array containing assigned cluster numbers for each datapoint in X
                    within_cluster_sum (float):   A float containing the within cluster sum squared
    """
    y = []
    within_cluster_sum = 0
    for _ in range(iterations):         #number of times to calculate new centroids
        within_cluster_sum = 0.0            #reset the sum so it uses the one for the last iteration each time 
        y = np.array([np.argmin(calculate_distances(centroids, X_data)) for X_data in X])       #calculate distances of each point towards all centroids      
        
        cluster_indices = []
        for i, center in enumerate(centroids):
            cluster_points = X[y == i]
            within_cluster_sum += np.sum((cluster_points - center) ** 2)
            cluster_indices.append(np.argwhere(y == i))

        new_cluster_centers = []
        for i, indices in enumerate(cluster_indices):
            if len(indices) == 0:
                new_cluster_centers.append(centroids[i])
            else:
                new_cluster_centers.append(np.mean(X[indices], axis=0)[0])
    
        if np.max(centroids - np.array(new_cluster_centers)) < 0.00000001:  #if maximum distance is less than 0.0001
            break
        else:
            centroids = np.array(new_cluster_centers)
    return y, within_cluster_sum


def main():
    X_train = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    d_dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
    x_labels = []
    for label in d_dates:
        if label < 20000301:
            x_labels.append('winter')
        elif 20000301 <= label < 20000601:
            x_labels.append('lente')
        elif 20000601 <= label < 20000901:
            x_labels.append('zomer')
        elif 20000901 <= label < 20001201:
            x_labels.append('herfst')
        else: # from 01-12 to end of year
            x_labels.append('winter')

    y_train = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    v_dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
    y_labels = []
    for label in v_dates:
        if label < 20010301:
            y_labels.append('winter')
        elif 20010301 <= label < 20010601:
            y_labels.append('lente')
        elif 20010601 <= label < 20010901:
            y_labels.append('zomer')
        elif 20010901 <= label < 20011201:
            y_labels.append('herfst')
        else: # from 01-12 to end of year
            y_labels.append('winter')
    
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))     #normalise data with min-max,
    y_train = (y_train - y_train.min(axis=0)) / (y_train.max(axis=0) - y_train.min(axis=0))     #the algorithm works better without it
    
    np.random.seed(0)       #set random seed to 0 so we always get the same results
    max_k = 9
    max_iterations = 100
    all_kmeans_clusters = []
    all_cluster_sums = []
    for k in range(1, max_k):
        centroids = np.random.uniform(X_train.min(axis=0), X_train.max(axis=0), size=(k, X_train.shape[1]))     #make k random cluster centroids
        cluster, within_cluster_sum = cluster_K_means(centroids, max_iterations, X_train)
        all_kmeans_clusters.append(cluster)
        all_cluster_sums.append(within_cluster_sum)


    plt.figure(figsize=(9, 6))
    plt.plot(range(1, max_k), all_cluster_sums, marker='.', ls='--', c='r')
    plt.title("Scree Plot")
    plt.xlabel("No. of K-clusters")
    plt.ylabel("Intra cluster distance")
    plt.xticks(range(1, max_k))
    plt.show()

    k = 7
    predictions = predict_cluster_labels(all_kmeans_clusters[k], x_labels)
    accuracy = evaluate(np.array(predictions), x_labels)
    print(f"kmeans_clustering met k={k} heeft een accuracy van: {accuracy:.2f}%\n")

            
if(__name__ == '__main__'):
    main() 