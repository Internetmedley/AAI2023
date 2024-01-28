import numpy as np


def calculate_distances(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=1))

def find_most_common_neighbour_K(distancesAndLabels, k): 
    """
    Returns the class of the most common neighbour from a list of tuples with distances and labels.

            Parameters:
                    distancesAndLabels (list(tuple)):   A numpy 2d-array
                    k (int):                            An integer

            Returns:
                    most_common_class (str):    String containing the class of the most frequent neighbour. 
    """                      
    most_common_class = ''
    
    distancesAndLabels.sort(key=lambda x : x[0])                                        #sort ascending so it start with shortest distance
    classes = np.unique(np.array([ v[1] for v in distancesAndLabels ]))                 #calculate which classes exist
    k_closest = np.array([ v[1] for v in distancesAndLabels[:k] ])                      #get the k closest neighbours for this distance
    
    frequencies = []
    while(True):
        for c in classes:
            frequencies.append(tuple((np.count_nonzero(k_closest == c), c)))            #count number of appearances of each label in list of closest neighbours

        frequencies.sort(key=lambda x : x[0], reverse=True)             #sort the elements descending
        if(frequencies[0][0] == frequencies[1][0]):                     #if true, there is a conflict meaning the highest frequent neighbours are the same
            k_closest = k_closest[:-1]                                  #leave out the last neighbour if there is a conflict and then repeat the process
            frequencies = []                                            #because we're sorting beforehand, comparing the first two elements always works  
            continue
        else:
            break
    most_common_class = frequencies[0][1]

    return most_common_class


def predict(k, distances):
    """
    Returns the prediction for the most common class from datapoints X_train neighbouring y_train given a K value for no. of neighbours.
    """
    most_common_classes = [ find_most_common_neighbour_K(d, k) for d in distances ]
    return most_common_classes


def evaluate(y_predictions, y_labels):
    """
    Returns an accuracy score given a list of predictions and labels.
    """
    matches = np.count_nonzero(y_labels == y_predictions) 
    accuracy = float(matches / len(y_labels)) * 100
    return accuracy

def find_best_K_val(X_distances, y_labels, epochs):
    """
    Returns the best value for K by iterating through a number of them equal to 1/3 of train size and comparing their scores.

            Parameters:
                    y_train (2d-array):     A numpy 2d-array
                    y_labels (np.array):    A numpy array
                    X_train (2d-array):     A numpy 2d-array
                    x_labels (np.array):    A numpy array

            Returns:
                    best_k (int):    Integer containing the best calculated value for K.
    """ 
    best_k = 0
    best_score = 0
    for k in range(1, epochs):                                                  
        predictions = np.array(predict(k, X_distances))
        accuracy = evaluate(predictions, y_labels=y_labels)
        print(f"Training... ({k}/{epochs-1}) - k:{k}, accuracy: {accuracy}")
        if( accuracy > best_score ):
            best_score = accuracy
            best_k = k
    print(f"De beste k is: {best_k} met een accuracy van: {best_score}%")
    return best_k

def K_means(k, centroids, max_iterations, X):
    y = []
    within_cluster_sum = 0
    for _ in range(max_iterations):         #number of times to calculate new centroids
        y = np.array([np.argmin(calculate_distances(centroids, X_data)) for X_data in X])       #calculate distances of each point towards all centroids      
        #print(y)

        cluster_indices = [np.argwhere(y == i) for i in range(k)]   #list of indices of datapoints grouped in clusters
        
        for i, c in enumerate(centroids):
            cluster_indices = np.argwhere(y == i)
            



        #print(cluster_indices)

        new_cluster_centers = []
        

        for i, indices in enumerate(cluster_indices):
            if len(indices) == 0:
                new_cluster_centers.append(centroids[i])
            else:
                new_cluster_centers.append(np.mean(X[indices], axis=0)[0])
                within_cluster_sum += np.sum((cluster_indices - centroids))
    
        if np.max(centroids - np.array(new_cluster_centers)) < 0.00000001:  #if maximum distance is less than 0.0001
            
            break
        else:
            centroids = np.array(new_cluster_centers)
        #print("calculating route")
    return y


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
    
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))     #normalise data
    y_train = (y_train - y_train.min(axis=0)) / (y_train.max(axis=0) - y_train.min(axis=0))
    
    np.random.seed(0)       #set random seed to 0 so we always get the same results
    max_k = 8
    max_iterations = 100
    for k in range(2, max_k):
        centroids = np.random.uniform(X_train.min(axis=0), X_train.max(axis=0), size=(k, X_train.shape[1]))     #make k random cluster centroids
        labels = K_means(k, centroids, max_iterations, X_train)
        #print(labels)
        

          

        
            #print(len(distances1))
            #print(len(distances2))

    

if(__name__ == '__main__'):
    main() 