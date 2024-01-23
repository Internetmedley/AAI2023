import numpy as np


def calculate_distances(a, b, labels):
    """
    Returns a list of tuples containing distance and class corresponding to the Euclidian distances between datapoints A and all neighbouring datapoints B.

            Parameters:
                    a (2d-array):       A numpy 2d-array
                    b (2d-array):       Another numpy 2d-array
                    labels (np.array):  A list of class labels

            Returns:
                    distances (list(tuple)): List of tuples containing distance and class
    """
    distances = []   
    for i in range(len(b)):
        distance = 0
        for j in range(len(b[0])):       
            distance += np.square(a[j] - b[i][j])   
        distances.append(tuple([np.sqrt(distance), labels[i]]))                         #sqrt want pythagoras
    return distances

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


def predict(k, y_train, X_train, x_labels):
    distances = calculate_distances(y_train, X_train, x_labels)
    most_common_label = find_most_common_neighbour_K(distances, k)
    return most_common_label

def evaluate(y_predictions, y_labels):
    matches = np.count_nonzero(y_labels == y_predictions) 
    accuracy = float(matches / len(y_labels)) * 100
    return accuracy

def find_best_K_val(y_train, y_labels, X_train, x_labels):
    """
    Returns the class of the most common neighbour from a list of tuples with distances and labels.

            Parameters:
                    distancesAndLabels (list(tuple)):   A numpy 2d-array
                    k (int):                            An integer

            Returns:
                    most_common_class (str):    String containing the class of the most frequent neighbour. 
    """ 
    best_k = 0
    best_score = 0
    epochs = int(len(X_train)/3)
    for k in range(1, int(len(X_train)/3)):                                                 #X_train / 3 so it will evaluate k values for 1 ... 121 to see which yieds best results
        predictions = np.array([ predict(k, d, X_train, x_labels) for d in y_train])
        accuracy = evaluate(predictions, y_labels=y_labels)
        print(f"Training... ({k}/{epochs}) - k:{k}, accuracy: {accuracy}")
        if( accuracy > best_score ):
            best_score = accuracy
            best_k = k
    print(f"De beste k is: {best_k} met een accuracy van: {best_score}%")
    return best_k



def main():
    d_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

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

    v_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    validates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
    v_labels = []
    for label in validates:
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

    k = find_best_K_val(v_data, v_labels, d_data, d_labels)

    days = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    predictions = np.array([ predict(k, d, d_data, d_labels) for d in days])
    print("Voorspellingen: ", predictions)   

if(__name__ == '__main__'):
    main() 