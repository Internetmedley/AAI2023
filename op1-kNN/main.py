import numpy as np

# The data that we are using has 11 attributes:
#     YYYYMMDD: date in year, months, days;
#     FG: day average windspeed (in 0.1 m/s);
#     TG: day average temperature (in 0.1 degrees Celsius);
#     TN: minimum temperature of day (in 0.1 degrees Celsius);
#     TX: maximum temperature of day (in 0.1 degrees Celsius);
#     SQ: amount of sunshine that day (in 0.1 hours); -1 for less than 0.05 hours;
#     DR: total time of precipitation (in 0.1 hours);
#     RH: total sum of precipitation that day (in 0.1 mm); -1 for less than 0.05mm.

def comp_distances(a,  b, d_labels):
    """Calculates the distance between  
    """
    distances = []
    #for i in range(len(a)):          
    for i in range(len(b)):
        distance = 0
        for j in range(len(b[0])):       
            distance += np.square(a[j] - b[i][j])   
        distances.append(tuple([np.sqrt(distance), d_labels[i]]))                               #sqrt want pythagoras
    return distances

def find_most_common_neighbour_K(distancesAndLabels, labels, k): 
    """Returns most common neighbour
    :param distancesAndLablels: a list of tuples containing float distance[0] and string label[0] 
    """                        
    most_common_label = ''

    k_closest = np.array([ v[1] for v in distancesAndLabels[:k] ])                       #get the k closest neighbours for this distance
    frequencies = []
    while(True):
        for l in labels:
            frequencies.append(tuple((np.count_nonzero(k_closest == l), l)))
            print(f"dit is l:{l}, dit is k_closest:{k_closest} en samen:", np.count_nonzero(k_closest == l))
        
        frequencies.sort(key=lambda x : x[0], reverse=True)             #sort the elements descending

        if(frequencies[0][0] == frequencies[1][0]):                     #if true, there is a conflict meaning the highest frequent neighbours are the same
            k_closest = k_closest[:-1]                                  #leave out the last neighbour if there is a conflict and then repeat the process
            frequencies = []                                            #because we're sorting beforehand, comparing the first two elements always works  
            continue
        else:
            break
        most_common_label = frequencies[0][1]
    return most_common_label

def find_best_K(validata, v_labels, data, d_labels, labels):
    #allDistancesAndLabels = []         #verzameling van alle distances en labels
    cv_scores = {}
    for a in range(len(validata)):      
        distances = comp_distances(validata[a], data, d_labels)
        distances.sort(key=lambda x : x[0])                 #sort based on first element, which is 
        #allDistancesAndLabels.append(distances)

        
        k_closest = []
        n_correct = 0
        best_k = 0
        best_score = 0
        for k in range(1, int(len(data/3))):
            most_common_label = find_most_common_neighbour_K(distances, labels, k)
            if(most_common_label == v_labels[a]):
                n_correct += 1
        if(n_correct > best_score):
            best_score = n_correct
            best_k = k
        cv_scores[k] = best_score / len(v_labels)
    
    print("beste k:", best_k, "beste score:", best_score)
    best_k = max(cv_scores, key=cv_scores.get)
    best_score = cv_scores[best_k]
    
    for k, score in cv_scores.items():
        print(f"k={k}: Mean Cross validation score = {score}")

    print(f"\nBest k value: {best_k} with mean cross validation score = {best_score}")

    return best_k

#def find_most_common_neighbour(distances, v_labels, data, labels):
    # cv_scores = {}
    # best_k = 0
    # best_score = 0
    # scores = []
    # for k in range(1, int(len(data)/3)):                                            #it is usually not a good idea to go higher than 1/3 of train data for k
    #     k_closest = []                                                                  #reset closest neighbours
    #     n_correct = 0
    #     for d in range(len(distances)):
    #         k_closest = np.array([ v[1] for v in distances[d][:k] ])                       #get the k closest neighbours for this distance
    #         frequencies = []
    #         while(True):
    #             for l in labels:
    #                 frequencies.append(tuple((np.count_nonzero(k_closest == l), l)))
    #             frequencies.sort(key=lambda x : x[0], reverse=True)             #sort the elements descending

    #             if(frequencies[0][0] == frequencies[1][0]):                     #if true, there is a conflict meaning the highest frequent neighbours are the same
    #                 k_closest = k_closest[:-1]                                  #leave out the last neighbour if there is a conflict and then repeat the process
    #                 frequencies = []                                            #because we're sorting beforehand, comparing the first two elements always works  
    #                 continue
    #             else:
    #                 break

    #         most_common_label = frequencies[0][1]
    #         if(most_common_label == v_labels[d]):
    #             n_correct += 1
    #     if(n_correct > best_score):
    #         best_score = n_correct
    #         best_k = k

    #     cv_scores[k] = best_score / len(v_labels)
    
    # print("beste k:", best_k, "beste score:", best_score)
    # best_k = max(cv_scores, key=cv_scores.get)
    # best_score = cv_scores[best_k]
    
    # for k, score in cv_scores.items():
    #     print(f"k={k}: Mean Cross validation score = {score}")

    # print(f"\nBest k value: {best_k} with mean cross validation score = {best_score}")
    # return k

def predict(test_data, data, k, d_labels, labels):
    distances = comp_distances(test_data, data, d_labels)
    
    most_common_label = find_most_common_neighbour_K(distances, labels, k)
    print(f"Precicted label: {most_common_label}")
    return most_common_label

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


    # The calculation of the Euclidean distance is a straight-forward application of the law of Pythagoras:
    # the distance d between points (a1, . . . , an) and (b1, . . . , bn) is d2 = (a1 − b1)2 + . . . + (an − bn)2.    
   

    labels=['winter', 'lente', 'zomer', 'herfst']
    best_k = find_best_K(validata, v_labels, data, d_labels, labels)

    print(best_k)

    days = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    [ predict(day, data, best_k, d_labels, labels) for day in days ]

if(__name__ == '__main__'):
    main() 