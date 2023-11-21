import numpy as np
from scipy.stats import mode

# The data that we are using has 11 attributes:
#     YYYYMMDD: date in year, months, days;
#     FG: day average windspeed (in 0.1 m/s);
#     TG: day average temperature (in 0.1 degrees Celsius);
#     TN: minimum temperature of day (in 0.1 degrees Celsius);
#     TX: maximum temperature of day (in 0.1 degrees Celsius);
#     SQ: amount of sunshine that day (in 0.1 hours); -1 for less than 0.05 hours;
#     DR: total time of precipitation (in 0.1 hours);
#     RH: total sum of precipitation that day (in 0.1 mm); -1 for less than 0.05mm.


cv_scores = {}
def find_best_K(distances, v_data, v_labels, d_labels, labels):
    best_k = 0
    scores = []
    for k in range(1, int(len(d_labels)/3)):                                            #it is usually not a good idea to go higher than 1/3 of train data for k
        k_closest = []                                                                  #reset closest neighbours
        n_correct = 0
        for d in range(len(distances)):
            k_closest = np.array([ v[1] for v in distances[d][:k] ])                       #get the k closest neighbours for this distance
            frequencies = []
            while(True):
                for l in labels:
                    frequencies.append(tuple((np.count_nonzero(k_closest == l), l)))
                frequencies.sort(key=lambda x : x[0], reverse=True)             #sort the elements descending
                print(frequencies)
                
                #print(frequencies[0][0] == frequencies[1][0])
                if(frequencies[0][0] == frequencies[1][0]):                     #if true, there is a conflict meaning the highest frequent neighbours are the same
                    k_closest = k_closest[:-1]                                  #leave out the last neighbour if there is a conflict and then repeat the process
                    frequencies = []
                                                                                #because we're sorting beforehand, comparing the first two elements always works
                    continue
                    
                else:
                    break
            
            most_common_label = frequencies[0][1]
            
            print(most_common_label == v_labels[d], "d: ", d, "k :", k) 

            
            if(most_common_label == v_labels[d]):
                scores.append(1)
            else:
                scores.append(0)
        cv_scores[k] = np.mean(scores)
    #print("beste k:", best_k, "beste score:", best_score)
    best_k = max(cv_scores, key=cv_scores.get)
    best_score = cv_scores[best_k]
    
    for k, score in cv_scores.items():
        print(f"k={k}: Mean Cross validation score = {score}")

    print(f"\nBest k value: {best_k} with mean cross validation score = {best_score}")


    return k


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


    # The calculation of the Euclidean distance is a straight-forward application of the law of Pythagoras:
    # the distance d between points (a1, . . . , an) and (b1, . . . , bn) is d2 = (a1 − b1)2 + . . . + (an − bn)2.    
    allDistancesAndLabels = []         #verzameling van alle distances en labels
    for a in range(len(validata)):      
        distances = []
        for b in range(len(data)):          
            distance = 0
            
            #linalg = np.linalg.norm(validata - data[b], axis=1)
            # for d in linalg:
            #     distances.append(tuple((linalg, d_labels[b]))) 

            for i in range(len(data[b])):       
                distance += np.square(validata[a][i] - data[b][i])   

            distances.append(tuple([np.sqrt(distance), d_labels[b]]))                               #sqrt want pythagoras

        distances.sort(key=lambda x : x[0])                 #sort based on first element, which is 
        allDistancesAndLabels.append(distances)

    
    #distances = np.linalg.norm(validata - d, axis=1)


    find_best_K(allDistancesAndLabels, data, v_labels, d_labels, labels=["winter", "lente", "zomer", "herfst"])


    # Algorithm 3.1 — k-Nearest Neighbours.
    # Given:
    # • Training set X of examples (~xi, yi) where
    # – ~xi is feature vector of example i; and
    # – yi is class label of example i.
    # • Feature vector ~x of test point that we want to classify.
    # Do:
    # 1. Compute distance D(~x,~xi);
    # 2. Select k closest instances ~x j1 , . . . , ~x jk with class labels y j1 , . . . , y jk ;
    # 3. Output class y∗, which is most frequent in y j1 , . . . , y jk 




    days = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})


if(__name__ == '__main__'):
    main() 