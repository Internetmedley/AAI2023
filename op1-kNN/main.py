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

def compute_distance():
    return

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
        if label < 20000301:
            v_labels.append('winter')
        elif 20000301 <= label < 20000601:
            v_labels.append('lente')
        elif 20000601 <= label < 20000901:
            v_labels.append('zomer')
        elif 20000901 <= label < 20001201:
            v_labels.append('herfst')
        else: # from 01-12 to end of year
            v_labels.append('winter')


    # The calculation of the Euclidean distance is a straight-forward application of the law of Pythagoras:
    # the distance d between points (a1, . . . , an) and (b1, . . . , bn) is d2 = (a1 − b1)2 + . . . + (an − bn)2.    
    allDistancesAndLabels = []         #verzameling van alle distances en labels
    for a in range(len(validata)):      
        distances = []
        distancesAndLabelsDict = { "winter" : [], "lente" : [], "zomer" : [], "herfst" : [] }      #deze gedeclareerd omdat het dan makkelijk appenden is
        for b in range(len(data)):          
            distance = 0
            for i in range(len(data[b])):       
                distance += np.square(validata[a][i] - data[b][i])   

            #hier moet de opgeslagen distance worden verwerkt door het op te slaan in een lijst 
            distances.append(np.sqrt(distance))                                 #sqrt want pythagoras
            distancesAndLabelsDict[d_labels[b]] = distances                     #pak het label van het andere punt van de vergelijking en zet de distances erbij

        #zet de dicts in een lijst om zo een lijst van 100 elementen te krijgen met dicts mey key=seizoen en value=distance
        #als de distance als key gebruikt zou worden kunnen er niet twee dezelfde distances zijn, want een dict heeft unique keys
        allDistancesAndLabels.append(distancesAndLabelsDict)
    #print(len(allDistancesAndLabels[50]["winter"]))


    #Select k closest instances
    k = 5

    #First sort the list of all distances and labels
    for i in allDistancesAndLabels:
        for key, value in i.items():
            sorted = np.sort(value)
            i[key] = sorted
    
    print(allDistancesAndLabels[50]["zomer"])




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