import numpy as np
import math


softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])    #drie klassen

class_targets = [0, 1, 1]           #label 0 in dit geval

neg_log = -np.log(softmax_output[                                   #print losses
    range(len(softmax_output)), class_targets               
])

avg_loss = np.mean(neg_log)
print(avg_loss)



for targ_idx, distribution in zip(class_targets, softmax_output):
    print(distribution[targ_idx])

