import math


softmax_output = [0.7, 0.1, 0.2]    #drie klassen
target_output = [1, 0, 0]           #label 0 in dit geval

loss = -(math.log(softmax_output[0]) * target_output[0] +           
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

loss = -(math.log(softmax_output[0]))       #categorical cross entropy, dit is gewoon -log eigenlijk met natural log base E, een van de meest populaire loss functions
loss = -(math.log(0.5))

print(loss)