import matplotlib.pyplot as plt
import numpy as np
import random

R1 = np.zeros((10000,500))
R2 = np.zeros((10000,500))

epsilon = 0.1
alpha = 0.1

for n in range(500):
    N = np.zeros((1, 10))
    q = np.zeros((1, 10))
    Q1 = np.zeros((1,10))
    Q2 = np.zeros((1,10))
    for i in range(1, 10000):
        num1 = random.random()/epsilon
        num2 = random.random()/epsilon
        if num1 < 1:
            A1 = np.random.randint(0, 10)
        else:
            A1 = np.argmax(Q1)
        if num2 < 1:
            A2 = np.random.randint(0, 10)
        else:
            A2 = np.argmax(Q2)
        r1 = q[0, A1] + np.random.randn(1)
        r2 = q[0, A2] + np.random.randn(1)
        N[0, A2] += 1 
        Q1[0, A1] += ( r1 - Q1[0, A1])*alpha
        Q2[0, A2] += ( r2 - Q2[0, A2]) / N[0, A2]
        q += np.random.randn(1, 10)/100
        R1[i, n] = r1
        R2[i, n] = r2
        
plt.plot(np.average(R1, axis = 1), color= 'r')
plt.plot(np.average(R2, axis = 1), color= 'b')
plt.show()
