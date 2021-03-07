import matplotlib.pyplot as plt
import numpy as np
import random

## 1 - epsilon greedy, constant step
## 2 - epsilon greedy, sample average
## 3 - upper confidence bound
## 4 - gradient bandit

R1 = np.zeros((100000,10))
R2 = np.zeros((100000,10))
R3 = np.zeros((100000,10))
R4 = np.zeros((100000,10))

c = 0
alpha = 0.1

for n in range(10):
    epsilon = 2**(n - 7)
    N2 = np.zeros((1, 10))
    N3 = np.zeros((1, 10))
    N4 = np.zeros((1, 10))
    q = np.zeros((1, 10))
    Q1 = np.zeros((1,10))
    Q2 = np.zeros((1,10))
    Q3 = np.zeros((1,10))
    avg4 = np.zeros((1,10))
    H = np.zeros((1,10))
    for i in range(1, 200000):
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

        doBCD = True
        for f in range(10):
            if N3[0,f]==0:
                A3 = f
                doBCD = False
                break
        if doBCD:
            A3 = np.argmax(Q3 + epsilon * np.sqrt(np.log(i)/N3))

        pri = np.exp(H)/np.sum(np.exp(H))
        A4 = np.argmax(pri)

        r1 = q[0, A1] + np.random.randn(1)
        r2 = q[0, A2] + np.random.randn(1)
        r3 = q[0, A3] + np.random.randn(1)
        r4 = q[0, A4] + np.random.randn(1)
        N2[0, A2] += 1
        N3[0, A3] += 1
        N4[0, A4] += 1
        Q1[0, A1] += ( r1 - Q1[0, A1]) * alpha
        Q2[0, A2] += ( r2 - Q2[0, A2]) / N2[0, A2]
        Q3[0, A3] += ( r3 - Q3[0, A3]) / N3[0, A3]
        H = H - epsilon*(r4 - avg4)*pri
        H[0,A4] += epsilon*(r4 - avg4[0,A4])
        avg4[0, A4] += ( r4 - avg4[0, A4]) / N4[0, A4]
        q += np.random.randn(1, 10)/100
        if i >= 100000:
            R1[i-100000, n] = r1
            R2[i-100000, n] = r2
            R3[i-100000, n] = r3
            R4[i-100000, n] = r4
        
plt.plot(np.average(R1, axis = 0), color= 'r')
plt.plot(np.average(R2, axis = 0), color= 'g')
plt.plot(np.average(R3, axis = 0), color= 'b')
plt.plot(np.average(R4, axis = 0), color= 'y')
plt.show()
