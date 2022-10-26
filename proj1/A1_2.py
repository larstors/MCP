from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import random

# initial conditions
x0 = 0
y0 = 0

######################### PART A ############################

def update(n=1000):
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    for i in range(n):
        dx = np.random.uniform(-1, 1, 1)
        dy = np.random.uniform(-1, 1, 1)
        L = np.sqrt(dx**2 + dy**2)
        dx = dx/L
        dy = dy/L

        x.append(x[-1]+dx)
        y.append(y[-1]+dy)
    return x, y

def coor_last_step(n):
    x = x0
    y = y0
    for i in range(n):
        dx = np.random.uniform(-1, 1, 1)
        dy = np.random.uniform(-1, 1, 1)
        L = np.sqrt(dx**2 + dy**2)
        dx = dx/L
        dy = dy/L

        x += dx
        y += dy
    return x, y

# Length of random walk
N = 1000



plt.figure()
for n in range(4):
    x = update()[0]
    y = update()[1]

    plt.plot(x, y, "-", label="Trajectory %d" % (n+1))


plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ######################## PART B ###################

# array for last coordinate and rms
last_coor = []


M = 1000
N = 10000

for i in range(M):
    x, y = update(N)
    last_coor.append(np.sqrt(x[-1]**2 + y[-1]**2))

last_coor = np.asarray(last_coor)

fig2 = plt.figure()
plt.hist(last_coor, 40)
plt.xlabel(r"$R_N$")
plt.ylabel(r"Frequency of $R_N$")
plt.grid()
plt.show()

print(np.sqrt(1/M*np.sum(last_coor*last_coor)), np.sqrt(N))
