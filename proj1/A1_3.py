import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import random

def neighbours(x, y, lattice):
    n = []
    if x+1 < 31 and lattice[x+1, y] == 0:
        n.append([x+1, y])
    
    if x-1 > -1 and lattice[x-1, y] == 0:
        n.append([x-1, y])
    
    if y+1 < 31 and lattice[x, y+1] == 0:
        n.append([x, y+1])
    
    if y-1 > -1 and lattice[x, y-1] == 0:
        n.append([x, y-1])
    
    return n

def neighboursFree(x, y):
    n = []
    if x+1 < 31:
        n.append([x+1, y])
    
    if x-1 > -1:
        n.append([x-1, y])
    
    if y+1 < 31:
        n.append([x, y+1])
    
    if y-1 > -1:
        n.append([x, y-1])
    
    return n



def visual(l, p, d=31):
    fig = plt.figure()
    x = np.array([np.arange(1, d+1) for i in range(31)]).T
    y = x.T
    plt.scatter(x, y, c=l)
    for i in range(len(p)-1):
        x = [p[i][0]+1, p[i+1][0]+1]
        y = [p[i][1]+1, p[i+1][1]+1]
        plt.plot(x, y, "r-")
    plt.show()

def energy(lattice, path, d=31, eps=1):
    count = 0

    for i in range(1, len(path)-1):
        x, y = path[i]
        r = neighboursFree(x, y)
        if lattice[x, y] == 2:
            for j in range(len(r)):
                if lattice[r[j][0], r[j][1]] == 2 and r[j] != path[i+1] and r[j] != path[i-1]:
                    count += 1

    x, y = path[0]
    r = neighboursFree(x, y)
    if lattice[x, y] == 2:
        for j in range(len(r)):
            if lattice[r[j][0], r[j][1]] == 2 and r[j] != path[1]:
                count += 1

    x, y = path[-1]
    r = neighboursFree(x, y)
    if lattice[x, y] == 2:
        for j in range(len(r)):
            if lattice[r[j][0], r[j][1]] == 2 and r[j] != path[-2]:
                count += 1
    
    count = int(count/2)

    return -eps*count

def energy2(lattice, path, d=31, eps=1):
    count = 0
    countpath = 0

    for i in range(len(path)-1):
        x, y = path[i]
        x1, y1 = path[i+1]
        if (lattice[x, y] == 2 and lattice[x1, y1]==2):
            countpath+=1
    
    for i in path:
        x, y = i
        if lattice[x, y] == 2:
            r = neighboursFree(x, y)
            for j in r:
                if lattice[j[0], j[1]]==2:
                    count += 1
    
    f = int(count/2) - countpath

    return -eps * f


# dimensions of lattice
d = 31

#probs
ph = 0.7
pp = 0.3

#Nr of iterations
M = 1000

# energy
epsilon = 1





E = []
L = []

for i in range(M):
    lattice = np.array([np.zeros(d) for i in range(d)])
    path = []
    r = np.random.uniform(0, 1, 1)
    if r < ph:
        lattice[15, 15] = 2
    else:
        lattice[15, 15] = 1

    path.append([15, 15])

    while len(neighbours(path[-1][0], path[-1][1], lattice)) != 0:
        neig = neighbours(path[-1][0], path[-1][1], lattice)
        r = np.random.randint(len(neig))
        x, y = neig[r]
        path.append([x, y])
        r = np.random.uniform(0, 1, 1)
        if r < ph:
            lattice[x, y] = 2
        else:
            lattice[x, y] = 1

        if len(neighbours(x, y, lattice)) == 0:
            E.append(energy(lattice, path))
            L.append(len(path))


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
plt.tight_layout()
ax[0].hist(E, bins=30, density=True)
ax[0].set_xlabel(r"$E$ in $\epsilon$")
ax[0].set_ylabel(r"$p(E)$")

ax[1].hist(L, bins=30, density=True)
ax[1].set_xlabel(r"$L$")
ax[1].set_ylabel(r"$p(L)$")

plt.show()


fig2, ax1= plt.subplots()
h = ax1.hist2d(E, L, bins=30)
fig.colorbar(h[3], ax=ax1)
ax1.set_xlabel(r"$E$ in $\epsilon$")
ax1.set_ylabel(r"$L$")
plt.show()
#visual(lattice, path, 31)