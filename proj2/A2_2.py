import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def dE(spins, i, j, J=1, N=30):
    """Energy difference in proposed jump

    Args:
        spins (array): spin configuration on square lattice
        i (_type_): x index
        j (_type_): y index
        J (int, optional): coupling constant. Defaults to 1.
        N (int, optional): dimension of lattice. Defaults to 30.

    Returns:
        float: energy difference
    """
    
    e = J*2 * spins[i, j] * (spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N])

    return e

def E(spins, J=1, N=30):
    """Energy of configuration

    Args:
        spins (array): spin configuration on square lattice
        J (int, optional): coupling constant. Defaults to 1.
        N (int, optional): dimension of lattice. Defaults to 30.

    Returns:
        float: energy of configuration
    """
    e = 0

    for i in range(N):
        for j in range(N):
            e += spins[i, j] * (spins[(i+1)%N, j] + spins[i, (j+1)%N])
    return -J*e

def magnetisation(spins):
    return np.sum(spins)

def run(n=30, T=1, L=5000):

    lattice = np.array([np.ones(n) for i in range(n)])

    E0 = E(lattice)
    M0 = magnetisation(lattice)

    E = []
    M = []

    E.append(E0)
    M.append(M0)

    for i in range(L):
        x = np.random.randint(0, n, 1)
        y = np.random.randint(0, n, 1)

        DE = dE(lattice, x, y)

        if DE <= 0:
            lattice[x, y] *= -1
            E.append(E[-1]+DE)
            M.append(M[-1] + 2 * lattice[x, y])
        elif np.random.rand() < np.exp(-1/T * DE):
            lattice[x, y] *= -1
            E.append(E[-1]+DE)
            M.append(M[-1] + 2 * lattice[x, y])
        else:
            E.append(E[-1])
            M.append(M[-1])
