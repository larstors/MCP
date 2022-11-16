import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from numba import njit
from joblib import Parallel, delayed

#@njit
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

#@njit
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

#@njit
def magnetisation(spins):
    return np.sum(spins)

#@njit
def run(n=30, T=1, L=5000, l=1):

    lattice = np.ones((n, n))

    E0 = E(lattice)
    M0 = magnetisation(lattice)

    Energy = []
    M = []

    Energy.append(E0)
    M.append(M0)

    for i in range(L):
        x = np.random.randint(0, n, 1)
        y = np.random.randint(0, n, 1)

        DE = dE(lattice, x, y)

        if DE <= 0:
            lattice[x, y] *= -1
            Energy.append(Energy[-1]+DE)
            M.append(M[-1] + 2 * lattice[x, y])
        elif np.random.rand() < np.exp(-1/T * DE):
            lattice[x, y] *= -1
            Energy.append(Energy[-1]+DE)
            M.append(M[-1] + 2 * lattice[x, y])
        else:
            Energy.append(Energy[-1])
            M.append(M[-1])

        # if i == 0 or i == L-1:
        #     plt.scatter(x=np.array([np.arange(0, n) for i in range(n)]), y=np.array([np.arange(0, n) for i in range(n)]).T, vmin=0, vmax=2, c=lattice+1)
        #     plt.show()
    return np.mean(M) #Energy, M

def analytical_m(T, Tc=2 / np.log(1 + np.sqrt(2)), J=1):
    z = np.exp(-2*J/T)
    if T < Tc:
        return ((1 + z**2)**(1/4) * (1 - 6*z**2 + z**4)**(1/8))/np.sqrt(1 - z**2)
    else:
        return 0

#nr of spins    
N = 30

#burn in
tburn = 1000000

total_time = tburn + 1000000

M = 100

T = np.linspace(1, 3, 40)

magnet = np.zeros(len(T))



magnet_per_spin = Parallel(n_jobs=6)(delayed(run)(T=t, L=total_time, l=m, n=N) for t in T for m in range(M))

for i in range(len(T)):
    magnet[i] = np.mean(magnet_per_spin[i*M:i*M+M])/(N**2)

m_anal = np.vectorize(analytical_m)

t = np.linspace(1, 3, 100)

plt.title(r"Magnetisation per spin for $%d$x$%d$ spins" % (N, N))
plt.plot(t, m_anal(t), "bx", label="Analytical result")
plt.plot(T, magnet, "-r", label="Simulation")
plt.xlabel(r"$k_B T$")
plt.ylabel(r"$m(T)$")
plt.grid()
plt.legend()
plt.savefig("2d_magnetisation.pdf", dpi=200)
#plt.show()
