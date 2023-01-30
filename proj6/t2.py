import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as sc
from numba import njit, jit


# @jit()
def Hamiltonian(N: int):
    """Function to calculate the hamiltonian given a specific spin chain length N

    Args:
        N (int): number of spins in chain

    Returns:
        np.ndarray: hamiltonian of the system
    """

    H = np.zeros((2**N, 2**N))

    for a in range(2**N):
        for lam in range(0, N - 1):
            j = (lam + 1) % N
            bit = format(a, "#0%db" % (N + 2))[2:]
            if bit[lam] == bit[j]:
                H[a, a] += 1 / 4
            else:
                H[a, a] -= 1 / 4
                b = np.bitwise_xor(a, 2 ** (N - 1 - lam) + 2 ** (N - 1 - j))
                H[a, b] = 1 / 2

    return H


def mz(a: np.ndarray, N: int):
    """Magnetisation of a state a (integer representation)

    Args:
        a (np.ndarray): state (Linear combination of multiple states, hence array)
        N (int): number of spins in chain

    Returns:
        int: magnetisation
    """
    for c in range(len(a)):
        if c > 1e-3:
            c = format(c, "#0%db" % (N + 2))[2:]
            m = 0
            for i in c:
                if i == "1":
                    m += 1
                else:
                    m -= 1
            break
    return m * 0.5


@njit()
def Z(eig: np.ndarray, T: float):
    """Partition sum of diagonalised problem

    Args:
        eig (np.ndarray): eigenvalues
        T (float): temperature

    Returns:
        float: Partition sum
    """
    beta = 1 / T
    return np.sum(np.exp(-beta * eig))


@njit()
def C(eig: np.ndarray, T: float):
    """Specific hear

    Args:
        eig (np.ndarray): Eigenenergies
        T (float): Temperature

    Returns:
        float: specific heat
    """

    beta = 1 / T
    partition_sum = Z(eig, T)
    mean_E = np.dot(eig, np.exp(-beta * eig)) / partition_sum
    var_E = np.dot(eig**2, np.exp(-beta * eig)) / partition_sum

    return beta**2 * (var_E - mean_E**2)


def magnetisation(H: np.ndarray, T: float, N: int):
    """average magnetisation of the system

    Args:
        H (np.ndarray): Hamiltonian
        T (float): Temperature
        N (int): Number of spins

    Returns:
        Float: Average value of magnetisation of the spin chain
    """
    eig_E, eig_v = np.linalg.eig(H)

    # need to transpose it as it otherwise is hard to handle
    eig_v = eig_v.T

    # make array with the respective magnetisations in it
    m = np.zeros_like(eig_E)
    for i in range(len(eig_v)):
        m[i] = mz(eig_v[i], N)

    beta = 1 / T
    partition_sum = Z(eig_E, T)
    mean_mz = np.dot(m, np.exp(-beta * eig_E)) / partition_sum
    var_mz = np.dot(m**2, np.exp(-beta * eig_E)) / partition_sum

    return beta * (var_mz - mean_mz**2)


def correlation(H: np.ndarray, T: float, N: int):
    """Function to calculate the spin-spin correlation of the 0th and 2nd spin of the chain (i.e. first and third)

    Args:
        H (np.ndarray): Hamiltonian
        T (float): Temperature
        N (int): Number of spins

    Returns:
        Float: Correlation of first and third spin in the chain
    """
    beta = 1 / T
    eig_E, eig_v = np.linalg.eig(H)
    partition_sum = Z(eig_E, T)
    # need to transpose it as it otherwise is hard to handle
    eig_v = eig_v.T

    corr = np.zeros_like(eig_E)
    c = 0
    for i in range(len(eig_v)):
        for j, val in enumerate(eig_v[i]):
            chain = format(j, "#0%db" % (N + 2))[2:]
            corr[i] += val**2 * (int(chain[-1]) - 0.5) * (int(chain[-3]) - 0.5)
        c += corr[i] * np.exp(-beta * eig_E[i])
    return 3 * c / Z


print(Hamiltonian(2))

a = np.array([0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0])
print(mz(a, 2))

for i in range(2, 7):
    H2 = Hamiltonian(i)
    start = time.time()
    l2, u2 = np.linalg.eig(H2)
    end = time.time()
    print(
        "#########################################################################################\n"
    )
    j = np.argmin(l2)
    print("For N=%d the lowest eigenvalue is E_0 = %f\n" % (i, l2[j]))
    print("and the corresponding eigenvector is:\n", u2.T[j], "\n")
    print("which has magnetisation %.1f\n" % mz(u2.T[j], i))
    print("Time for diagonalisation for N=%d is %f seconds" % (i, end - start))

# timing = []
# for i in range(2, 13):
#     H2 = Hamiltonian(i)
#     start = time.time()
#     l2, u2 = np.linalg.eig(H2)
#     end = time.time()
#     timing.append([2**i, end - start])
# timing = np.asarray(timing)
# print(timing)

# fig = plt.figure()
# plt.plot(timing[:, 0], timing[:, 1])
# plt.xlabel(r"$N$")
# plt.ylabel(r"Diagonalisation time in [s]")
# plt.grid()
# plt.yscale("log")

# plt.show()

temp = np.array([0.1, 1, 10, 100])
N = np.array([4, 6, 8, 10, 12])
