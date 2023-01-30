import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as sc
from numba import njit, jit


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


def Nup(a: int, N: int):
    a = format(a, "#0%db" % (N + 2))[2:]
    m = 0
    for i in a:
        if i == "1":
            m += 1
    return m


def block(N: int, nup: int):
    a = 0
    sa = []
    for i in range(2**N):
        if Nup(i, N) == nup:
            a += 1
            sa.append(i)
    M = a

    H = np.zeros((M, M))

    for a in range(M):
        for i in range(N-1):
            j = (i + 1) % N
            bit = format(sa[a], "#0%db" % (N + 2))[2:]
            if bit[i] == bit[j]:
                H[a, a] += 1 / 4
            else:
                H[a, a] -= 1 / 4
                s = np.bitwise_xor(sa[a], 2 ** (N - 1 - i) + 2 ** (N - 1 - j))
                b = np.argwhere(sa == s)[0, 0]
                H[a, b] = 1 / 2
    return H


def timing(N):
    minE = 0

    start = time.time()
    for i in range(N+1):
        H = block(N, i)
        l2, u2 = np.linalg.eig(H)
        if min(l2) < minE:
            minE = min(l2)
    end = time.time()
    return minE, end-start


for i in range(2, 7):

    print(timing(i))

diag_time_no = []
diag_time = []
N = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
for i in N:
    diag_time.append(timing(i)[1])
    start = time.time()
    H2 = Hamiltonian(i)
    l2, u2 = np.linalg.eig(H2)
    end = time.time()
    diag_time_no.append(end - start)


f = plt.figure()
plt.plot(N, diag_time, label="Block diag.")
plt.plot(N, diag_time_no, label="No block diag.")
plt.xlabel(r"$N$")
plt.ylabel("Diagonalisation Time in [s]")
plt.legend()
plt.grid()
plt.yscale("log")
plt.savefig("Diagonalisation_time_block.pdf", dpi=200)
