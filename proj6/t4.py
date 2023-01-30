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
        emin = lanczos(H, len(H), len(H))[0]
        if emin < minE:
            minE = emin
    end = time.time()
    return minE, end-start


def lanczos(H: np.ndarray, lam: int, M: int):
    """Lanczos method

    Args:
        H (np.ndarray): (block) matrix
        lam (int): number of iterations
        M (int): Size of matrix
    """
    a = []
    b = []
    N = []

    v = np.zeros((lam+1, M))
    v[0, :] = np.zeros(M)
    v[0, 0] = 1
    w = np.dot(H, v[0, :])
    N0 = np.linalg.norm(v[0, :])**2
    H00 = np.dot(w, v[0, :])
    a.append(H00 / N0)
    h = np.zeros((lam, lam))
    h[0, 0] = a[0]
    e0 = a[0]
    N.append(N0)

    v[1, :] = w - a[0] * v[0, :]
    for i in range(1, lam-1):
        w = np.dot(H, v[i, :])
        Hii = np.dot(w, v[i, :])

        Nii = np.linalg.norm(v[i, :])**2
        # print(Nii)
        a.append(Hii/Nii)
        b.append(Nii/N[-1])
        N.append(Nii)
        v[i+1, :] = w - a[-1] * v[i, :] - b[-1] * v[i-1, :]

        h[i, i-1] = np.sqrt(b[-1])
        h[i-1, i] = np.sqrt(b[-1])
        h[i, i] = a[-1]

        eigE, eigv = np.linalg.eig(h)

        if np.abs(e0 - min(eigE)) < 1e-6:
            e0 = min(eigE)
            break
        else:
            e0 = min(eigE)
    # print(e0)
    # print(b)
    if M > 4:
        return e0, eigE[:4]
    else:
        return e0, 0


def timing2(N):
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
    t = timing(i)
    print("For n = %d get E0 = %.4f taking a total of t = %g seconds" %
          (i, t[0], t[1]))

diag_time_lan = []
diag_time_no = []
diag_time = []
N = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
for i in N:
    diag_time_lan.append(timing(i)[1])
    diag_time.append(timing2(i)[1])
    start = time.time()
    H2 = Hamiltonian(i)
    l2, u2 = np.linalg.eig(H2)
    end = time.time()
    diag_time_no.append(end - start)

f = plt.figure()
plt.plot(N, diag_time, label="Block diag.")
plt.plot(N, diag_time_lan, label="Block diag. with Lanczos")
plt.plot(N, diag_time_no, label="No block diag.")
plt.xlabel(r"$N$")
plt.ylabel("Diagonalisation Time in [s]")
plt.legend()
plt.grid()
plt.yscale("log")
plt.savefig("Diagonalisation_time_block_lanc.pdf", dpi=200)


B = block(10, 5)
M = len(B)
lam = []
E0 = []
E1 = []
E2 = []
E3 = []

for i in range(5, M):
    E = lanczos(B, i, M)[1]
    lam.append(i)
    E0.append(E[0])
    E1.append(E[1])
    E2.append(E[2])
    E3.append(E[3])

f2 = plt.figure()
plt.plot(lam, E0, label=r"$E_0$")
plt.plot(lam, E1, label=r"$E_1$")
plt.plot(lam, E2, label=r"$E_2$")
plt.plot(lam, E3, label=r"$E_3$")
plt.legend()
plt.grid()
plt.xlabel(r"$\Lambda$")
plt.ylabel(r"Energies")
plt.savefig("Lanczos_please.pdf", dpi=200)
