import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as sc
from numba import njit, jit


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
        emin = lanczos(H, len(H), len(H))
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

        h[i, i-1] = np.sqrt(b[i-1])
        h[i-1, i] = np.sqrt(b[i-1])
        h[i, i] = a[i]

        eigE, eigv = np.linalg.eig(h)

        if np.abs(e0 - min(eigE)) < 1e-6:
            e0 = min(eigE)
            break
        else:
            e0 = min(eigE)

        # print(b)

    return e0


for i in range(2, 7):
    t = timing(i)
    print("For n = %d get E0 = %4.f taking a total of t = %g seconds" %
          (i, t[0], t[1]))
