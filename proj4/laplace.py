import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import ctypes

# library = ctypes.CDLL("./libfun.so")
# ar = [0, 1, 2, 3]
# arr = (ctypes.c_int * len(ar))(*ar)
# print(ar, arr)




@jit(fastmath=True)
def jacobi(A, x, b):
    """quick and dirty solver for single iteration of jacobi method

    Args:
        A (array): matrix
        x (array): current guess
        b (array): rhs

    Returns:
        array: improvement
    """
    new_x = b
    for i in range(len(new_x)):
        new_x[i] -= np.sum(A[i, :]*x[:]) - A[i,i] * x[i]
        new_x[i] /= A[i,i]

    return new_x

@jit(fastmath=True)
def GS(A, x, b):
    """quick and dirty solver for single iteration of Gauss Seidel method

    Args:
        A (array): matrix
        x (array): current guess
        b (array): rhs

    Returns:
        array: improvement
    """
    new_x = b
    N = len(new_x)
    for i in range(N):
        for j in range(i):
            new_x[i] -= A[i, j] * new_x[j]
        for j in range(i+1, N):
            new_x[i] -= A[i, j] * x[j]
        new_x[i] /= A[i,i]

    return new_x


@jit(fastmath=True)
def SOR(A, x, b, w):
    """quick and dirty solver for single iteration of SOR method

    Args:
        A (array): matrix
        x (array): current guess
        b (array): rhs
        w (double): weight

    Returns:
        array: improvement
    """
    new_x = w*b
    N = len(new_x)
    for i in range(N):
        for j in range(i):
            new_x[i] -= A[i, j] * new_x[j] + (1-w)*A[i, j] * x[j]
        for j in range(i+1, N):
            new_x[i] -= w*A[i, j] * x[j]
        new_x[i] /= A[i,i]

    return new_x


Vtop = 100
Vrst = 0
eps = 1e-3

alpha = [0.5, 1.0, 1.25, 1.5, 1.75, 1.99]
alpha_special = 2.5

@njit(fastmath=True)
def initial():
    """generating the matrix A and initial condition x

    Returns:
        array: matrix and initial condition x
    """
    # assuming square box
    L = 1
    dl = 1e-1

    # dimension
    N = int(L/dl)

    # lattice
    x = np.zeros(N*N)

    for i in range(N):
        x[i] = Vtop

    # matrix A
    A = np.zeros((N*N, N*N))

    for i in range(N*N):
        for j in range(N*N):
            if i < N or i >= N**2-N:
                if i == j:
                    A[i, j] = 1
            elif i%N == 0 or i%N == N-1:
                if j == i:
                    A[i, j] = 1
            elif  i > N and i%N > 0 and i%N < N-1 and i < N**2-N:
                if j == i:
                    A[i, j] = -4
                elif j == i + 1:
                    A[i, j] = 1
                elif j == i- 1:
                    A[i, j] = 1
                elif j == i + N:
                    A[i, j] = 1
                elif j == i - N:
                    A[i, j] = 1
    return A, x 

@njit(fastmath=True)
def distance(a):
    N = int(np.sqrt(len(a)))
    epsilon = 0
    epsilonmax = 0
    check = 0
    for i in range(N**2):
        e = 0
        if i > N and i < N**2 - N and i%N > 0 and i%N < N-1:
            e = np.abs(a[i] - 0.25 * (a[i-1] + a[i+1] + a[i-N] + a[i+N]))
            if e > epsilonmax:
                epsilonmax = e
            epsilon += e
            check += 1
    return epsilon/check, epsilonmax
                



@jit(fastmath=True)
def solver(nmax):
    init = initial()
    A = init[0]
    b = init[1]
    x0 = b
    xnew = b
    k = np.zeros(nmax)
    kmax = np.zeros(nmax)
    for n in range(nmax):
        xnew = jacobi(A, xnew, b)
        e = distance(xnew)
        k[n] = e[0]
        kmax[n] = e[1]
    

    return xnew, k, kmax




sol = solver(100)

fig1 = plt.figure()
plt.plot(np.arange(0, len(sol[2])), sol[1])
plt.title("aver_eps")
plt.show()

fig2 = plt.figure()
plt.plot(np.arange(0, len(sol[2])), sol[1])
plt.title("max_eps")
plt.show()


N = int(np.sqrt(len(sol[0])))
sol = sol[0].reshape(N, N)
dl = 1/N
x, y = np.meshgrid(np.linspace(0, 1, N, endpoint=True), np.linspace(0, 1, N, endpoint=True))

fig3 = plt.show()
plt.imshow(sol)
plt.colorbar()
plt.show()
