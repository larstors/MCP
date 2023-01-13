import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import curve_fit

@njit(fastmath=True)
def V_1d(x: np.ndarray, w=1.0, m=1.0):
    """1d harmonic potential

    Args:
        x (np.ndarray): positions
        w (float, optional): frequency. Defaults to 1.0.
        m (float, optional): mass. Defaults to 1.0.

    Returns:
        float: value of potential given specific configuration x
    """
    v = 0
    for i in range(1, len(x)):
        v +=  (x[i] + x[i-1])**2 
    return m * w * 0.5 * v / 4

@njit(fastmath=True)
def T_1d(x: np.ndarray, dtau: float, m=1.0):
    """Kinetic energy of 1d system 

    Args:
        x (np.ndarray): configuration
        dtau (float): discretisation of time
        m (float, optional): mass. Defaults to 1.0.

    Returns:
        float: kinetic energy given specific configuration x
    """
    t = 0
    for i in range(1, len(x)):
        t += (x[i] - x[i-1])**2
    return 0.5 * t / (dtau ** 2)

@njit(fastmath=True)
def V_2d(x: np.ndarray, y: np.ndarray, w1: float, w2: float, m=1.0):
    """2d (in general) non-symmetric harmonic potential

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
        w1 (float): x-directin strength
        w2 (float): y-direction strength
        m (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    v = 0
    for i in range(1, len(x)):
        v += (x[i] + x[i-1])**2 * w1 ** 2
    
    # allowing for different discretisations in x and y
    for i in range(1, len(y)):
        v += (y[i] + y[i-1])**2 * w2 ** 2

    return v * m * 0.5 / 4.0

@njit(fastmath=True)
def T_2d(x: np.ndarray, y: np.ndarray, dtau: float, m=1.0):
    """Kinetic energy of 2d system

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
        dtau (float): discretisation of time
        m (float, optional): mass. Defaults to 1.0.
    """

    t = 0
    for i in range(1, len(x)):
        t +=  (x[i] - x[i-1])**2 
    # allowing for different discretisation in y direction
    for i in range(1, len(y)):
        t += (y[i] - y[i-1])**2

    return 0.5 * t/ dtau ** 2

@njit(fastmath=True)
def dS_1(i: int, x: np.ndarray, x_new: np.ndarray, dtau: float):
    """function to calculate change in S

    Args:
        i (int): index of position changed
        x (np.ndarray): old position
        x_new (np.ndarray): proposed position 
        dtau (float): discretisation in time
    """
    
    ds = ((x_new[i] - x_new[i-1])/dtau) ** 2 + ((x_new[i] - x_new[i+1])/dtau) ** 2 + ((x_new[i] + x_new[i-1])/2) ** 2 + \
        ((x_new[i] + x_new[i+1])/2) ** 2 - ((x[i] - x[i-1])/dtau) ** 2 - ((x[i] - x[i+1])/dtau) ** 2 - \
        ((x[i] + x[i-1])/2) ** 2 - ((x[i] + x[i+1])/2) ** 2
    
    return ds
@njit(fastmath=True)
def dS_2(i: int, x: np.ndarray, x_new: np.ndarray, y: np.ndarray, y_new: np.ndarray, dtau: float, w1: float, w2: float):
    """Change in S for proposed position

    Args:
        i (int): index of proposed change
        x (np.ndarray): config in x
        x_new (np.ndarray): proposed config in x
        y (np.ndarray): config in y
        y_new (np.ndarray): proposed config in y
        dtau (float): discretisation in tau
        w1 (float): strength of V in x
        w2 (float): strength of V in y

    Returns:
        float: change in S from x, y to x_new, y_new
    """

    # change in x coordinate
    ds = ((x_new[i] - x_new[i-1])/dtau) ** 2 + ((x_new[i] - x_new[i+1])/dtau) ** 2 + (w1 * (x_new[i] + x_new[i-1])/2) ** 2 + \
        (w1 * (x_new[i] + x_new[i+1])/2) ** 2 - ((x[i] - x[i-1])/dtau) ** 2 - ((x[i] - x[i+1])/dtau) ** 2 - \
        (w1 * (x[i] + x[i-1])/2) ** 2 - (w1 * (x[i] + x[i+1])/2) ** 2
    # change in y coordinate
    ds += ((y_new[i] - y_new[i-1])/dtau) ** 2 + ((y_new[i] - y_new[i+1])/dtau) ** 2 + (w2 * (y_new[i] + y_new[i-1])/2) ** 2 + \
        (w2 * (y_new[i] + y_new[i+1])/2) ** 2 - ((y[i] - y[i-1])/dtau) ** 2 - ((y[i] - y[i+1])/dtau) ** 2 - \
        (w2 * (y[i] + y[i-1])/2) ** 2 - (w2 * (y[i] + y[i+1])/2) ** 2
    
    return ds

@njit(fastmath=True)
def path(step: int, neq: int, dx: float, N: int, tau: float, dim: int, tau0 = 0.0):
    """calculation of path 

    Args:
        step (int): number of steps in the simulation
        neq (int): number of steps to let system equilibrate
        dx (float): steplength of spacial direction
        N (int): number of time steps
        tau (float): maximal time
        dim (int): dimensions of the system
        tau0 (float, optional): initial time. Defaults to 0.0.
    """
    #unnecessarily complicated way of calculating dtau
    dtau = (tau - tau0) / (N - 1)

    # initial configuration of positions
    x = np.random.uniform(0, dx, N) - 0.5 * dx
    x_new = x

    x[0] = 0
    x[-1] = 0
    # array for kinetic and potential energy
    kin = np.zeros(step)
    pot = np.zeros(step)

    # position histogram
    max_h = 500
    hist = np.ones((N-2) * max_h)
    check = 0
    for n in range(step):
        # pick a position at random, but keeping ends fixed
        j = np.random.randint(1, N-1, size=1)[0]

        # propose an update
        x_new[j] = x[j] + (1 - 2 * np.random.rand()) * dx


        r = np.random.rand() 
        ds = dS_1(j, x, x_new, dtau)

        if r < np.exp(-dtau * ds):
            x[j] = x_new[j]
        
        kin[n] = T_1d(x, dtau)
        pot[n] = V_1d(x)

        if n > neq and n%100 == 0 and check < max_h:
            hist[check*(N-2):(check+1)*(N-2)] = x[1:-1]
            check += 1
        x_new = x

    return kin, pot, hist


def gauss(x, m, s):
    return 1 / np.sqrt(2 * np.pi * s**2) * np.exp(- (x - m)**2 / (2 * s**2))



# ########################### PART A #############################

steps = 10000000
neq = 20000
dx = 0.1
N = 400
tau_f = 100
dim = 1


run = path(steps, neq, dx, N, tau_f, dim)

it = np.arange(0, steps, 1)


fig = plt.figure()

plt.plot(it, run[0], label=r"$T$")
plt.plot(it, run[1], label=r"$V$")
plt.xlabel("Iteration")
plt.ylabel(r"Energy in a.u.")
plt.legend()
plt.grid()
plt.savefig("A3_1.pdf", dpi=200)



fig = plt.figure()

plt.plot(it[1:], run[0][1:]/run[1][1:], label=r"$T/V$")
plt.xlabel("Iteration")
plt.ylabel(r"Energy in a.u.")
plt.legend()
plt.grid()
plt.savefig("A3_2.pdf", dpi=200)


d, x = np.histogram(run[2], bins=200, density=True)

y = []
for i in range(len(x)-1):
    y.append((x[i+1] + x[i])/2)

par = curve_fit(gauss, y, d, p0=[0, 1])[0]


fig = plt.figure()

plt.hist(run[2], bins=100, density=True, label="Data")
plt.plot(y, gauss(y, *par), label="Gaussian fit")
plt.xlabel(r"$x$")
plt.ylabel(r"$P(x)$")
plt.grid()
plt.legend()
plt.savefig("A3_3.pdf", dpi=200)


