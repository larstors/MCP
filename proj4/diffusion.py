import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from numba import njit, jit
from joblib import Parallel, delayed

# constants
K = 210
C = 900
rho = 2700
L = 1
Nt = 10000

@njit(fastmath=True)
def initial(x: np.ndarray):
    return np.sin(np.pi * x/L)

@njit(fastmath=True)
def analytical(x: np.ndarray, t: float):
    return np.sin(np.pi * x/L) * np.exp( - np.pi**2 * K * t /(L**2 * C * rho))

@njit(fastmath=True)
def error(a: np.ndarray, b: np.ndarray, N: int):
    return 1/N * np.sum(np.abs(a[1:-1] - b[1:-1])) 

#@jit(fastmath=True)
def FTCS(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt[i])
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = K/(C*rho) * dt / (dx**2)
    #iteration
    for n in range(Nmax):
        rod_now[1:-1] = (1 - 2*a) * rod_past[1:-1] + a * (rod_past[0:-2] + rod_past[2:])
        rod_past = rod_now.copy()
    
    return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt

def EulerBack(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt[i])
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = K/(C*rho) * dt / (dx**2)

    alpha = np.ones(N-1)*(-a)
    beta = np.ones(N)*(1+2*a)
    gamma = np.ones(N-1)*(-a)
    alpha[-1] = 0
    gamma[0] = 0
    beta[0] = 1
    beta[-1] = 1
    #iteration
    for n in range(Nmax):
        g = np.zeros(N-1)
        h = np.zeros(N-1)
        # doing the downward recursion
        for i in range(N-1, 0, -1):
            if i == N-1:
                g[i] = - alpha[-1] / beta[-1]
                h[i] = rod_past[-1] / beta[-1]
            g[i-1] = - alpha[i] / (beta[i] + gamma[i] * g[i])
            h[i-1] = (rod_past[i] - gamma[i] * h[i])/(beta[i] + gamma[i] * g[i])
        #upward recursion
        for i in range(0, N-1):
            if i==0:
                rod_now[i] = (rod_past[i]-gamma[i]*h[i])/ (beta[i] + gamma[i] * g[i])
            rod_now[i+1] = g[i] * rod_now[i] + h[i]
        
        rod_past = rod_now.copy()
    return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt

def Crank(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt[i])
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    b = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = K/(C*rho) * dt / (dx**2)

    alpha = np.ones(N-1)*(-a)
    beta = np.ones(N)*(1+2*a)
    gamma = np.ones(N-1)*(-a)
    alpha[-1] = 0
    gamma[0] = 0
    beta[0] = 1
    beta[-1] = 1
    #iteration
    for n in range(Nmax):
        g = np.zeros(N-1)
        h = np.zeros(N-1)
        b[0] = rod_past[0]
        b[-1] = rod_past[-1]
        b[1:-1] = a * rod_past[:-2] + (1 - 2*a)*rod_past[1:-1] + a * rod_past[2:]
        # doing the downward recursion
        for i in range(N-1, 0, -1):
            if i == N-1:
                g[i] = - alpha[-1] / beta[-1]
                h[i] = rod_past[-1] / beta[-1]
            g[i-1] = - alpha[i] / (beta[i] + gamma[i] * g[i])
            h[i-1] = (b[i] - gamma[i] * h[i])/(beta[i] + gamma[i] * g[i])
        #upward recursion
        for i in range(0, N-1):
            if i==0:
                rod_now[i] = (b[i]-gamma[i]*h[i])/ (beta[i] + gamma[i] * g[i])
            rod_now[i+1] = g[i] * b[i] + h[i]
        
        rod_past = rod_now.copy()
    return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt
    

def Dufort(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt[i])
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    rod_future = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = 2*K/(C*rho) * dt / (dx**2)
    for n in range(Nmax):
        # need to do one step first so as to get the proper past
        if n == 0:
            rod_now = analytical(x, 1*dt)

        else:
            rod_future[1:-1] = (1 - a)/(1 + a) * rod_past[1:-1] + a/(1 + a) * (rod_now[:-2] + rod_now[2:])
            rod_past = rod_now.copy()
            rod_now = rod_future.copy()
        
    return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt



# sim1 = FTCS(0.01, 0.1)
# init = initial(sim1[0])


# plt.plot(sim1[0], init, label="n=0")
# plt.plot(sim1[0], sim1[1], label="n=%d" % Nt)
# plt.plot(sim1[0], sim1[2], label="Ana. Sol.")
# plt.xlabel(r"$x$")
# plt.ylabel(r"$T(x, t)$")
# plt.legend()
# plt.grid()
# plt.savefig("2_1.pdf", dpi=200)


##################### part b #########################
# some appropriate dts? say 20?
dt = np.logspace(-3, np.log10(0.7), 100)
# supposed to run until t=100
tmax = 100
# # simple list for epsilon
# eps = []
# for i in range(len(dt)):
#     N = int(tmax/dt[i])
#     sim = FTCS(dx=0.01, dt=dt[i], Nmax=N)
#     eps.append(sim[3])
# eps = np.array(eps)

# fig = plt.figure()
# plt.plot(dt, eps)
# plt.xlabel(r"$\Delta t$")
# plt.ylabel(r"$\epsilon (t=100)$")
# plt.grid()
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig("error_ftcs.pdf", dpi=200)

# a = 0.01**2 * (2*K/(C*rho))**(-1)
# print("FTCS is stable for dt<=%.3f" % a)

# #################### part c ####################

epsFTCS = Parallel(n_jobs=4)(delayed(FTCS)(dx=0.01, dt=i, ) for i in dt)
epsEB = []
epsCr = []
epsDuf = []
for i in range(len(dt)):
    N = int(tmax/dt[i])
    sim = FTCS(dx=0.01, dt=dt[i], Nmax=N)
    epsFTCS.append(sim[3])



epsFTCS = np.array(epsFTCS)

