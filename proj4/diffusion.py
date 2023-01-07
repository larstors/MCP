import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from numba import njit, jit
from joblib import Parallel, delayed
from matplotlib import colors
plt.rcParams.update({'font.size': 20})



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
        Nmax = int(tmax/dt)
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = K/(C*rho) * dt / (dx**2)

    temp = []
    #iteration
    for n in range(Nmax):
        rod_now[1:-1] = (1 - 2*a) * rod_past[1:-1] + a * (rod_past[0:-2] + rod_past[2:])
        rod_past = rod_now
        temp.append(rod_past.copy())
    
    
    #return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), L/dxNmax),dt*Nt
    return error(rod_now, analytical(x, dt*Nmax), L/dx)
    #return temp, x


def EulerBack(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt)
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = K/(C*rho) * dt / (dx**2)
    F = np.zeros((N, N))
    F[0, 0] = 1
    F[-1, -1] = 1
    for i in range(1, N-1):
        F[i, i] = (1 + 2*a)
        F[i, i+1] = -a
        F[i, i-1] = -a
    temp = []
    #iteration
    for n in range(Nmax):
        rod_now = np.linalg.inv(F).dot(rod_past)
        rod_past = rod_now.copy()
        temp.append(rod_now.copy())
    #return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt
    return error(rod_now, analytical(x, dt*Nmax), L/dx)
    #return temp, x


def Crank(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt)
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    b = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = K/(C*rho) * dt / (2*dx**2)
    F = np.zeros((N, N))
    B = np.zeros((N, N))
    F[0, 0] = 1
    F[-1, -1] = 1
    B[0, 0] = 1
    B[-1, -1] = 1
    for i in range(1, N-1):
        F[i, i] = (1 + 2*a)
        F[i, i+1] = -a
        F[i, i-1] = -a

        B[i, i] = (1 - 2*a)
        B[i, i+1] = a
        B[i, i-1] = a
    temp = []
    #iteration
    for n in range(Nmax):
        b = B.dot(rod_past)
        A1 = np.linalg.inv(F)
        rod_now = A1.dot(b)
        rod_past = rod_now.copy()
        temp.append(rod_now.copy())
    #return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt
    return error(rod_now, analytical(x, dt*Nmax), L/dx)
    #return temp, x
    

def Dufort(dx: float, dt: float, Nmax=Nt, tmax=0):
    # lattice size 
    N = int(L/dx)
    if tmax > 0:
        Nmax = int(tmax/dt)
    #system array
    rod_past = np.zeros(N)
    rod_now = np.zeros(N)
    rod_future = np.zeros(N)
    # x coordinates
    x = np.linspace(0, L, N, endpoint=True)
    # initial conditions
    rod_past = initial(x)
    a = 2*K/(C*rho) * dt / (dx**2)
    temp = []
    for n in range(Nmax+1):
        # need to do one step first so as to get the proper past
        if n == 0:
            rod_now = analytical(x, 1*dt)

        else:
            rod_future[1:-1] = (1 - a)/(1 + a) * rod_past[1:-1] + a/(1 + a) * (rod_now[:-2] + rod_now[2:])
            rod_past = rod_now.copy()
            rod_now = rod_future.copy()
            temp.append(rod_future.copy())
        
    #return x, rod_now, analytical(x, dt*Nmax), error(rod_now, analytical(x, dt*Nmax), Nmax),dt*Nt
    return error(rod_now, analytical(x, dt*Nmax), L/dx)
    #return temp, x


# sim1 = FTCS(0.01, 0.1)
# t = np.arange(0, 0.1*Nt, 0.1)

# # sim1 = Dufort(0.01, 0.1)
# # t = np.arange(0, 0.1*Nt, 0.1)


# an = []
# for i in t:
#     an.append(analytical(sim1[1], i))


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), sharex=True, sharey=True)
# plt.tight_layout()
# im = ax[0].imshow(sim1[0], aspect="auto", vmin=-0.2, vmax=1)
# im1 = ax[1].imshow(an, aspect="auto", vmin=-0.2, vmax=1)
# ax[0].set_title("Simulation")
# ax[1].set_title("Analytical")
# ax[0].set_ylabel("Iteration")
# ax[0].set_xlabel(r"$x$")
# ax[1].set_xlabel(r"$x$")
# plt.colorbar(im1, ax=ax)
# plt.savefig("2d_temp.pdf", dpi=200, bbox_inches="tight")
# #init = initial(sim1[0])

# fig2 = plt.figure()
# plt.imshow(np.array(sim1[0])-np.array(an), aspect="auto")
# plt.colorbar()
# plt.ylabel("Iterations")
# plt.xlabel(r"$x$")
# plt.title("Simulation - Analytical")
# plt.savefig("2d_temp_comp.pdf", dpi=200)
# plt.plot(sim1[0], init, label="n=0")
# plt.plot(sim1[0], sim1[1], label="n=%d" % Nt)
# plt.plot(sim1[0], sim1[2], label="Ana. Sol.")
# plt.xlabel(r"$x$")
# plt.ylabel(r"$T(x, t)$")
# plt.legend()
# plt.grid()
# #plt.savefig("2_1.pdf", dpi=200)

##################### part b #########################
# some appropriate dts? say 20?
dt = np.logspace(-3, 1, 30)
# supposed to run until t=100
tmax = 100
# # simple list for epsilon
eps = []
a = 0.01**2 * (2*K/(C*rho))**(-1)
for i in range(len(dt)):
    N = int(tmax/dt[i])
    sim = FTCS(dx=0.01, dt=dt[i], Nmax=N)
    eps.append(sim)
eps = np.array(eps)

fig = plt.figure()
plt.plot(dt, eps)
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$\epsilon (t=100)$")
plt.grid()
plt.plot(np.ones(2)*a, [1e-4, 1e0])
plt.axis([1e-3, 1e0, 1e-4, 1e0])
plt.xscale("log")
plt.yscale("log")
plt.savefig("error_ftcs.pdf", dpi=200, bbox_inches="tight")


# print("FTCS is stable for dt<=%.3f" % a)

# #################### part c ####################

epsFTCS = Parallel(n_jobs=4)(delayed(FTCS)(dx=0.01, dt=i, Nmax=10, tmax=100) for i in dt)
epsEB = Parallel(n_jobs=4)(delayed(EulerBack)(dx=0.01, dt=i, Nmax=10, tmax=100) for i in dt)
epsCr = Parallel(n_jobs=4)(delayed(Crank)(dx=0.01, dt=i, Nmax=10, tmax=100) for i in dt)
epsDuf = Parallel(n_jobs=4)(delayed(Dufort)(dx=0.01, dt=i, Nmax=10, tmax=100) for i in dt)


f2 = plt.figure()
plt.plot(dt, epsFTCS, "-o", label="FTCS")
plt.plot(dt, epsEB, "-o", label="Euler Back")
plt.plot(dt, epsCr, "-o", label="Crank-Nicolson")
plt.plot(dt, epsDuf, "-o", label="Dufort-Frankel")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$\epsilon(t_\mathrm{max})$")
plt.legend()
plt.axis([dt.min(), dt.max(), 1e-6, 1e1])
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.savefig("2method_comp.pdf", dpi=200, bbox_inches="tight")


