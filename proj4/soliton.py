import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from colorspacious import cspace_converter
import scipy as sc
from numba import njit, jit
from joblib import Parallel, delayed
plt.rcParams.update({'font.size': 7})
from scipy.signal import argrelextrema
from scipy.signal import find_peaks 


eps = 0.2
mu = 0.1
dx = 0.4
dt = 0.1
N = 130
L = N*dx 

x = np.linspace(0, L, 130, endpoint=True, dtype=np.float64)

@njit(fastmath=True)
def initial1(x: np.ndarray):
    return 0.5 * (1 - np.tanh((x - 25) / 5))

@njit(fastmath=True)
def initial2(x: np.ndarray):
    return 0.8 * (1 - np.tanh(3 * x / 12 - 3)**2) + 0.3 * (1 - np.tanh(4.5*x / 26 - 4.5)**2)


def solver(init: int):
    if init==1:
        u = initial1(x)
    elif init==2:
        u = initial2(x)
    
    un = u.copy()
    uf = u.copy()
    pos = []
    pos1 = []
    stab = []
    for i in range(5000):
        if i%50 == 0:
            pos.append(uf.copy())
        if i==0:
            un[2:-2] = u[2:-2] - eps/6 * dt/dx * (u[3:-1] + u[2:-2] + u[1:-3])*(u[3:-1] + u[1:-3])\
                        - mu/2 * dt/dx**3 * (u[4:] + 2*u[1:-3] - 2 * u[3:-1] - u[:-4])
            if init==1:
                un[-1] = 0
                un[0] = 1
            elif init==2:
                un[-1] = 0
                un[0] = 0
            un[1] = u[1] - eps/6 * dt/dx * (u[2] + u[1] + u[0])*(u[2] + u[0])\
                        - mu/2 * dt/dx**3 * (u[0] - u[2])
                        
            un[-2] = u[-2] - eps/6 * dt/dx * (u[-1] + u[-2] + u[-3])*(u[-1] + u[-3])\
                        - mu/2 * dt/dx**3 * (u[-3] - u[-1])

        else:
            uf[2:-2] = u[2:-2] - eps/3 * dt/dx * (un[3:-1] + un[2:-2] + un[1:-3])*(un[3:-1] - un[1:-3])\
                        -mu * dt / dx**3 * (un[4:] + 2 * un[1:-3] - 2 * un[3:-1] - un[0:-4])
            if init==1:
                uf[-1] = 0
                uf[0] = 1
            elif init==2:
                uf[-1] = 0
                uf[0] = 0
            if np.any(uf>20) == True:
                print(i, un, uf)
                break
            uf[1] = un[1] - eps/6 * dt/dx * (un[2] + un[1] + un[0])*(un[2] + un[0])\
                        - mu/2 * dt/dx**3 * (u[0] - u[2])
            uf[-2] = un[-2] - eps/6 * dt/dx * (un[-1] + un[-2] + un[-3])*(un[-1] + un[-3])\
                        - mu/2 * dt/dx**3 * (un[-3] - un[-1])
            
            u = un.copy()
            un = uf.copy()
        stab.append(stable(uf.copy()))
        pos1.append(uf.copy())

    if init==1:
        return pos, stab, pos1
    elif init==2:
        return pos


def stable(u):
    maxu = max(np.abs(u))
    return dt/dx * (eps * maxu + 4 * mu / dx**2)



# pos = solver(1)



# fig, ax = plt.subplots(nrows=len(pos[0]), ncols=1, sharex=True, sharey=True)
# plt.tight_layout()
# for i in range(len(pos[0])):
#     ax[i].plot(x, pos[0][i], "m-", label=r"n=%d" % (250*i))
#     ax[i].legend()
#     ax[i].grid()
# fig.text(0.01, 0.5, r'$u(x, t)$', va='center', rotation='vertical')
# ax[7].set_xlabel(r"$x$")
# ax[0].set_title("Soliton evolution")
# plt.savefig("soliton_evo.pdf", dpi=200, bbox_inches="tight")

# plt.rcParams.update({'font.size': 12})
# fig2 = plt.figure()
# plt.plot(np.arange(1, len(pos[1])+1), pos[1])
# plt.xlabel(r"Iteration step")
# plt.ylabel(r"Maximal stability")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid()
# plt.savefig("max_stab.pdf", dpi=200)



sol2 = np.array(solver(2))

# # initializing a figure in 
# # which the graph will be plotted
fig = plt.figure() 
   
# marking the x-axis and y-axis
axis = plt.axes(xlim =(0, L),ylim =(0, 2)) 

# initializing a line variable
line, = axis.plot([], [], lw = 3)
axis.set_ylabel(r"$u(x,t)$")
axis.set_xlabel(r"$x$")
axis.grid()
# data which the line will 
# contain (x, y)
def init(): 
    line.set_data([], [])
    return line,
   
def animate(i):

    # plots a sine graph
    y = sol2[i]
    line.set_data(x, y)
    axis.set_title("Iteration=%d" % i)
    
    
    return line,
   
anim = FuncAnimation(fig, animate, init_func = init,
                     frames = len(sol2), interval = 4, blit = True)
  
   
anim.save('soliton_2.gif', 
          writer = 'pillow', fps = 10)


x = []
y = []
for i in sol2:
    y.append(i[argrelextrema(i, np.greater)])
    peaks, _ = find_peaks(i, height=3e-1)
    x.append(peaks)

z = np.array([0+50*i for i in range(len(sol2))])

xmax = np.array([max(sol2[i, x[i]]) for i in range(len(x))])


#fig, ax = plt.subplots(nrows=len(sol2), ncols=1, sharex=True, sharey=True)
# plt.tight_layout()
# for i in range(len(sol2)):
#     ax[i].plot(x, sol2[i], "m-", label=r"n=%d" % (250*i))
#     ax[i].legend()
#     ax[i].grid()
# fig.text(0.01, 0.5, r'$u(x, t)$', va='center', rotation='vertical')
# ax[len(sol2)-1].set_xlabel(r"$x$")
# ax[0].set_title("Soliton evolution")
# plt.savefig("soliton_evo_2.pdf", dpi=200, bbox_inches="tight")
fig = plt.figure()
im = plt.imshow(sol2, aspect="auto", cmap="turbo")
col = plt.colorbar()
col.set_label(r"$u(x, t)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$t$ [50 iteration steps]")
plt.savefig("timewave.pdf")