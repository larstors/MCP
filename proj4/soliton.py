import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import scipy as sc
from numba import njit, jit
from joblib import Parallel, delayed
plt.rcParams.update({'font.size': 7})


eps = 0.2
mu = 0.1
dx = 0.4
dt = 0.1
N = 130
L = N*dx 

x = np.linspace(0, L, 130, endpoint=True)

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
    for i in range(2000):
        if i%250 == 0:
            pos.append(uf.copy())
        if i==0:
            un[2:-2] = u[2:-2] - eps/6 * dt/dx * (u[3:-1] + u[2:-2] + u[1:-3])*(u[3:-1] + u[1:-3])\
                        - mu/2 * dt/dx**3 * (u[4:] + 2*u[1:-3] - 2 * u[3:-1] - u[:-4])
            un[-1] = 0
            un[0] = 1
            un[1] = u[1] - eps/6 * dt/dx * (u[2] + u[1] + u[0])*(u[2] + u[0])\
                        - mu/2 * dt/dx**3 * (u[0] - u[2])
            un[-2] = u[-2] - eps/6 * dt/dx * (u[-1] + u[-2] + u[-3])*(u[-1] + u[-3])\
                        - mu/2 * dt/dx**3 * (u[-3] - u[-1])

        else:
            uf[2:-2] = u[2:-2] - eps/3 * dt/dx * (un[3:-1] + un[2:-2] + un[1:-3])*(un[3:-1] - un[1:-3])\
                        -mu * dt / dx**3 * (un[4:] + 2 * un[1:-3] - 2 * un[3:-1] - un[0:-4])
            
            u = un.copy()
            un = uf.copy()
        stab.append(stable(uf.copy()))
        pos1.append(uf.copy())

    if init==1:
        return pos, stab, pos1
    elif init==2:
        return pos1


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



sol2 = solver(1)[2]

# initializing a figure in 
# which the graph will be plotted
fig = plt.figure() 
   
# marking the x-axis and y-axis
axis = plt.axes(xlim =(0, L),ylim =(0, 2)) 

# initializing a line variable
line, = axis.plot([], [], lw = 3)
   
# data which the line will 
# contain (x, y)
def init(): 
    line.set_data([], [])
    return line,
   
def animate(i):

    # plots a sine graph
    y = sol2[i]
    line.set_data(x, y)
      
    return line,
   
anim = FuncAnimation(fig, animate, init_func = init,
                     frames = 1000, interval = 2, blit = True)
  
   
anim.save('continuousSineWave.gif', 
          writer = 'pillow', fps = 15)