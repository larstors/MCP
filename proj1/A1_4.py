from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import random


def decay(N0, lambd):
    N = []
    N.append(N0)
    t = []
    t.append(0)
    while N[-1] > 0:
        dN = 0
        for i in range(N[-1]):
            r = np.random.uniform(0, 1, 1)
            if (r < lambd):
                dN += 1
        t.append(t[-1]+1)
        N.append(N[-1] - dN)
    
    return t, N

def cont(t, N0, lambd):
    return N0 * np.exp(-lambd * t)


# ####################### PART A ############################
l = 0.03
N0 = [10, 100, 1000, 10000, 100000]
c = ["red", "blue", "green", "magenta", "cyan"]
style = ["-", "--", "-.", "-x", "-o"]
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 25))
#plt.tight_layout()
for n in range(len(N0)):
    t, N = decay(N0[n], l)
    t = np.asarray(t)
    N = np.asarray(N)
    f = 10
    ax[0].plot(t[::f], cont(t[::f], N0[n], l), style[n], color="black", label=r"$N_\mathrm{cont}$ w. $N(0) = %g$" % N0[n])
    ax[0].plot(t, N, color=c[n], label=r"Discrete w. $N(0) = %g$" % N0[n])
    ax[1].plot(cont(t, N0[n], l), np.abs((N - cont(t, N0[n], l))/cont(t, N0[n], l)), color=c[n], label=r"$N(0)=%g$" % N0[n])

ax[0].set_yscale("log")
ax[0].axis([-20, max(t)*1.1, .4, max(N0)*2])
ax[0].set_xlabel(r"$t$ [s]")
ax[0].set_ylabel(r"$N(t)$")
ax[0].grid()
ax[0].legend()

ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set_xlabel(r"$N_\mathrm{cont.}(t)$")
ax[1].set_ylabel(r"$|N(t) - N_\mathrm{cont.}(t)|/N_\mathrm{cont.}(t)$")
ax[1].grid()
ax[1].legend()
plt.show()

# ####################### PART B ############################

l = 0.3
N0 = np.array([10, 100, 1000, 10000, 100000])
c = ["red", "blue", "green", "magenta", "cyan"]
style = ["-", "--", "-.", "-x", "-o"]
fig1 = plt.figure()
for n in range(len(N0)):
    t, N = decay(N0[n], l)
    t = np.asarray(t)
    N = np.asarray(N)
    f = 1
    plt.plot(t[::f], cont(t[::f], N0[n], l), style[n], color="black", label=r"$N_\mathrm{cont}$ w. $N(0) = %g$" % N0[n])
    plt.plot(t, N, color=c[n], label=r"Discrete w. $N(0) = %g$" % N0[n])

plt.yscale("log")
plt.axis([-1, max(t)*1.1, .4, max(N0)*2])
plt.xlabel(r"$t$ [s]")
plt.ylabel(r"$N(t)$")
plt.grid()
plt.legend()
plt.show()


