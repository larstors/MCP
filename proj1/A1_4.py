from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import random

#TODO add Gillespie?

def decay(N0, lambd):
    """Function to stochastically calculate the number of particles left

    Args:
        N0 (int): initial number of particles
        lambd (float): rate of decay

    Returns:
        list: list of times and N at each time
    """
    # output lists
    N = []
    N.append(N0)
    t = []
    t.append(0)
    # while still stuff to decay
    while N[-1] > 0:
        dN = 0
        # check if particles decay
        for i in range(N[-1]):
            r = np.random.uniform(0, 1, 1)
            if (r < lambd):
                dN += 1
        #update time and N
        t.append(t[-1]+1)
        N.append(N[-1] - dN)
    
    return t, N

def cont(t, N0, lambd):
    """Solution of continuous model

    Args:
        t (float): time
        N0 (int): initial N
        lambd (float): decay rate

    Returns:
        float: predicted number of particles per continuum model
    """
    return N0 * np.exp(-lambd * t)


# ####################### PART A ############################
# constants
l = 0.03
N0 = [10, 100, 1000, 10000, 100000]
# stuff for plotting
c = ["red", "blue", "green", "magenta", "cyan"]
style = ["-", "--", "-.", "-x", "-o"]
fig1, ax = plt.subplots()
# making curve for each N0
for n in range(len(N0)):
    t, N = decay(N0[n], l)
    t = np.asarray(t)
    N = np.asarray(N)
    f = 10
    ax.plot(t, N, color=c[n], label=r"$N(0) = %g$" % N0[n])
    if n == len(N0)-1:
        plt.plot(t[::f], cont(t[::f], N0[n], l), style[2], color="black", label=r"Continuous Model")
    else:
        plt.plot(t[::f], cont(t[::f], N0[n], l), style[2], color="black")

ax.set_title(r"Decay with rate $\lambda=%.2f\mathrm{s}^{-1}$" % l)
ax.set_yscale("log")
ax.axis([-20, max(t)*1.1, .4, max(N0)*2])
ax.set_xlabel(r"$t$ [s]")
ax.set_ylabel(r"$N(t)$")
ax.grid()
ax.legend()


plt.savefig("decay_low.pdf", dpi=200)

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
    plt.plot(t, N, color=c[n], label=r"$N(0) = %g$" % N0[n])
    if n == len(N0)-1:
        plt.plot(t[::f], cont(t[::f], N0[n], l), style[2], color="black", label=r"Continuous Model")
    else:
        plt.plot(t[::f], cont(t[::f], N0[n], l), style[2], color="black")

plt.title(r"Decay with rate $\lambda=%.2f\mathrm{s}^{-1}$" % l)
plt.yscale("log")
plt.axis([-1, max(t)*1.1, .4, max(N0)*2])
plt.xlabel(r"$t$ [s]")
plt.ylabel(r"$N(t)$")
plt.grid()
plt.legend()
plt.savefig("decay_high.pdf", dpi=200)


