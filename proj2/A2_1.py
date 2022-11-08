import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


# 
J = 1

# ####################################### PART A ################################
# constants
kbT = 1
# Nr of spins
N = 20
# external field 
H = 0

# number of spin flips
L = 500


def dE(spin, i, kbT=1):
    #spin that gets flipped
    s = spin[i]
    # energy difference
    E = 2 * s * (spin[(i+1)%N] + spin[(i-1)%N])
    return np.exp(-E*kbT)



# initial spin configuration (all up)
spins = np.ones(N)


fig = plt.figure()


for l in range(L):
    # what spin flips?
    i = np.random.randint(0, N, 1)
    DE = dE(spins, i)
    A = min(1, DE)
    if A == 1:
        spins[i] *= -1
    #elif -np.log(np.random.rand()) < DE:
    elif np.random.rand() < DE:
        spins[i] *= -1

    
    if l % 50 == 0:
        plt.scatter(x=np.arange(1, N+1), y=np.ones(N)*l, c=spins)



plt.xlabel("Index on chain")
plt.ylabel(r"Time $t$")


plt.savefig("1_1d_ising_evolution.pdf", dpi=200)