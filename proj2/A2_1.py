from statistics import variance
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt




def dE(spin, i, kbT=1, J=1, N=20):
    """Function for energy difference

    Args:
        spin (array): spin configuration
        i (int): index of spin that gets flipped
        kbT (float, optional): value of k_b T. Defaults to 1.

    Returns:
        float: difference in energy due to spin flip
    """
    #spin that gets flipped
    s = spin[i]
    # energy difference
    E = 2 * s * (spin[(i+1)%N] + spin[(i-1)%N])
    return np.exp(-E*J/kbT)


def E(spins, J=1, T=1, N=20):
    """Energy of configuration

    Args:
        spins (array): spin configuration
        J (int, optional): value of J. Defaults to 1.
        T (int, optional): value of k_b T. Defaults to 1.

    Returns:
        float: energy of spin configuration
    """
    e = 0
    for i in range(N):
        e += spins[i]*spins[(i+1)%N]
    return -J*e


def run(N=20, L=500, t=1, j=1):
    """Run of 1d Ising model

    Args:
        N (int, optional): number of spins. Defaults to 20.
        L (int, optional): length of simulation. Defaults to 500.
        t (int, optional): value of k_b T. Defaults to 1.
        j (int, optional): value of J. Defaults to 1.

    Returns:
        array: energy at each step
    """
    # for output
    energy = np.ones(L+1)
    #spins
    spins = np.ones(N)
    #add initial energy
    energy[0] = E(spins, J=j, T=t)
    #let system run
    for l in range(L):
        # what spin flips?
        i = np.random.randint(0, N, 1)
        DE = dE(spins, i, t)
        A = min(1, DE)
        #does spin flip?
        if np.random.rand() < A:
            spins[i] *= -1

        #add energy
        energy[l+1] = E(spins, J=j, T=t)
        
    return energy





# ####################################### PART A ################################
# constants
kbT = 1
# 
J = 1
# Nr of spins
N = 20
# external field 
H = 0

# number of spin flips
L = 500




# initial spin configuration (all up)
spins = np.ones(N)


fig = plt.figure()


for l in range(L):
    # what spin flips?
    i = np.random.randint(0, N, 1)
    #energy difference
    DE = dE(spins, i, kbT)
    # threshold
    A = min(1, DE)
    #does it flip
    if np.random.rand() < A:
        spins[i] *= -1

    # plotting
    if l % 5 == 0:
        scat = plt.scatter(x=np.arange(1, N+1), y=np.zeros(N)+l, vmin=-1, vmax=1, c=spins+1)
        


lab = [r"$s_i=-1$", r"$s_i=1$"]
hand = scat.legend_elements()[0]
legend1=plt.legend(handles=hand, labels=lab, framealpha=1, loc="lower right")
plt.xlabel("Index on chain")
plt.ylabel(r"Time $t$")


plt.savefig("1_1d_ising_evolution.pdf", dpi=200)


# # ########################### PART B ################################

# L = 1000

# t = np.array([0.1, 1, 10])
# time = np.arange(0, L+1, 1)

# fig1 = plt.figure()
# for i in range(3):
#     plt.plot(time, run(N, L, t[i], 1), label=r"$k_b T = %g$" % t[i])
# plt.legend()
# plt.grid()
# plt.xlabel(r"Time $L$")
# plt.ylabel(r"$E$")

# plt.savefig("energyGoBOING.pdf", dpi=200)

# # number of iterations
# M = 100

# meanE = np.array([np.zeros(L+1) for i in range(3)])
# meanE2 = np.array([np.zeros(L+1) for i in range(3)])

# var = np.array([np.zeros(L+1) for i in range(3)])

# for i in range(3):
#     kbT = t[i]
#     for m in range(M):
#         energy = run(N, L, kbT, 1)
#         meanE[i] += energy
#         meanE2[i] += energy ** 2

#     meanE[i] /= M
#     meanE2[i] /= M
    
#     var[i] = np.sqrt((meanE2[i] - meanE[i]**2)/M)

# fig2 = plt.figure()

# for i in range(3):
#     plt.plot(time, meanE[i], label=r"$\langle E\rangle, k_b T = %g$" % t[i])
# plt.legend()
# plt.grid()
# plt.xlabel(r"Time $L$")
# plt.ylabel(r"$\langle E\rangle$")
# plt.savefig("mean_energy.pdf", dpi=200)

# fig3 = plt.figure()

# for i in range(3):
#     plt.plot(time, var[i], label=r"$\sigma_E, k_b T = %g$" % t[i])
# plt.legend()
# plt.grid()
# plt.xlabel(r"Time $L$")
# plt.ylabel(r"$\sigma_E$")
# plt.savefig("variance.pdf", dpi=200)


# # ############################## PART C #########################


# def run_2(N=20, L=500, t=1, j=1, tburn=1000):
#     """Run of 1d Ising model modified to equilibrate in the beginning
#     and we only need average energy, so can change the output

#     Args:
#         N (int, optional): number of spins. Defaults to 20.
#         L (int, optional): length of simulation. Defaults to 500.
#         t (int, optional): value of k_b T. Defaults to 1.
#         j (int, optional): value of J. Defaults to 1.

#     Returns:
#         array: energy at each step
#     """
#     # for output
#     energy = 0
#     check = 0
#     #spins
#     spins = np.ones(N)
#     #let system run
#     for l in range(L+tburn):
#         # what spin flips?
#         i = np.random.randint(0, N, 1)
#         DE = dE(spins, i, t)
#         A = min(1, DE)
#         #does spin flip?
#         if np.random.rand() < A:
#             spins[i] *= -1

#         if l > tburn:
#             check += 1
#             energy += E(spins, J=j, T=t)
        
#     return energy/check

# def analytical(T, J=1):
#     return -J * np.tanh(J/T)


# T = np.arange(1, 11)

# Energy = np.ones(len(T))


# for t in range(len(T)):
#     for m in range(M):
#         Energy[t] += run_2(L=1000)
#     Energy[t] /= M

# fig4 = plt.figure()
# plt.plot(T, Energy/N, label=r"$\frac{1}{N}\langle E\rangle$")
# plt.plot(T, analytical(T), label="analytical")
# plt.legend()
# plt.grid()
# plt.xlabel(r"$k_B T$")
# plt.ylabel(r"$E$")
# plt.savefig("energy_per_spin.pdf", dpi=200)




# ########################## PART D #########################

def run_3(N=20, L=500, t=1, j=1, tburn=1000):
    """Run of 1d Ising model modified to equilibrate in the beginning
    and we only need average energy, so can change the output

    Args:
        N (int, optional): number of spins. Defaults to 20.
        L (int, optional): length of simulation. Defaults to 500.
        t (int, optional): value of k_b T. Defaults to 1.
        j (int, optional): value of J. Defaults to 1.

    Returns:
        array: energy at each step
    """
    # for output
    energy = 0
    E2 = 0
    check = 0
    #spins
    spins = np.ones(N)
    #let system run
    for l in range(L+tburn):
        # what spin flips?
        i = np.random.randint(0, N, 1)
        DE = dE(spins, i, t)
        A = min(1, DE)
        #does spin flip?
        if np.random.rand() < A:
            spins[i] *= -1

        if l > tburn:
            e = E(spins, J=j, T=t)
            check += 1
            energy += e
            E2 += e**2
        
    return energy/check, E2/check

def heat_cap(T, J=1):
    return (J/T)**2 * 1/(np.cosh(J/T))**2 

M = 100
T = np.arange(1, 11)

Energy = np.ones(len(T))

meanE = np.ones(len(T))
varE = np.ones(len(T))

for t in range(len(T)):
    for m in range(M):
        r = run_3(L=1000)
        Energy[t] += r[0]
        varE += (r[1] - r[0]**2)
    varE /= M * N * T[t]**2

fig5 = plt.figure()
plt.plot(T, varE, label=r"$c_V$ simulation")
plt.plot(T, heat_cap(T), label="analytical")
plt.legend()
plt.grid()
plt.xlabel(r"$k_B T$")
plt.ylabel(r"$c_V$")
plt.savefig("heat_cap.pdf", dpi=200)

