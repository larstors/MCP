import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy import integrate

@njit(fastmath=True)
def Trial_Psi(r1: np.ndarray, r2: np.ndarray, kappa: float, beta: float, alpha: float):
    """test function for a helium atom using the Pade-Jastrow ansatz

    Args:
        r1 (np.ndarray): position of first electron
        r2 (np.ndarray): position of second electron
        kappa (float): coefficient for non-interacting model
        beta (float): first coefficient for interaction
        alpha (float): second coefficient for interaction

    Returns:
        float: value of test function given positions of electrons and the corresponding coefficients
    """
    radius1 = np.linalg.norm(r1)
    radius2 = np.linalg.norm(r2)
    radius12 = np.linalg.norm(r1 - r2)

    return np.exp(-kappa * radius1) * np.exp(-kappa * radius2) \
            * np.exp(beta * radius12 / (1 + alpha * radius12))


@njit(fastmath=True)
def Local_Energy(r1: np.ndarray, r2: np.ndarray, kappa: float, beta: float, alpha: float):
    """Local energy of helium atom using Pade-Jastrow ansatz

    Args:
        r1 (np.ndarray): position of first electron
        r2 (np.ndarray): position of second electron
        kappa (float): coefficient for non-interacting model
        beta (float): first coefficient for interaction
        alpha (float): second coefficient for interaction

    Returns:
        float: Local energy
    """
    radius1 = np.linalg.norm(r1)
    radius2 = np.linalg.norm(r2)
    radius12 = np.linalg.norm(r1 - r2)
    u = 1 + alpha * radius12
    if radius1 == 0:
        print("case 1: ", radius1, r1)
    elif radius2 == 0:
        print("case 2: ", radius2, r2)
    elif radius12 == 0:
        print("case 3: ", radius12, r1, r2)
    
    return (kappa - 2) / radius1 + (kappa - 2) / radius2 \
            + 1 / radius12 * (1 - 2 * beta / u ** 2) \
            + 2 * beta * alpha / u ** 3 \
            - kappa ** 2 - beta ** 2 / u ** 4 \
            + kappa * beta / u ** 2 * np.dot((r1 / radius1 \
            - r2 / radius2),  (r1 - r2) / radius12)

@njit(fastmath=True)
def Quant_Force(r1: np.ndarray, r2: np.ndarray, kappa: float, beta: float, alpha: float):
    """Quantum force of Pade-Jastrow ansatz

    Args:
        r1 (np.ndarray): position of first electron
        r2 (np.ndarray): position of second electron
        kappa (float): coefficient for non-interacting model
        beta (float): first coefficient for interaction
        alpha (float): second coefficient for interaction

    Returns:
        np.ndarray: Local energy
    """
    radius1 = np.linalg.norm(r1)
    radius2 = np.linalg.norm(r2)
    radius12 = np.linalg.norm(r1 - r2)
    u = 1 + alpha * radius12
    if radius1 == 0:
        print("case 1: ", radius1, r1)
    elif radius2 == 0:
        print("case 2: ", radius2, r2)
    elif radius12 == 0:
        print("case 3: ", radius12, r1, r2)
    
    force = np.zeros(6)

    force[:3] = - kappa * r1 / radius1 + beta * (r1 - r2) / (radius12 * (1 + alpha * radius12)**2)
    force[3:] = - kappa * r2 / radius2 - beta * (r1 - r2) / (radius12 * (1 + alpha * radius12)**2)

    return force

# TODO this surely is solvable without all the for loops, right?
# TODO maybe use C-Types to do this in C/C++, should be a lot faster...

# ! NORMALISATION!!!!!!

@njit(fastmath=True)
def Metropolis_Monte_Carlo(M: int, N: int, n: int, s: float, kappa: float, beta: float, alpha: float, method="uniform"):
    """Metropolis Monte Carlo using variantinal Monte Carlo

    Args:
        M (int): Number of walkers
        N (int): Number of simulation steps
        n (int): Number of considered steps (counting from last)
        s (float): Step size for updates
        kappa (float): coefficient for non-interacting model
        beta (float): first coefficient for interaction
        alpha (float): second coefficient for interaction
        method (str, optional): Method for walker updates. Defaults to "uniform".

    Returns:
        array: index, average energy at index, variance at index
    """
    # arrays that are needed
    w1 = np.random.uniform(-0.5, 0.5, size=3*M)
    w2 = np.random.uniform(-0.5, 0.5, size=3*M)
    energy = np.zeros(M)
    mean_energy = np.zeros(N//n)
    variance = np.zeros(N//n)
    index = np.zeros(N//n)


    # Doing the steps
    for i in range(N+1):
        # from uniform distribution
        if method=="uniform":
            # update particles
            for m in range(M):
                
                # probability before doing the updates
                rho = Trial_Psi(w1[3*m:3*m+3], w2[3*m:3*m+3], kappa, beta, alpha)**2
                r = rho
                chose = np.random.rand()
                # updating one of the two electrons of each walker
                if chose < 0.5:
                    trial_1 = w1[3*m:3*m+3] + 0.5*(np.random.rand(3)*2*s-s)
                    trial_2 = w2[3*m:3*m+3]
                else:
                    trial_1 = w1[3*m:3*m+3]
                    trial_2 = w2[3*m:3*m+3] + 0.5*(np.random.rand(3)*2*s-s)
                # probability after the updates
                rho_prime = Trial_Psi(trial_1, trial_2, kappa, beta, alpha)**2
                # accept new position?
                #print(m, rho_prime, rho)

                if rho == 0:
                    print("now", m, rho_prime, rho)
                if np.random.rand() < rho_prime / rho:
                    w1[3*m:3*m+3] = trial_1 
                    w2[3*m:3*m+3] = trial_2
                    r = rho_prime
                
                # add energy to sum
                energy[m] += Local_Energy(w1[3*m:3*m+3], w2[3*m:3*m+3], kappa, beta, alpha) * r
                

        # to decrease computational effort and RAM we only take 
        # every nth measurement into consideration
        if i%n == 0:
            mean_energy[i//n] = np.mean(energy/n)
            variance[i//n] = np.std(energy/n)
            index[i//n] = i
            # start energy from 0 again
            energy *= 0

    return index, mean_energy, variance


# ########################### PART A ###########################

choice_s = np.array([0.1, 1.0, 10])

#run1 = Metropolis_Monte_Carlo(300, 30000, 1000, 0.1, 2.0, 0.5, 0.15)

result = Parallel(n_jobs=1)(delayed(Metropolis_Monte_Carlo)(M=300, N=30000, n=1000, s=ch, \
        kappa=2.0, beta=0.5, alpha=0.15, method="uniform") for ch in choice_s)

fig1 = plt.figure()

for l, so in enumerate(choice_s):
    plt.errorbar(result[l][0], result[l][1], yerr=result[l][2], fmt="-x", label=r"$s=%.1f$" % so)
    plt.fill_between(result[l][0], result[l][1]-result[l][2], result[l][1]+result[l][2], alpha=0.2)
plt.xlabel(r"Step $n$")
plt.ylabel(r"$\langle E_L^n\rangle$")
plt.legend()
plt.grid()
plt.savefig("A1_1.pdf", dpi=200)


print("Std.dev. for s=0.1, =1.0, and =10.0 are respectively", result[0][2][-1], result[1][2][-1], result[2][2][-1])


# ########################### PART A ###########################
# TODO is this really it?
s_final = 1.0

alp = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

result = Parallel(n_jobs=3)(delayed(Metropolis_Monte_Carlo)(M=300, N=10000, n=1000, s=s_final, \
        kappa=2.0, beta=0.5, alpha=al, method="uniform") for al in alp)

fig2 = plt.figure()

for l, a in enumerate(alp):
    plt.errorbar(result[l][0], result[l][1], yerr=result[l][2], fmt="-x", label=r"$\alpha=%.1f$" % a)
    plt.fill_between(result[l][0], result[l][1]-result[l][2], result[l][1]+result[l][2], alpha=0.2)
plt.xlabel(r"Step $n$")
plt.ylabel(r"$\langle E_L^n\rangle$")
plt.legend()
plt.grid()
plt.savefig("A1_2.pdf", dpi=200)





# o = integrate.nquad(normalisation, [[-1e4,1e4], [-1e4,1e4], [-1e4,1e4], [-1e4,1e4], [-1e4,1e4], [-1e4,1e4]], full_output=True, args=(2, 0.5, 0.15))
# print(o)