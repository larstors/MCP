import numpy as np
import matplotlib.pyplot as plt
from numba import njit


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
    return np.exp(-kappa * np.linalg.norm(r1)) * np.exp(-kappa * np.linalg.norm(r2)) \
            * np.exp(beta * np.linalg.norm(r1-r2) / (1 + alpha * np.linalg.norm(r1-r2)))


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
    radius2 = np.linalg.norm(r1)
    radius12 = np.linalg.norm(r1 - r2)
    u = 1 + alpha * radius12

    return (kappa - 2) / radius1 + (kappa - 2) / radius2 \
            + 1 / radius12 * (1 - 2 * beta / u ** 2) \
            + 2 * beta * alpha / u ** 3 \
            - kappa ** 2 - beta ** 2 / u ** 4 \
            + kappa * beta / u ** 2 * np.dot((r1 / radius1 \
            - r2 / radius2),  (r1 - r2) )/ radius12



# TODO this surely is solvable without all the for loops, right?
# TODO maybe use C-Types to do this in C/C++, should be a lot faster...

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
    walkers = np.random.uniform(-0.5, 0.5, size=6*M)
    energy = np.zeros(M)
    mean_energy = np.zeros(N//n)
    variance = np.zeros(N//n)
    index = np.zeros(N//n)


    # Doing the steps
    for i in range(N+1):
        # from uniform distribution
        if method=="uniform":
            # two probabilities (we can neglect normalisation as same
            # for both)
            rho = np.zeros(M)
            rho_prime = np.zeros(M)
            # picking what elecron to update
            ind = np.random.randint(0, 2, size=M)
            # position after update
            new_walkers = walkers
            # update particles
            for m in range(M):
                # probability before doing the updates
                rho[m] = Trial_Psi(walkers[6*m:6*m+3], walkers[6*m+3:6*m+6], kappa, beta, alpha)**2
                # updating one of the two electrons of each walker
                new_walkers[6*m+3*ind[m]:6*m+3+3*ind[m]] = np.random.uniform(-s/2, s/2, size=3)
                # probability after the updates
                rho_prime[m] = Trial_Psi(new_walkers[6*m:6*m+3], new_walkers[6*m+3:6*m+6], kappa, beta, alpha)**2

                if rho_prime[m] / rho[m] >= 1:
                    walkers[6*m:6*m+6] = new_walkers[6*m:6*m+6]
                elif np.random.uniform(0, 1, size=1)[0] < rho_prime[m] / rho[m]:
                    walkers[6*m:6*m+6] = new_walkers[6*m:6*m+6]

                energy[m] += Local_Energy(walkers[6*m:6*m+3], walkers[6*m+3:6*m+6], kappa, beta, alpha)
        




        if i%n == 0 and i > 0:
            mean_energy[i//n-1] = np.mean(energy/n)
            variance[i//n-1] = np.std(energy/n)
            index[i//n-1] = i
            energy *= 0





    return index, mean_energy, variance



run1 = Metropolis_Monte_Carlo(300, 30000, 1000, 0.1, 2.0, 0.5, 0.15)


plt.errorbar(run1[0], run1[1], yerr=run1[2], fmt="-x")
plt.fill_between(run1[0], run1[1]-run1[2], run1[1]+run1[2], alpha=0.1)
plt.xlabel(r"Step $n$")
plt.ylabel(r"$\langle E_L^n\rangle$")
plt.grid()
plt.show()
