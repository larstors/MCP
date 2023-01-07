import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed


def fit(r, a, b):
    return a * r * np.exp(- b * r)


@njit(fastmath=True)
def growth(EL: float, Et: float, dtau: float):
    """ Determines the growth factor given position and E_T

    Args:
        EL (float): local energy
        Et (float): _description_
        dtau (float): discretisation of time

    Returns:
        float: growth factor
    """
    q = np.exp(- dtau(EL - Et))
    return q


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

    return np.exp(-kappa * radius1) * np.exp(-kappa * radius2) * np.exp(beta * radius12 / (1 + alpha * radius12))


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

    locEnergy = (kappa - 2) / radius1 + (kappa - 2) / radius2 + 1 / radius12 * (1 - 2 * beta / u ** 2) + 2 * beta * alpha / u ** 3 - \
        kappa ** 2 - beta ** 2 / u ** 4 + kappa * beta / u ** 2 * \
        np.dot((r1 / radius1 - r2 / radius2),  (r1 - r2) / radius12)
    return locEnergy


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

    force1 = np.zeros(3)
    force2 = np.zeros(3)

    force1 = - kappa * r1 / radius1 + beta * \
        (r1 - r2) / (radius12 * (1 + alpha * radius12)**2)
    force2 = - kappa * r2 / radius2 - beta * \
        (r1 - r2) / (radius12 * (1 + alpha * radius12)**2)
    return force1, force2


@njit(fastmath=True)
def FP_Greens(r: np.ndarray, y: np.ndarray, F: np.ndarray, dtau: float):
    """Green's function for Fokker Planck propagation

    Args:
        r (np.ndarray): position of first electron
        y (np.ndarray): proposed position of first electron
        F (np.ndarray): Force on first electron
        dtau (float): discretisation of FP process

    Returns:
        np.ndarray: Local energy
    """
    exponent = np.linalg.norm(y - r - F * dtau / 2.0)**2

    return 1.0 / np.sqrt(2.0 * np.pi*dtau) * np.exp(- exponent / (2.0 * dtau))


# TODO this surely is solvable without all the for loops, right?
# TODO maybe use C-Types to do this in C/C++, should be a lot faster...

# ! NORMALISATION!!!!!!


@njit(fastmath=True)
def Metropolis_Monte_Carlo(M: int, N: int, n: int, s: float, kappa: float, beta: float, alpha: float, n_equil: int, dtau: float, E0: float):
    """Metropolis Monte Carlo using variantinal Monte Carlo

    Args:
        M (int): Number of walkers
        N (int): Number of simulation steps
        n (int): Number of considered steps (counting from last)
        s (float): Step size for updates
        kappa (float): coefficient for non-interacting model
        beta (float): first coefficient for interaction
        alpha (float): second coefficient for interaction
        n_equil (int): Number of steps to allow for equilibration (burn-in period)
        dtau (float): time discretisation for FP dynamics
        E0 (float): ground state energy estimate

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
    total_local_energy = np.zeros(M)
    dens1 = np.zeros((N - n_equil) * M)
    dens2 = np.zeros((N - n_equil) * M)
    dens_rel = np.zeros((N - n_equil) * M)
    ET = E0
    et = np.zeros()
    # Doing the steps
    for i in range(N+1):
        # update particles
        for m in range(M):

            # quantum force and random noise
            F = Quant_Force(w1[3*m:3*m+3], w2[3*m:3*m+3],
                            kappa, beta, alpha)

            eta = np.random.normal(0, 1, 3)
            # probability before doing the updates
            rho = Trial_Psi(w1[3*m:3*m+3], w2[3*m:3*m+3],
                            kappa, beta, alpha)**2
            # updating the two electrons of each walker
            trial_1 = w1[3*m:3*m+3] + 0.5 * \
                (np.random.rand(3)*2*s-s) + eta * \
                np.sqrt(dtau) + dtau * F[0] / 2.0
            trial_2 = w2[3*m:3*m+3] + 0.5 * \
                (np.random.rand(3)*2*s-s) + eta * \
                np.sqrt(dtau) + dtau * F[1] / 2.0

            # trial force
            trial_F = Quant_Force(
                trial_1, trial_2, kappa, beta, alpha)
            # probability after the updates
            rho_prime = Trial_Psi(trial_1, trial_2, kappa, beta, alpha)**2
            # accept probability

            P_acc = min(1, rho_prime / rho * FP_Greens(trial_1, w1[3*m:3*m+3], trial_F[0], dtau) / FP_Greens(
                w1[3*m:3*m+3], trial_1, F[0], dtau) * FP_Greens(trial_2, w2[3*m:3*m+3], trial_F[1], dtau) / FP_Greens(w2[3*m:3*m+3], trial_2, F[1], dtau))

            # update position if accepting
            if np.random.rand() < P_acc:
                w1[3*m:3*m+3] = trial_1
                w2[3*m:3*m+3] = trial_2

            # add energy to sum
            energy[m] += Local_Energy(w1[3*m:3*m+3],
                                      w2[3*m:3*m+3], kappa, beta, alpha)

            if i >= n_equil:
                total_local_energy[m] += Local_Energy(
                    w1[3*m:3*m+3], w2[3*m:3*m+3], kappa, beta, alpha)
                dens1[(i - n_equil) * M + m] = np.linalg.norm(w1[3*m:3*m+3])
                dens2[(i - n_equil) * M + m] = np.linalg.norm(w2[3*m:3*m+3])
                dens_rel[(i - n_equil) * M +
                         m] = np.linalg.norm(w1[3*m:3*m+3] - w2[3*m:3*m+3])

            # update ET
            ET = E0 + alpha / dtau * np.log(M / (len(w1)/3))
            if i >= n_equil:
                et = np.append(et, ET)
        # to decrease computational effort and RAM we only take
        # every nth measurement into consideration
        if i % n == 0:
            mean_energy[i//n] = np.mean(energy/n)
            variance[i//n] = np.std(energy/n)
            index[i//n] = i
            # start energy from 0 again
            energy *= 0

    tot_E = np.mean(total_local_energy / (N - n_equil))
    tot_var_E = np.std(total_local_energy / (N - n_equil))

    return index, mean_energy, variance, tot_E, tot_var_E, dens1, dens2, dens_rel


# ########################## PART A ###########################
# we assume both cusp conditions are fulfilled
kappa = 2
beta = 0.5
