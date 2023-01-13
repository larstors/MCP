import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
from joblib import Parallel, delayed
import math

def fit(r, a, b):
    return a * r * np.exp(- b * r)


@jit(fastmath=True)
def growth(EL: float, Et: float, dtau: float):
    """ Determines the growth factor given position and E_T

    Args:
        EL (float): local energy
        Et (float): _description_
        dtau (float): discretisation of time

    Returns:
        float: growth factor
    """
    q = np.exp(- dtau*(EL - Et))
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

    force1 = - 2 * kappa * r1 / radius1 + 2 * beta * \
        (r1 - r2) / (radius12 * (1 + alpha * radius12)**2)
    force2 = - 2 * kappa * r2 / radius2 - 2 * beta * \
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

# TODO make single step, with this we should be able to update the thingy faster

@njit(fastmath=True)
def single_step(M: int, N: int, n: int, s: float, kappa: float, beta: float, alpha: float, n_equil: int, dtau: float, ET: float, w1: np.ndarray, w2: np.ndarray):
    # array for new walkers
    new_walk1 = w1
    new_walk2 = w2
    del_1 = np.array([])
    del_2 = np.array([])
    # update particles
    for m in range(int(len(w1)/3)):

        # quantum force and random noise
        F = Quant_Force(w1[3*m:3*m+3], w2[3*m:3*m+3],
                        kappa, beta, alpha)

        eta1 = np.random.normal(0, 1, 3)
        eta2 = np.random.normal(0, 1, 3)
        # probability before doing the updates
        rho = Trial_Psi(w1[3*m:3*m+3], w2[3*m:3*m+3],
                        kappa, beta, alpha)**2
        # updating the two electrons of each walker
        # print(np.shape(eta1), np.shape(F[0]), np.shape(F), i, m)
        trial_1 = w1[3*m:3*m+3] + eta1 * np.sqrt(dtau) + dtau * F[0] / 2.0
        trial_2 = w2[3*m:3*m+3] + eta2 * np.sqrt(dtau) + dtau * F[1] / 2.0

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

        e = Local_Energy(
                    w1[3*m:3*m+3], w2[3*m:3*m+3], kappa, beta, alpha)

        q = growth(EL=e, Et=ET, dtau=dtau)
        
        if q <= 1 and np.random.rand() > q:
            del_1 = np.append(del_1, np.arange(3*m,3*m+3, dtype=int))
            del_2 = np.append(del_2, np.arange(3*m,3*m+3, dtype=int))
        elif q > 1:
            m_i = math.floor(q + np.random.rand())
            for j in range(m_i):
                new_walk1 = np.append(new_walk1, w1[3*m:3*m+3])
                new_walk2 = np.append(new_walk2, w2[3*m:3*m+3])



    del_1 = del_1.astype(np.int32)
    del_2 = del_2.astype(np.int32)

    # update the walkers 
    new_walk1 = np.delete(new_walk1, del_1)
    new_walk2 = np.delete(new_walk2, del_2)

    # just in case
    if len(w1) != len(w2):
        print("wow, this is wrong")
    
    return new_walk1, new_walk2


#@njit(fastmath=True)
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

    dens1 = np.array([])
    dens2 = np.array([])
    dens_rel = np.array([])
    ET = E0
    et = np.array([])
    iteration = np.array([])
    # Doing the steps
    for i in range(N+1):
        print(i, len(w1)/3)

        # array for new walkers
        new_walk1 = w1.copy()
        new_walk2 = w2.copy()
        del_1 = np.array([])
        del_2 = np.array([])
        # update particles
        for m in range(int(len(w1)/3)):

            # quantum force and random noise
            F = Quant_Force(w1[3*m:3*m+3], w2[3*m:3*m+3],
                            kappa, beta, alpha)

            eta1 = np.random.normal(0, 1, 3)
            eta2 = np.random.normal(0, 1, 3)
            # probability before doing the updates
            rho = Trial_Psi(w1[3*m:3*m+3], w2[3*m:3*m+3],
                            kappa, beta, alpha)**2
            # updating the two electrons of each walker
            # print(np.shape(eta1), np.shape(F[0]), np.shape(F), i, m)
            trial_1 = w1[3*m:3*m+3] + eta1 * np.sqrt(dtau) + dtau * F[0] / 2.0
            trial_2 = w2[3*m:3*m+3] + eta2 * np.sqrt(dtau) + dtau * F[1] / 2.0

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

            e = Local_Energy(
                        w1[3*m:3*m+3], w2[3*m:3*m+3], kappa, beta, alpha)

            q = growth(EL=e, Et=ET, dtau=dtau)
            
            if q <= 1 and np.random.rand() > q:
                del_1 = np.append(del_1, np.arange(3*m,3*m+3, dtype=int))
                del_2 = np.append(del_2, np.arange(3*m,3*m+3, dtype=int))
            elif q > 1:
                m_i = math.floor(q + np.random.rand())
                for j in range(m_i):
                    new_walk1 = np.append(new_walk1, w1[3*m:3*m+3])
                    new_walk2 = np.append(new_walk2, w2[3*m:3*m+3])



            # if i >= n_equil:
            #     dens1 = np.append(dens1, np.linalg.norm(w1[3*m:3*m+3]))
            #     dens2 = np.append(dens2, np.linalg.norm(w2[3*m:3*m+3]))
            #     dens_rel = np.append(dens_rel, np.linalg.norm(
            #         w1[3*m:3*m+3] - w2[3*m:3*m+3]))

        # update ET
        ET = E0 + np.log(M / (len(w1)/3))
        if i >= n_equil:
            iteration = np.append(iteration, i)
            et = np.append(et, ET)
        del_1 = del_1.astype(np.int32)
        del_2 = del_2.astype(np.int32)

        # update the walkers 
        w1 = np.delete(new_walk1, del_1)
        w2 = np.delete(new_walk2, del_2)


        # just in case
        if len(w1) != len(w2):
            print("wow, this is wrong")
        
    return iteration, et, np.mean(et)#, dens1, dens2, dens_rel


# ########################## PART A ###########################
# we assume both cusp conditions are fulfilled
kappa = 2.0
beta = 0.5

# given values
dtau = 0.03
N = 40000
neq = 10000
M0 = 300
E0 = -2.891

run = Metropolis_Monte_Carlo(M0, N, 1, 1.0, kappa, beta, 0.12, neq, dtau, E0)

it = run[0]
val = run[1]

fig = plt.figure()
plt.plot(it[::100], val[::100])
plt.savefig("A2_1.pdf", dpi=200)
print(run[2])