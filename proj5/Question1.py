import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed


def fit(r, a, b):
    return a * r * np.exp(- b * r)


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
def Metropolis_Monte_Carlo(M: int, N: int, n: int, s: float, kappa: float, beta: float, alpha: float, n_equil: int, dtau: float, method="uniform"):
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
    total_local_energy = np.zeros(M)
    dens1 = np.zeros((N - n_equil) * M)
    dens2 = np.zeros((N - n_equil) * M)
    dens_rel = np.zeros((N - n_equil) * M)

    # Doing the steps
    for i in range(N+1):
        # from uniform distribution
        if method == "uniform":
            # update particles
            for m in range(M):

                # probability before doing the updates
                rho = Trial_Psi(w1[3*m:3*m+3], w2[3*m:3*m+3],
                                kappa, beta, alpha)**2
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
                energy[m] += Local_Energy(w1[3*m:3*m+3],
                                          w2[3*m:3*m+3], kappa, beta, alpha)

                if i > n_equil:
                    total_local_energy[m] += Local_Energy(
                        w1[3*m:3*m+3], w2[3*m:3*m+3], kappa, beta, alpha)

        # Fokker Planck dynamics
        elif method == "FP":

            # TODO finish this
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


# ########################### PART A ###########################

choice_s = np.array([0.1, 1.0, 10])

#run1 = Metropolis_Monte_Carlo(300, 30000, 1000, 0.1, 2.0, 0.5, 0.15)

result = Parallel(n_jobs=3)(delayed(Metropolis_Monte_Carlo)(M=300, N=30000, n=1000, s=ch,
                                                            kappa=2.0, beta=0.5, alpha=0.15, n_equil=30000, dtau=0, method="uniform") for ch in choice_s)

fig1 = plt.figure()

for l, so in enumerate(choice_s):
    plt.errorbar(result[l][0], result[l][1], yerr=result[l]
                 [2], fmt="-x", label=r"$s=%.1f$" % so)
    plt.fill_between(result[l][0], result[l][1]-result[l]
                     [2], result[l][1]+result[l][2], alpha=0.2)
plt.xlabel(r"Step $n$")
plt.ylabel(r"$\langle E_L^n\rangle$")
plt.legend()
plt.grid()
plt.savefig("A1_1.pdf", dpi=200)


print("Std.dev. for s=0.1, =1.0, and =10.0 are respectively ",
      result[0][2][-1], result[1][2][-1], result[2][2][-1])


# ########################### PART B ###########################
# TODO is this really it?
s_final = 1.0

alp = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

result = Parallel(n_jobs=6)(delayed(Metropolis_Monte_Carlo)(M=300, N=10000, n=1000, s=s_final,
                                                            kappa=2.0, beta=0.5, alpha=al, n_equil=10000, dtau=0, method="uniform") for al in alp)

fig2 = plt.figure()

for l, a in enumerate(alp):
    plt.errorbar(result[l][0], result[l][1], yerr=result[l]
                 [2], fmt="-x", label=r"$\alpha=%.1f$" % a)
    plt.fill_between(result[l][0], result[l][1]-result[l]
                     [2], result[l][1]+result[l][2], alpha=0.2)
    print("Std.dev. for alpha=%.2f is %.3f" % (a, result[l][2][-1]))
plt.xlabel(r"Step $n$")
plt.ylabel(r"$\langle E_L^n\rangle$")
plt.legend()
plt.grid()
plt.savefig("A1_2.pdf", dpi=200)


# ############################ PART C #########################

alp = np.linspace(0.0, 0.5, 51, endpoint=True)
s_final = 1.0
N_it = 40000
n_eq = 10000


result = Parallel(n_jobs=10)(delayed(Metropolis_Monte_Carlo)(M=300, N=N_it, n=1000, s=s_final,
                                                             kappa=2.0, beta=0.5, alpha=al, n_equil=n_eq, dtau=0, method="uniform") for al in alp)


E = []
varE = []
for i in result:
    E.append(i[3])
    varE.append(i[4])


fig3 = plt.figure()
plt.errorbar(x=alp, y=E, yerr=varE, fmt="-x", color="m")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\langle\bar{E}_L\rangle$")
plt.grid()
plt.savefig("A1_3.pdf", dpi=200)

min_index_E = np.argmin(E)
min_index_var_E = np.argmin(varE)

print("With N=%d we get the minima for E_L=%.3f and std(E_L)=%.3f at alpha = %.2f and %.2f, respectively" %
      (N_it, E[min_index_E], varE[min_index_var_E], alp[min_index_E], alp[min_index_var_E]))

# ############################ PART D #########################

kap = np.linspace(1.7, 2.2, 31, endpoint=True)
s_final = 1.0
alp_min = alp[min_index_var_E]

result = Parallel(n_jobs=10)(delayed(Metropolis_Monte_Carlo)(M=300, N=N_it, n=1000, s=s_final,
                                                             kappa=ka, beta=0.5, alpha=alp_min, n_equil=n_eq, dtau=0, method="uniform") for ka in kap)


E = []
varE = []
for i in result:
    E.append(i[3])
    varE.append(i[4])


fig3 = plt.figure()
plt.errorbar(x=kap, y=E, yerr=varE, fmt="-x", color="m")
plt.xlabel(r"$\kappa$")
plt.ylabel(r"$\langle\bar{E}_L\rangle$")
plt.grid()
plt.savefig("A1_4.pdf", dpi=200)

min_index_E = np.argmin(E)
min_index_var_E = np.argmin(varE)

print("With N=%d we get the minima for E_L=%.3f and std(E_L)=%.3f at kappa = %.2f and %.2f, respectively" %
      (N_it, E[min_index_E], varE[min_index_var_E], kap[min_index_E], kap[min_index_var_E]))

# ############################### PART E #########################

bet = np.linspace(0.2, 0.6, 31, endpoint=True)
s_final = 1.0
kap_min = kap[min_index_var_E]

result = Parallel(n_jobs=10)(delayed(Metropolis_Monte_Carlo)(M=300, N=N_it, n=1000, s=s_final,
                                                             kappa=kap_min, beta=be, alpha=alp_min, n_equil=n_eq, dtau=0, method="uniform") for be in bet)


E = []
varE = []
for i in result:
    E.append(i[3])
    varE.append(i[4])


fig3 = plt.figure()
plt.errorbar(x=bet, y=E, yerr=varE, fmt="-x", color="m")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\langle\bar{E}_L\rangle$")
plt.grid()
plt.savefig("A1_5.pdf", dpi=200)

min_index_E = np.argmin(E)
min_index_var_E = np.argmin(varE)

print("With N=%d we get the minima for E_L=%.3f and std(E_L)=%.3f at beta = %.2f and %.2f, respectively" %
      (N_it, E[min_index_E], varE[min_index_var_E], bet[min_index_E], bet[min_index_var_E]))

res = Metropolis_Monte_Carlo(M=300, N=N_it, n=1000, s=s_final,
                             kappa=1.85, alpha=0.18, beta=0.38, n_equil=n_eq, dtau=0)

print("Optimal choice yields E=%f with std(E)=%f" % (res[3], res[4]))

# ############################### PART G #########################

tau = np.array([0.01, 0.05, 0.1, 0.2, 1.0])
N_it = 20000
n_eq = 10000

result = Parallel(n_jobs=5)(delayed(Metropolis_Monte_Carlo)(M=300, N=N_it, n=1000, s=s_final,
                                                            kappa=1.85, alpha=0.18, beta=0.38, n_equil=n_eq, dtau=t, method="FP") for t in tau)

E = []
varE = []
for i in result:
    E.append(i[3])
    varE.append(i[4])


fig3 = plt.figure()
plt.errorbar(x=tau, y=E, yerr=varE, fmt="-x", color="m")
plt.xlabel(r"$\Delta \tau$")
plt.ylabel(r"$\langle\bar{E}_L\rangle$")
plt.grid()
plt.savefig("A1_6.pdf", dpi=200)

min_index_E = np.argmin(E)
min_index_var_E = np.argmin(varE)

print("With N=%d we get the minima for E_L=%.3f and std(E_L)=%.3f at delta tau = %.2f and %.2f, respectively" %
      (N_it, E[min_index_E], varE[min_index_var_E], tau[min_index_E], tau[min_index_var_E]))

tau = 0.05

res = Metropolis_Monte_Carlo(M=300, N=N_it, n=1000, s=s_final, kappa=1.85,
                             alpha=0.18, beta=0.38, n_equil=n_eq, dtau=tau, method="FP")


fig6, ax = plt.subplots(nrows=3, ncols=1)
plt.tight_layout()
ax[0].hist(res[5], bins=200, density=True, label="Electron 1")
ax[1].hist(res[6], bins=200, density=True, label="Electron 2")
ax[2].hist(res[7], bins=200, density=True)

ax[0].set_ylabel(r"$\rho(r_1)$")
ax[1].set_ylabel(r"$\rho(r_2)$")
ax[2].set_ylabel(r"$\rho(|r_1-r_2|)$")
ax[0].set_xlabel(r"$r_1$")
ax[1].set_xlabel(r"$r_2$")
ax[2].set_xlabel(r"$|r_1-r_2|$")
ax[0].grid()
ax[1].grid()
ax[2].grid()

plt.savefig("A1_7.pdf", dpi=200, bbox_inches="tight")


# o = integrate.nquad(normalisation, [[-1e4,1e4], [-1e4,1e4], [-1e4,1e4], [-1e4,1e4], [-1e4,1e4], [-1e4,1e4]], full_output=True, args=(2, 0.5, 0.15))
# print(o)
