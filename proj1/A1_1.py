import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import random



# Parameters for the integration
a = 0       #starting point
b = 1       #end point
N = 1000    #number of sample points
V = b - a   #integration volume
M = 1000    # number of iterations
Nr_bins = 30 #number of bins

# Doing the integration by hand yields
I = 0.2

def gauss(x, mu, sig):
    return 1/np.sqrt(2*np.pi*sig**2)*np.exp(-1/(2*sig**2) * (x - mu)**2)

def f(x):
    """function to integrate

    Args:
        x (array): array of values at which the function is evaluated
    """
    return x**4

def average(y, n=N):
    """function to calculate average

    Args:
        y (array): data to find average of

    Returns:
        float: average of y
    """
    return 1/n * np.sum(y)

# #################### PART A #################

# random numbers from uniform distribution
x = np.random.uniform(a, b, N)

# the monte-carlo estimator
I_N = V*average(f(x))
# and the error
sigma_Na = V * np.sqrt((average(f(x)*f(x)) - average(f(x))**2)/(N-1))
# distance between actual value and estimator in units of error
distance = np.abs(I - I_N)/sigma_Na

print("The actual result is ", 0.2)
print("Monte-Carlo estimator yields ", I_N, " with a standard error ", sigma_Na)
print("The estimator is %.2f sigma_N away" % distance)


# #################### PART B #################

data = []
x_range = []

for i in range(M):
    # random numbers from uniform distribution
    x = np.random.uniform(a, b, N)

    # the monte-carlo estimator
    I_N = V*average(f(x))
    # and the error
    sigma_N = V * np.sqrt((average(f(x)*f(x)) - average(f(x))**2)/(N-1))

    data.append(I_N)

# make histogram
datahist, bins = np.histogram(data, Nr_bins, density="True")

for i in range(len(bins)-1):
    x_range.append((bins[i+1] + bins[i])/2)

# parameters for gaussian fit
par, cov = opt.curve_fit(gauss, xdata=x_range, ydata=datahist, p0=(I, sigma_N))

#for smooth gaussian
x_range = np.linspace(min(x_range), max(x_range), 1000)

#plotting hist and gaussian
fig1 = plt.figure()
plt.title(r"Histogram of estimators of $\int_0^1 x^4 \mathrm{d}x$ with Gaussian fit")
plt.grid()
plt.plot(x_range, gauss(x_range, *par), label="Gaussian fit")
plt.hist(data, bins=Nr_bins, density="True", label=r"$I_N$")
plt.xlabel(r"$I_N$")
plt.ylabel(r"Prob. density $P(I_N)$")
plt.legend()
#plt.show()
plt.savefig("histGauss.pdf", dpi=200)

print("Standard error from a) is %f and the error from the Gaussian fit is %f" % (sigma_N, par[1]))

# ######################## PART C ##################

def g(x, choice):
    """improvement function for sampling

    Args:
        x (array): random number
        choice (int): type of function

    Returns:
        _type_: improved sampling
    """
    if choice == 1:
        return 2*x
    elif choice == 2:
        return 3*x**2
    elif choice == 3:
        return 4*x**3
    elif choice == 4:
        return 5*x**4


x = np.linspace(0, 1, 10000)
for i in range(1, 5):
    # draw x from the corresponding g
    p = np.asarray(random.choices(x, weights=g(x, i), k=N))

    I_N = V*average(f(p)/g(p, i), N)

    sigma_N = np.sqrt((V**2*average(f(p)*f(p)/(g(p, i)**2), N) - I_N**2)/(N-1))
    print("We get I_N %f and sigma_N %f for function %d" % (I_N, sigma_N, i))

# maximal number of n
maxN = 10
# matrix with values
s = np.array([np.zeros(maxN) for i in range(4)])
n = []
for i in range(maxN):
    n.append(int(10*2**(i+1)))

for j in range(len(n)):
    N = n[j]
    for i in range(1, 5):
        p = np.asarray(random.choices(x, weights=g(x, i), k=N))
        I_N = average(f(p)/g(p, i), N)
        sigma_N = np.sqrt((average(f(p)*f(p)/(g(p, i)**2), N) - I_N**2)/(N-1))

        s[i-1, j] = sigma_N

#label
l = [r"$g(x)=2x$", r"$g(x)=3x^2$",r"$g(x)=4x^3$",r"$g(x)=5x^4$"]

fig2 = plt.figure()
for i in range(4):
    plt.plot(n, s[i, :], label=l[i])
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid()
plt.xlabel(r"$N$")
plt.ylabel(r"$\sigma_N$")
plt.title("Standard error for different distributions")
#plt.show()
plt.savefig("loglog.pdf", dpi=200)
