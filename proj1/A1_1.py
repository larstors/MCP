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

def average(y):
    """function to calculate average

    Args:
        y (array): data to find average of

    Returns:
        float: average of y
    """
    return 1/N * np.sum(y)

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
plt.title(r"Histogram of estimators of $\int_0^1 x^4 \mathrm{d}x$ with Gaussian fit")
plt.grid()
plt.plot(x_range, gauss(x_range, *par), label="Gaussian fit")
plt.hist(data, bins=Nr_bins, density="True", label=r"$I_N$")
plt.xlabel(r"$I_N$")
plt.ylabel(r"Prob. density $P(I_N)$")
plt.legend()
plt.show()

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


x = np.linspace(0, 1, 1000)
for i in range(1, 5):
    # draw x from the corresponding g
    p = np.asarray(random.choices(x, weights=g(x, i), k=M))

    I_N = V*average(f(p)/g(p, i))

    sigma_N = np.sqrt((V**2*average(f(p)**2/g(p, i)**2) - I_N**2)/(N-1))
    print("We get I_N %f and sigma_N %f for function %d" % (I_N, sigma_N, i))

print(np.shape(p))