import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema



A = np.array([np.array([i, i+2, i**2, i-2]) for i in range(-2, 2, 1)])

x = [argrelextrema(A[i,:], np.greater) for i in range(len(A))]

print(A, x)