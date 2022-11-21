import numpy as np
import matplotlib.pyplot as plt

"""Just testing some stuff 
"""

mom = np.genfromtxt("test.txt", delimiter=" ")

print(np.shape(mom))




fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 30))

ax[0].hist(mom[:, 0], bins=30, density=True)
ax[1].hist(mom[:, 1], bins=30, density=True)
ax[2].hist(mom[:, 2], bins=30, density=True)

plt.show()
