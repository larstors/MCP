import numpy as np
import matplotlib.pyplot as plt

"""Just testing some stuff 
"""

mom = np.genfromtxt("test.txt", delimiter=" ")

n = 1
n1 = 2

coord = []
coord1 = []

for i in range(72*n, 72*n1):
    coord.append(mom[i, 0])
    coord.append(mom[i, 1])
    coord.append(mom[i, 2])

for i in range(72*n1, 72*(n1+1)):
    coord1.append(mom[i, 0])
    coord1.append(mom[i, 1])
    coord1.append(mom[i, 2])

plt.scatter(x=coord[::3], y=coord[1::3], c="red")
plt.scatter(x=coord1[::3], y=coord1[1::3], c="green")

print(coord1[3:6])

plt.show()



# c = np.genfromtxt("position_comparison.txt", delimiter=" ")

# fig = plt.figure()
# plt.hist(c[:, 0]-c[:, 1], bins=60, density=True)
# plt.axis([-1, 1, 0, 1])
# plt.show()