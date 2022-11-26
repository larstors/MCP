import numpy as np
import matplotlib.pyplot as plt

"""Just testing some stuff 
"""

mom = np.genfromtxt("test.txt", delimiter=" ")

n = 3
n1 = n+1

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

fig, ax = plt.subplots()
ax.scatter(x=coord[::3], y=coord[1::3], c="red")
ax.scatter(x=coord1[::3], y=coord1[1::3], c="green")

ax.set_aspect("equal")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(r"First two layers forming a (christmas) FCC")
print(coord1[3:6])

plt.savefig("christmas_fcc.pdf", dpi=200)



# c = np.genfromtxt("position_comparison.txt", delimiter=" ")

# fig = plt.figure()
# plt.hist(c[:, 0]-c[:, 1], bins=30, density=True)
# #plt.axis([-12, 12, 0, 1])
# plt.show()