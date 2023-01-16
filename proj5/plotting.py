import numpy as np
import matplotlib.pyplot as plt



# ener = np.loadtxt("energies_1.txt", delimiter=" ")


# plt.plot(ener[:, 0], ener[:, 1], label=r"$T$")
# plt.plot(ener[:, 0], ener[:, 2], label=r"$V$")
# plt.xlabel(r"$n$")
# plt.ylabel("Energies in a.u.")
# plt.legend()
# plt.grid()
# plt.show()


x = np.genfromtxt("test.txt", delimiter=" ")

# plt.hist(x, bins=40, density=True)
# plt.show()



def f(i, x, p, dt):
    return 1.0/(2 * dt**2) * ((x[i+1] - p) ** 2 + (x[i-1] - p) ** 2 - (x[i+1] - x[i]) ** 2 - (x[i-1] - x[i]) ** 2) + 1 / 8 * ((x[i+1] + p) ** 2 + (x[i-1] + p) ** 2 - (x[i+1] + x[i]) ** 2 - (x[i-1] + x[i]) ** 2)

x_n = x
x_n[287] = -0.025746224234527527

to = 21.967896671006407
vo = 0.2892641221206392

tn = 21.812650006374202
vn = 0.28944696078937715



print(tn + vn - to - vo, x[287], x_n[287])
print(f(287, x, x_n[287], 0.25))