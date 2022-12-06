import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import ctypes


# library = ctypes.CDLL("./libfun.so")
# ar = [0, 1, 2, 3]
# arr = (ctypes.c_int * len(ar))(*ar)
# print(ar, arr)


N = 30

width, height = N, N
minima, maxima = 0, 100.0 
metal_box = np.zeros((width, height), dtype=float)
threshold = 1e-3


boundary_location = []
for i in range(width):                      
    metal_box[i][0] = minima 
    metal_box[i][width-1] = 0
    metal_box[0][i] = maxima
    metal_box[width-1][i] = 0
    boundary_location.append((i,0))
    boundary_location.append((i,width-1))
    boundary_location.append((0,i))
    boundary_location.append((width-1,i))


#@jit(fastmath=True)
def jacobi(scheme, boundary, nmax, N):
    inmatrix = scheme.copy()
    outmatrix = scheme.copy()
    it = 0
    maximum = []
    average = []
    for n in range(nmax):
        for i in range(N):
            for j in range(N):
                if (i,j) in boundary:
                    pass
                else:
                    outmatrix[i, j] = (inmatrix[i-1][j] + inmatrix[i+1][j] + inmatrix[i][j-1] + inmatrix[i][j+1])/4
        inmatrix = outmatrix.copy()
        it+=1
        de = distance(boundary=boundary, a=inmatrix, b=outmatrix)
        maximum.append(de[0])
        average.append(de[1])
        if de[0] < threshold:
            print(it)
            break

    return outmatrix, it, maximum, average

def GS(scheme, boundary, nmax, N):
    outmatrix = scheme.copy()
    it = 0
    maximum = []
    average = []
    for n in range(nmax):
        inmatrix = outmatrix.copy()
        for i in range(N):
            for j in range(N):
                if (i,j) in boundary:
                    pass
                else:
                    outmatrix[i, j] = (outmatrix[i-1][j] + outmatrix[i+1][j] + outmatrix[i][j-1] + outmatrix[i][j+1])/4

        it+=1
        de = distance(boundary=boundary, a=inmatrix, b=outmatrix)
        maximum.append(de[0])
        average.append(de[1])
        if de[0] < threshold:
            print(it)
            break

    return outmatrix, it, maximum, average

def SOR(scheme, boundary, nmax, N, alpha):
    outmatrix = scheme.copy()
    inmatrix = scheme.copy()
    it = 0
    maximum = []
    average = []
    for n in range(nmax):
        for i in range(N):
            for j in range(N):
                if (i,j) in boundary:
                    pass
                else:
                    inmatrix[i, j] = (outmatrix[i-1][j] + outmatrix[i+1][j] + outmatrix[i][j-1] + outmatrix[i][j+1])/4
                    adjustment = alpha * (inmatrix[i, j] - outmatrix[i, j])
                    outmatrix[i, j] = adjustment + outmatrix[i, j]

        it+=1
        de = distance(boundary=boundary, a=inmatrix, b=outmatrix)
        maximum.append(de[0])
        average.append(de[1])
        if de[0] < threshold:
            print(it)
            break

    return outmatrix, it, maximum, average


def distance(boundary, a, b):
    average = 0
    max = 0
    n = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if (i, j) in boundary:
                pass
            else:
                n+=1
                e = np.abs(b[i, j] - (b[i+1, j]+b[i-1, j]+b[i, j+1]+b[i, j-1])/4)
                average += e
                if e > max:
                    max = e
    return max, average/n

                
alpha = [0.5, 1.0, 1.25, 1.5, 1.75, 1.99]
alpha_special = 2.5




output = jacobi(metal_box, boundary_location, 1000, width)
output2 = GS(metal_box, boundary_location, 1000, width)
#output3 = SOR(metal_box, boundary_location, 1000, width, 1.5)

maximum = []
average = []
it = []

for i in alpha:
    o = SOR(metal_box, boundary_location, 1000, width, i)
    maximum.append(o[2])
    average.append(o[3])
    it.append(o[1])

f = plt.figure()
plt.plot(np.arange(0, output[1]), output[2], label="Jacobi")
plt.plot(np.arange(0, output2[1]), output2[2], label="Gauss-Seidel")
for i in range(len(alpha)):
    plt.plot(np.arange(0, it[i]), maximum[i], label=r"SOR: $\alpha=%.2f$" % alpha[i])
plt.xlabel("Iterations")
plt.ylabel(r"$\mathrm{max}_{ij}\epsilon_{ij}$")
plt.legend()
plt.yscale("log")
plt.savefig("epsmax.pdf", dpi=200)

f1 = plt.figure()
plt.plot(np.arange(0, output[1]), output[3], label="Jacobi")
plt.plot(np.arange(0, output2[1]), output2[3], label="Gauss-Seidel")
for i in range(len(alpha)):
    plt.plot(np.arange(0, it[i]), average[i], label=r"SOR: $\alpha=%.2f$" % alpha[i])
plt.xlabel("Iterations")
plt.ylabel(r"$\langle\epsilon\rangle_{ij}$")
plt.legend()
plt.yscale("log")
plt.savefig("epsavg.pdf", dpi=200)








fig = plt.figure()
plt.imshow(output[0])
plt.colorbar()
plt.savefig("color.pdf")

fig2 = plt.figure()
plt.imshow(output2[0])
plt.colorbar()
plt.savefig("colorGS.pdf")

"""fig3 = plt.figure()
plt.imshow(output3[0])
plt.colorbar()
plt.savefig("colorSOR.pdf")
"""

"""#@jit(fastmath=True)
def solver(nmax):
    init = initial()
    A = init[0]
    b = init[1]
    x0 = b
    xnew = b
    k = np.zeros(nmax)
    kmax = np.zeros(nmax)
    xnew = jacobi(A, xnew, b, nmax, len(b))

    return xnew
"""

"""#i = initial()[0]
#np.savetxt("check.txt", i)

sol = solver(1000)

print(np.shape(sol))

fig1 = plt.figure()
plt.plot(np.arange(0, len(sol[2])), sol[1])
plt.title("aver_eps")
plt.savefig("eps.pdf")

fig2 = plt.figure()
plt.plot(np.arange(0, len(sol[2])), sol[1])
plt.title("max_eps")
plt.savefig("epsmax.pdf")


N = int(np.sqrt(len(sol[0])))
sol = sol[0].reshape(N, N)
dl = 1/N
x, y = np.meshgrid(np.linspace(0, 1, N, endpoint=True), np.linspace(0, 1, N, endpoint=True))

fig3 = plt.figure()
plt.imshow(sol)
plt.colorbar()
plt.savefig("color.pdf")
"""