import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
import ctypes


# library = ctypes.CDLL("./libfun.so")
# ar = [0, 1, 2, 3]
# arr = (ctypes.c_int * len(ar))(*ar)
# print(ar, arr)


N = 100

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


@jit(fastmath=True)
def jacobi(scheme, nmax, N):
    inmatrix = scheme.copy()
    outmatrix = scheme.copy()
    it = 0
    maximum = np.zeros(nmax)
    average = np.zeros(nmax)
    for n in range(nmax):
        for i in range(1, N-1):
            for j in range(1, N-1):
                # if (i,j) in boundary:
                #     pass
                # else:
                outmatrix[i, j] = (inmatrix[i-1][j] + inmatrix[i+1][j] + inmatrix[i][j-1] + inmatrix[i][j+1])/4
        inmatrix = outmatrix.copy()
        it+=1
        de = distance( a=inmatrix, b=outmatrix)
        maximum[n] = de[0]
        average[n] = de[1]
        if de[0] < threshold:
            print(it)
            break

    return outmatrix, it, maximum, average

#@jit(fastmath=True)
def GS(scheme, nmax, N):
    outmatrix = scheme.copy()
    it = 0
    maximum = np.zeros(nmax)
    average = np.zeros(nmax)
    for n in range(nmax):
        inmatrix = outmatrix.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                # if (i,j) in boundary:
                #     pass
                # else:
                outmatrix[i, j] = (outmatrix[i-1][j] + outmatrix[i+1][j] + outmatrix[i][j-1] + outmatrix[i][j+1])/4

        it+=1
        de = distance(a=inmatrix, b=outmatrix)
        maximum[n] = de[0]
        average[n] = de[1]
        if de[0] < threshold:
            print(it)
            break

    return outmatrix, it, maximum, average

#@jit(fastmath=True)
def SOR(scheme, nmax, N, alpha):
    outmatrix = scheme.copy()
    inmatrix = scheme.copy()
    it = 0
    maximum = np.zeros(nmax)
    average = np.zeros(nmax)
    for n in range(nmax):
        for i in range(1, N-1):
            for j in range(1, N-1):
                inmatrix[i, j] = (outmatrix[i-1][j] + outmatrix[i+1][j] + outmatrix[i][j-1] + outmatrix[i][j+1])/4
                adjustment = alpha * (inmatrix[i, j] - outmatrix[i, j])
                outmatrix[i, j] = adjustment + outmatrix[i, j]

        it+=1
        de = distance(a=inmatrix, b=outmatrix)
        maximum[n] = de[0]
        average[n] = de[1]
        if de[0] < threshold:
            print(it)
            break

    return outmatrix, it, maximum, average


@jit(fastmath=True)
def distance( a, b, N=100):
    average = 0
    max = 0
    n = 0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # if (i, j) in boundary:
            #     pass
            # else:
            n+=1
            e = np.abs(b[i, j] - (b[i+1, j]+b[i-1, j]+b[i, j+1]+b[i, j-1])/4)
            average += e
            if e > max:
                max = e
    return max, average/n

                
alpha = [0.5, 1.0, 1.25, 1.5, 1.75, 1.99]
alpha_special = 2.01


maxiteration = 100000

output = jacobi(metal_box, maxiteration, width)
output2 = GS(metal_box, maxiteration, width)
#output3 = SOR(metal_box, boundary_location, maxiteration, width, 1.5)

maximum = []
average = []
it = []

for i in alpha:
    o = SOR(metal_box, maxiteration, width, i)
    maximum.append(o[2])
    average.append(o[3])
    it.append(o[1])

f = plt.figure()
plt.plot(np.arange(0, output[1]), output[2][output[2]>0], label="Jacobi")
plt.plot(np.arange(0, output2[1]), output2[2][output2[2]>0], "-", label="Gauss-Seidel")
for i in range(len(alpha)):
    plt.plot(np.arange(0, it[i]), maximum[i][maximum[i]>0], "-.", label=r"SOR: $\alpha=%.2f$" % alpha[i])
plt.xlabel("Iterations")
plt.ylabel(r"$\mathrm{max}_{ij}\epsilon_{ij}$")
plt.legend()
plt.yscale("log")
plt.savefig("epsmax.pdf", dpi=200)

f1 = plt.figure()
plt.plot(np.arange(0, output[1]), output[3][output[2]>0], label="Jacobi")
plt.plot(np.arange(0, output2[1]), output2[3][output2[2]>0], "-",  label="Gauss-Seidel")
for i in range(len(alpha)):
    plt.plot(np.arange(0, it[i]), average[i][average[i]>0], "-.", label=r"SOR: $\alpha=%.2f$" % alpha[i])
plt.xlabel("Iterations")
plt.ylabel(r"$\langle\epsilon_{ij}\rangle_{ij}$")
plt.legend()
plt.yscale("log")
plt.savefig("epsavg.pdf", dpi=200)


oo = SOR(metal_box, 4000, width, alpha_special)

f = plt.figure()
plt.plot(np.arange(0, oo[1]), oo[2])
plt.xlabel("Iterations")
plt.ylabel(r"$\mathrm{max}_{ij}\epsilon_{ij}$")
plt.yscale("log")
plt.savefig("epsmax_high.pdf", dpi=200)

f1 = plt.figure()
plt.plot(np.arange(0, oo[1]), oo[3])
plt.xlabel("Iterations")
plt.ylabel(r"$\langle\epsilon_{ij}\rangle_{ij}$")
plt.yscale("log")
plt.savefig("epsavg_high.pdf", dpi=200)



fig = plt.figure()
plt.imshow(output[0])
c1 = plt.colorbar()
c1.set_label(r"$\phi$")
plt.savefig("color.pdf")

fig2 = plt.figure()
plt.imshow(output2[0])
c1 = plt.colorbar()
c1.set_label(r"$\phi$")
plt.savefig("colorGS.pdf")

"""fig3 = plt.figure()
plt.imshow(output3[0])
plt.colorbar()
plt.savefig("colorSOR.pdf")
"""




def anal(n, x, y, L=1):
    out = 0
    for k in range(1, n+1, 2):
        out += 400/(k*np.pi) * np.sin(k*np.pi*y/L)*np.exp(-k*np.pi*x)
    return out

A1 = np.zeros((N, N))
A2 = np.zeros((N, N))
A3 = np.zeros((N, N))
A4 = np.zeros((N, N))



for i in range(N):
    for j in range(N):
        x = i/N
        y = j/N
        A1[i, j] = anal(1, x, y)
        A2[i, j] = anal(10, x, y)
        A3[i, j] = anal(100, x, y)
        A4[i, j] = anal(1000, x, y)



k = np.linspace(0, width, N)
x, y = np.meshgrid(k, k)


fig5, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
c = ax[0, 0].imshow(A1)
ax[0, 1].imshow(A2)
ax[1, 0].imshow(A3)
ax[1, 1].imshow(A4)
fig5.colorbar(c, ax=ax)
ax[0, 0].set_title(r"$n=1$")
ax[0, 1].set_title(r"$n=10$")
ax[1, 0].set_title(r"$n=100$")
ax[1, 1].set_title(r"$n=1000$")

plt.savefig("analytical.pdf", dpi=200)

fig6 = plt.figure()
A = A4 - output2[0]
plt.imshow(A)
c1 = plt.colorbar()
c1.set_label(r"$\delta\phi$")
plt.title("Analytical(truncated) - Numerical")
plt.savefig("comparison.pdf", dpi=200)


















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