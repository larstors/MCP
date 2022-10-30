import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import random
from matplotlib.lines import Line2D

def neighbours(x, y, lattice):
    """Function to identify the free neighbours of each lattice site. Note that it has closed boundaries.

    Args:
        x (int): x coordinate on the lattice. Ranging from 0 to d-1
        y (int): y coordinate on the lattice. Ranging from 0 to d-1
        lattice (array): 2D array containing the lattice configuration of the different monomers

    Returns:
        list: list of coordinates of free neighbours
    """
    n = []
    if x+1 < 31 and lattice[x+1, y] == 0:
        n.append([x+1, y])
    
    if x-1 > -1 and lattice[x-1, y] == 0:
        n.append([x-1, y])
    
    if y+1 < 31 and lattice[x, y+1] == 0:
        n.append([x, y+1])
    
    if y-1 > -1 and lattice[x, y-1] == 0:
        n.append([x, y-1])
    
    return n

def neighboursFree(x, y):
    """Function to identify the all neighbours of each lattice site. Note that it has closed boundaries.

    Args:
        x (int): x coordinate on the lattice. Ranging from 0 to d-1
        y (int): y coordinate on the lattice. Ranging from 0 to d-1

    Returns:
        list: list of coordinates of all neighbours
    """
    n = []
    if x+1 < 31:
        n.append([x+1, y])
    
    if x-1 > -1:
        n.append([x-1, y])
    
    if y+1 < 31:
        n.append([x, y+1])
    
    if y-1 > -1:
        n.append([x, y-1])
    
    return n



def visual(l, p, name, e, L, d=31):
    """Function to visualise the lattice

    Args:
        l (array): 2D array of the lattice
        p (array): array containing the path of the random walker
        name (string): name for savefile
        e (int): energy of path
        L (int): Length of path
        d (int, optional): dimension of the lattice. Defaults to 31.
    """
    # name for different parts of the plot
    lab = ["Substrate", "P monomer", "H monomer", "Chain"]
    fig = plt.figure()
    
    # draw the path, there is probably a more efficient way of doing it
    for i in range(len(p)-1):
        x = [p[i][0]+1, p[i+1][0]+1]
        y = [p[i][1]+1, p[i+1][1]+1]
        plt.plot(x, y, "r-")
        
    # title 
    plt.title(r"Example for system with $E=%d\epsilon$ and $L=%d$" % (e, L))
    
    # add label to the path
    line = Line2D([0], [0], color="red", linestyle='-')
    
    #plotting actual lattice configuration
    x = np.array([np.arange(1, d+1) for i in range(31)]).T
    y = x.T
    scat = plt.scatter(x, y, c=l)
    # add legend
    hand = scat.legend_elements()[0]
    hand.append(line)
    legend1=plt.legend(handles=hand, labels=lab, framealpha=1)
    plt.savefig(name+".pdf", dpi=200)



def energy(lattice, path, d=31, eps=1):
    """Function to calculate energy of path

    Args:
        lattice (array): 2D array of lattice configuration
        path (array): array containing path
        d (int, optional): dimension of lattice. Defaults to 31.
        eps (int, optional): scale of energy. Defaults to 1.

    Returns:
        int: energy of path
    """
    # count variable for H-H links
    count = 0
    # loop over inner path
    for i in range(1, len(path)-1):
        #coordinate of location
        x, y = path[i]
        # all neighbours
        r = neighboursFree(x, y)
        # is it H?
        if lattice[x, y] == 2:
            # loop over neighbours
            for j in range(len(r)):
                #determine if off-path H-H connection 
                if lattice[r[j][0], r[j][1]] == 2 and r[j] != path[i+1] and r[j] != path[i-1]:
                    count += 1

    # Do the same for beginning
    x, y = path[0]
    r = neighboursFree(x, y)
    if lattice[x, y] == 2:
        for j in range(len(r)):
            if lattice[r[j][0], r[j][1]] == 2 and r[j] != path[1]:
                count += 1

    # and end of path
    x, y = path[-1]
    r = neighboursFree(x, y)
    if lattice[x, y] == 2:
        for j in range(len(r)):
            if lattice[r[j][0], r[j][1]] == 2 and r[j] != path[-2]:
                count += 1
    
    # divide by to because otherwise double count all links
    count = int(count/2)

    return -eps*count

def energy2(lattice, path, d=31, eps=1):
    """Alternative function to calculate energy of path

    Args:
        lattice (array): 2D array of lattice configuration
        path (array): array containing path
        d (int, optional): dimension of lattice. Defaults to 31.
        eps (int, optional): scale of energy. Defaults to 1.

    Returns:
        int: energy of path
    """
    # count for all H-H links and for those on the path
    count = 0
    countpath = 0

    # H-H links on path
    for i in range(len(path)-1):
        x, y = path[i]
        x1, y1 = path[i+1]
        if (lattice[x, y] == 2 and lattice[x1, y1]==2):
            countpath+=1
    
    # H-H links on lattice
    for i in path:
        x, y = i
        if lattice[x, y] == 2:
            r = neighboursFree(x, y)
            for j in r:
                if lattice[j[0], j[1]]==2:
                    count += 1
    
    # H-H off-path links are just difference
    f = int(count/2) - countpath

    return -eps * f


# dimensions of lattice
d = 31

#probs
ph = 0.7
pp = 0.3

#Nr of iterations
M = 1000

# energy
epsilon = 1




# list for energy and lengths
E = []
L = []

#energies for outputs
en = [-45, -10, 0]
is_output = [0, 0, 0] # 0 if not made plot yet, 1 otherwise

# loop over M individual runs
for i in range(M):
    # initialize empty lattice and path
    lattice = np.array([np.zeros(d) for i in range(d)])
    path = []
    # place initial monomer
    r = np.random.uniform(0, 1, 1)
    if r < ph:
        lattice[15, 15] = 2
    else:
        lattice[15, 15] = 1

    # add initial position
    path.append([15, 15])

    # walker moves as long as there is an empty spot as neighbouring site
    while len(neighbours(path[-1][0], path[-1][1], lattice)) != 0:
        # neighboours
        neig = neighbours(path[-1][0], path[-1][1], lattice)
        # choose random direction
        r = np.random.randint(len(neig))
        x, y = neig[r]
        path.append([x, y])
        #place monomer
        r = np.random.uniform(0, 1, 1)
        if r < ph:
            lattice[x, y] = 2
        else:
            lattice[x, y] = 1

        # if walker has ended save energy and length
        if len(neighbours(x, y, lattice)) == 0:
            e = energy(lattice, path)
            E.append(e)
            L.append(len(path))

            # plots for high, medium and low energies
            if (e < en[0] and is_output[0] == 0):
                visual(lattice, path, "high_energy_example", e, len(path))
                is_output[0] = 1
            elif (e < en[1] and e > en[0] and is_output[1] == 0):
                visual(lattice, path, "medium_energy_example", e, len(path))
                is_output[1] = 1
            elif (e < en[2] and e > en[1] and is_output[2] == 0):
                visual(lattice, path, "low_energy_example", e, len(path))
                is_output[2] = 1


# Plotting of histogram and correlation
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.4, hspace=None)
ax[0].hist(E, bins=30, density=True)
ax[0].set_xlabel(r"$E$ in $\epsilon$")
ax[0].set_ylabel(r"$p(E)$")
ax[1].hist(L, bins=30, density=True)
ax[1].set_xlabel(r"$L$")
ax[1].set_ylabel(r"$p(L)$")
plt.savefig("3energy_l_hist.pdf", dpi=200, bbox_inches="tight")


fig2, ax1= plt.subplots()
h = ax1.hist2d(E, L, bins=30, density=True)
fig.colorbar(h[3], ax=ax1)
ax1.set_xlabel(r"$E$ in $\epsilon$")
ax1.set_ylabel(r"$L$")
plt.savefig("3_hist_correlation.pdf", dpi=200)
