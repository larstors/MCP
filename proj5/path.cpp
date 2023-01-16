/*
A c++ script to solve the path integral task in MCP project 5

The problem is to solve for the ground state of the harmonic oscillator
*/ 

#include <vector>
#include <list>
#include <map>
#include <valarray>
#include <numeric>
#include <functional>
#include <random>
#include <iostream>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string> 
#include <iomanip>
#include <sstream>
#include <chrono>

#include "CLI11.hpp"

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>



using namespace std;
using vecd = std::vector<double>;
using vec = std::vector<int>;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;



/**
 * @brief Function to calculate the potential energy of the 1D path
 * 
 * @param pos Position at each time increment
 * @param mass Mass of oscillator
 * @param omega Frequency
 * @return double Potential energy of the harmonic oscillator
 */
double V1(vecd pos, double mass, double omega){
    double v = 0;
    // loop over positions
    for (unsigned n = 1; n < pos.size(); n++){
        v += pow(pos[n] + pos[n-1], 2);
    }
    return 0.5 * 0.25 * mass * pow(omega, 2) * v;
}

/**
 * @brief Function to calculate the kinetic energy of 1D path
 * 
 * @param pos Position at each time increment
 * @param mass Mass of oscillator
 * @param dtau Time increment
 * @return double Kinetic energy of the harmonic oscillator
 */
double T1(vecd pos, double mass, double dtau){
    double t = 0;
    // loop over positions
    for (unsigned n = 1; n < pos.size(); n++){
        t += pow(pos[n] - pos[n-1], 2);
    }
    return 0.5 * mass * 1.0 / pow(dtau, 2) * t;
}

/**
 * @brief Function to calculate the difference in S from the initial to the proposed state
 * 
 * @param x Position at each time
 * @param prop_x Proposal of change in one time increment
 * @param j Index of time increment
 * @param dtau Time increment
 * @param mass Mass
 * @param omega Frequency
 * @return double Difference in S from initial to proposed state
 */
double dS1(vecd x, double prop_x, int j, double dtau, double mass, double omega){
    /*
    I am well aware that this can become far more efficient by just taking the relevant terms,
    but at this point I cant be bothered with writing them again.
    */
    vecd prop_pos = x;
    prop_pos[j] = prop_x;

    // old kinetic and potential energy
    double Told = T1(x, mass, dtau);
    double Vold = V1(x, mass, omega);

    // New kinetic and potential energy
    double Tnew = T1(prop_pos, mass, dtau);
    double Vnew = V1(prop_pos, mass, omega);

    double ds = Tnew + Vnew - Told - Vold;

    return ds;
}


int main(){
    // output files
    ofstream energies1, dens1;
    energies1.open("energies_1.txt");
    dens1.open("density_1.txt");



    // constants
    double m = 1.0;
    double w = 1.0;
    // size of discretisation
    int N = 401;
    // time discretisation 
    double dtau = 100.0/400.0;
    // vector with positions
    vecd x (N);
    // fix the initial and final position
    x[0] = x[-1] = 0;
    // increment of spacial adjustment
    double dx = 1.0;

    // distributions
    std::mt19937 rng((std::random_device())());
    std::uniform_int_distribution<int> index(1, N-1);
    std::uniform_real_distribution<double> position_shift(0, 1);
    // Maximal number of iterations
    int max_it = 100000;
    // number of steps before equilibrium
    int n_equil = 90000;

    // doing the actual simulation
    for (int i = 0; i < max_it; i++){
        // random position gets updated
        int j = index(rng);                     
        // propose new position
        double shift = position_shift(rng);    
        double x_prop = x[j] + (1 - 2 * shift) * dx;

        // metropolis accept step
        double met_step = exp(- dtau * dS1(x, x_prop, j, dtau, m, w));
        double r = min(1.0, met_step);

        if (position_shift(rng) < r){           
            x[j] = x_prop;
        }

        
        // print energies etc into output files
        energies1 << i << " " << T1(x, m, dtau) << " " << V1(x, m, w) << endl;

        // if (i > n_equil && i%1000==0){
        //     for (unsigned n = 0; n < x.size(); n++){
        //         dens1 << x[n] << " ";
        //     }
        //     dens1 << endl;
        // }
        

    }


    energies1.close();
    dens1.close();


    return 0;

}

