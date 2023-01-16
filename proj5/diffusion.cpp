/*
A c++ script to solve the DMC task in MCP project 5

The problem is to solve for the ground state of the helium atom
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
 * @brief Function to calculate the trial function given a specific configuration
 * 
 * @param r Configuration of 6 * M electron coordinates
 * @param kappa First variable
 * @param beta Second variable
 * @param alpha Third variable
 * @param M Number of electron pairs
 * @return vecd Trial function for each system
 */
vecd trial_psi(vecd r, double kappa, double beta, double alpha, int M){
    vecd psi(M);
    //cout << r.size() << " " << M << endl;

    for (int i = 0; i < M; i++){
        double r1 = sqrt(r[6*i] * r[6*i] + r[6*i + 1] * r[6*i + 1] + r[6*i + 2] * r[6*i + 2]);
        double r2 = sqrt(r[6*i + 3] * r[6*i + 3] + r[6*i  + 4] * r[6*i + 4] + r[6*i + 5] * r[6*i + 5]);
        double r12 = sqrt(pow(r[6*i] - r[6*i+3], 2) + pow(r[6*i+1] - r[6*i+4], 2) + pow(r[6*i + 2] - r[6*i+5], 2));
        psi[i] = exp(- kappa * r1) * exp( - kappa * r2) * exp(beta * r12 / (1 + alpha * r12));
    }
    return psi;
}

/**
 * @brief Function to calculate the local energy given a specific configuration
 * 
 * @param r Configuration of 6 * M electron coordinates
 * @param kappa First variable
 * @param beta Second variable
 * @param alpha Third variable
 * @param M Number of electron pairs
 * @return vecd Local energy of the M systems
 */
vecd local_energy(vecd r, double kappa, double beta, double alpha, int M){
    vecd eloc(M);
    for (int i = 0; i < M; i++){
        double r1 = sqrt(r[6*i] * r[6*i] + r[6*i + 1] * r[6*i + 1] + r[6*i + 2] * r[6*i + 2]);
        double r2 = sqrt(r[6*i + 3] * r[6*i + 3] + r[6*i  + 4] * r[6*i + 4] + r[6*i + 5] * r[6*i + 5]);
        double r12 = sqrt(pow(r[6*i] - r[6*i+3], 2) + pow(r[6*i+1] - r[6*i+4], 2) + pow(r[6*i + 2] - r[6*i+5], 2));
        double u = 1 + alpha * r12;
        eloc[i] = (kappa - 2) / r1 + (kappa - 2) / r2 + 1 / r12 * (1 - 2 * beta / pow(u, 2)) + 2 * beta * alpha / pow(u, 3) - pow(kappa, 2) - pow(beta, 2) / pow(u, 4);
        double ska = 0;
        for (int j = 0; j < 3; j++){
            ska += (r[6*i] / r1 - r[6*i + 3] / r2) * (r[6*i] - r[6*i + 3]) / r12;
        }
        eloc[i] += kappa * beta / pow(u, 2) * ska;
    }
    return eloc;


}

/**
 * @brief Function to calculate the quantum force given a specific configuration
 * 
 * @param r Configuration of 6 * M electron coordinates
 * @param kappa First variable
 * @param beta Second variable
 * @param alpha Third variable
 * @param M Number of electron pairs
 * @return vecd Force acting on the M systems
 */
vecd force(vecd r, double kappa, double beta, double alpha, int M){
    vecd f(6 * M);
    for (int i = 0; i < M; i++){
        double r1 = sqrt(r[6*i] * r[6*i] + r[6*i + 1] * r[6*i + 1] + r[6*i + 2] * r[6*i + 2]);
        double r2 = sqrt(r[6*i + 3] * r[6*i + 3] + r[6*i  + 4] * r[6*i + 4] + r[6*i + 5] * r[6*i + 5]);
        double r12 = sqrt(pow(r[6*i] - r[6*i+3], 2) + pow(r[6*i+1] - r[6*i+4], 2) + pow(r[6*i + 2] - r[6*i+5], 2));
        double u = 1 + alpha * r12;
        for (int j = 0; j < 3; j++){
            f[6 * i + j] = - 2 * kappa * r[6*i + j] / r1 + 2 * beta * (r[6*i + j] - r[6*i + j + 3])/(r12 * pow(u, 2));
            f[6 * i + j + 3] = - 2 * kappa * r[6*i + j + 3] / r2 - 2 * beta * (r[6*i + j] - r[6*i + j + 3])/(r12 * pow(u, 2));    
        }
    }
    return f;
}

/**
 * @brief Greens function for transition from r to y
 * 
 * @param r Old config
 * @param y New config
 * @param F Force on old config
 * @param dtau time discretisation
 * @param M Number of walkers
 * @return vecd Greens function transitions of M systems
 */
vecd FP_Greens(vecd r, vecd y, vecd F, double dtau, int M){
    vecd green(M);
    for (int i = 0; i < M; i++){
        double exponent = 0;
        for (int j = 0; j < 6; j++){
            exponent -= pow(y[6*i+j] - r[6*i+j] -  F[6*i+j] * dtau / 2.0, 2);
        }
        green[i] = 1.0 / sqrt(2.0 * M_PI * dtau) * exp(exponent / (2.0 * dtau));
    }
    return green;
}

/**
 * @brief Growth factor for each system
 * 
 * @param EL Local energy
 * @param Et Trial energy
 * @param dtau Time discretisation
 * @param M Number of walkers
 * @return vecd Growth factor
 */
vecd growth(vecd EL, float Et, double dtau, int M){
    vecd g(M);

    for (int i = 0; i<M; i++){
        g[i] = exp(- dtau * (EL[i] - Et));
    }
    return g;
}

/**
 * @brief Single update step with accept reject
 * 
 * @param x Old config
 * @param eta Gaussian noise for position update
 * @param met_acc Accept probability
 * @param dtau Time discretisation
 * @param kappa First parameter
 * @param beta Second parameter
 * @param alpha Third parameter
 * @param M Number of walkers
 * @return vecd Updated position
 */
vecd single_step(vecd x, vecd eta, vecd met_acc, double dtau, double kappa, double beta, double alpha, int M){
    // array with new walkers and filling it with initial stuff
    vecd new_walkers;
    for (int i = 0; i < x.size(); i++){
        new_walkers.push_back(x[i]);
    }
    // get force, rho, ...
    vecd F_old = force(x, kappa, beta, alpha, M);
    vecd psi_old = trial_psi(x, kappa, beta, alpha, M);
    vecd rho;
    

    // do the actual loop for the random walkers
    for (int i = 0; i < M; i++){
        rho.push_back(psi_old[i] * psi_old[i]);

        // update positions
        for (int n = 0; n < 6; n++){
            new_walkers[6*i + n] += F_old[6*i + n] * dtau / 2.0 + sqrt(dtau) * eta[6*i + n];
        }
    }

    vecd F_new = force(new_walkers, kappa, beta, alpha, M);
    vecd psi_new = trial_psi(new_walkers, kappa, beta, alpha, M);
    vecd green_old = FP_Greens(x, new_walkers, F_old, dtau, M);
    vecd green_new = FP_Greens(new_walkers, x, F_new, dtau, M);
    double rho_new;
    for (int i = 0; i < M; i++){
        rho_new = psi_new[i] * psi_new[i];
        double r = min(1.0, rho_new / rho[i] * green_new[i] / green_old[i]);
        if (met_acc[i] < r){
            for (int n = 0; n < 6; n++){
                x[6*i + n] = new_walkers[6*i + n];
            }
        }
    }
    return x;
}



int main(){
    // output files
    ofstream outfile;
    outfile.open("energy.txt");
    
    
    // rng
    std::mt19937 rng((std::random_device())());
    std::uniform_real_distribution<double> position_shift(0, 1);
    std::normal_distribution<double> noise(0, 1);

    // for iteration
    int max_it = 30000;
    int n_eq = 10000;

    // initial number of walkers and energy
    int M0 = 300;
    int M = M0;
    double E0 = -2.891;
    double ET = E0;
    
    // constant that fullfil cusp conditions
    double kappa = 2.0;
    double beta = 0.5;
    double alpha = 0.18;

    double dtau = 0.03;
    // start with array of 
    vecd x (6 * M0);
    // fill initial array with random config
    for (int j = 0; j < 6 * M0; j++) x[j] = position_shift(rng);
    // energy average (after equilibration)
    double av_E = 0;
    double check = 0;


    for (int it = 0; it < max_it; it++){
        vecd et;
        vecd acc;
        for (int i = 0; i < M; i++){
            et.push_back(noise(rng));
            acc.push_back(position_shift(rng));
        }
        
        x = single_step(x, et, acc, dtau, kappa, beta, alpha, M);


        // adjusting with birth / death
        // for saving positions
        // TODO
        vecd new_pos;
        // growth factor
        vecd loc_E = local_energy(x, kappa, beta, alpha, M);
        vecd q = growth(loc_E, ET, dtau, M);

        for (int i = 0; i < M; i++){
            if (q[i] <= 1 && position_shift(rng) < q[i]){
                for (int n = 0; n < 6; n++){
                    new_pos.push_back(x[6*i + n]);
                }
            }

            else if (q[i] > 1){
                int m = floor(q[i] + position_shift(rng));
                for (int j = 0; j < m; j++){
                    for (int n = 0; n < 6; n++){
                        new_pos.push_back(x[6*i + n]);
                    }
                }
            }
        }
        
        // take new positions
        x = new_pos;

        // adjust M
        M = x.size()/6;

        // adjusting the energy
        ET = E0 + log(double(M0) / double(M));

        outfile << it << " " << ET << endl;
        if (it > n_eq){
            check += 1;
            av_E += ET;
        }
    }

    cout << "Average energy after equilibration is " << av_E / check << endl;


    return 0;
}