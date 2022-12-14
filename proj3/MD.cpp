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


using namespace std;
using vecd = std::vector<long double>;
using vec = std::vector<int>;
using hist = std::vector<double>;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template<typename Engine> class ClusterWriter;


struct Parameters {
    double      mass = 48.0;     // assuming the particles have the same mass
    vecd         L {10, 20, 30}; // dimensions of box
    double      T0 = 1;         // initial temperature
    int    N = 100;             // number of particles
    double rstar = 3.3;         // for book keeping
    double a = 1;               // lattice constant 
    double hist_depth = 1;      // depth of histogram (needed to declare the array)
    double dx = 0.05;          // step size of histogram
    double rho  = 0.8;
};

template<typename Engine>
class MD {

    friend class ClusterWriter<Engine>;

    struct Particle{
        vecd position = vecd(3);    // position of partilce
        vecd velocity = vecd(3);    // velocity of particle
        vecd neigh = vecd(864); // for book keeping
    };


    const long double kb = 1;//1.38e-23;

    Parameters P; // A local copy of the model parameters
    std::vector<Particle> past, present, future; // particles at t-dt, t, t+dt
    Engine& rng; // Source of noise: this is a reference as there should only be one of these!
    std::normal_distribution<double> maxwell;
    hist hi;

    /**
     * @brief Force on each particle in each coordinate using Lenard-Jones potential (could probs move this into 
     *        public as it doesnt really need to stay private, no?)
     * 
     * @return vecd 
     */
    vecd force(int o, int burn){
        // vector with force for each particle and each component
        vecd f(P.N * 3);
        if (o == 0){
            //vecd dist = minimal_distance();
            // loop over output for particles
            for (int i = 0; i < P.N; i++){
                // loop over all particles
                for (int j = 0; j < P.N; j++){
                    if (j > i){
                        // calculating optimal position for image
                        vecd new_r (3);

                        //int l = 0;
                        for (int k = 0; k < 3; k++){
                            new_r[k] = future[j].position[k];
                            if (fabs(future[i].position[k] - future[j].position[k]) < double(P.L[0]) / 2.0){
                                new_r[k] = future[j].position[k];
                                //l = 1;
                            }
                            else if (future[i].position[k] - future[j].position[k] > double(P.L[0]) / 2.0){
                                new_r[k] = future[j].position[k] + P.L[0];
                                //l = 2;
                            }
                            else if (future[i].position[k] - future[j].position[k] < - double(P.L[0]) / 2.0){
                                new_r[k] = future[j].position[k] - P.L[0];
                                //l = 3;
                            }
                            //cout << l  << " " << future[i].position[k] - new_r[k] << " " << future[i].position[k] - future[j].position[k] << " " << P.L[0] * 0.5 << endl;
                            assert(fabs(future[i].position[k] - new_r[k]) <= P.L[0] * 0.5);
                            
                        }
                        
                        double dist = distance(future[i].position, new_r);
                        //cout << i << " " << j << endl;
                        //for (int k = 0; k < 3; k++) cout << past[i].position[k] << " " << past[j].position[k] << endl;
                        assert(dist > 0);
                        if (dist < 2.5){
                            if (burn > 0){
                                int count = 0;
                                double x = P.dx;
                                while (x < dist){
                                    count++;
                                    x += P.dx;
                                }
                                hi[count]+=1;
                            }
                            
                            for (int k = 0; k < 3; k++){
                                f[3*i + k] += 48.0*(1 * pow(dist, -14) - 0.5 * pow(dist, -8)) * (future[i].position[k] - new_r[k]);
                                f[3*j + k] -= 48.0*(1 * pow(dist, -14) - 0.5 * pow(dist, -8)) * (future[i].position[k] - new_r[k]);
                                assert(isnan(f[3*j + k]) == false);
                                assert(isnan(f[3*i + k]) == false);
                            }
                        }
                        
                    }
                }
            }
        }
        else {
            // loop over output for particles
            for (int i = 0; i < P.N; i++){
                // loop over all particles
                for (int j = i+1; j < P.N; j++){
                    if (future[i].neigh[j] != 0){

                        vecd new_r (3);
                        
                        // calculating optimal position for image
                        for (int k = 0; k < 3; k++){
                            new_r[k] = future[j].position[k];
                            if (fabs(future[i].position[k] - future[j].position[k]) < double(P.L[0]) / 2.0){
                                new_r[k] = future[j].position[k];
                            }
                            else if (future[i].position[k] - future[j].position[k] > double(P.L[0]) / 2.0){
                                new_r[k] = future[j].position[k] + P.L[0];
                            }
                            else if (future[i].position[k] - future[j].position[k] < - double(P.L[0]) / 2.0){
                                new_r[k] = future[j].position[k] - P.L[0];
                            }
                            assert(fabs(future[i].position[k] - new_r[k]) <= P.L[0] * 0.5);
                        }
                        double dist = distance(future[i].position, new_r);
                        if (dist < 2.5){
                            if (burn > 0){
                                int count = 0;
                                double x = P.dx;
                                while (x < dist){
                                    count++;
                                    x += P.dx;
                                }
                                hi[count]+=1;
                            }
                            for (int k = 0; k < 3; k++){
                                f[3*i + k] += 48.0*(1 / pow(dist, 14) - 0.5 / pow(dist, 8)) * (future[i].position[k] - new_r[k]);
                                f[3*j + k] -= 48.0*(1 / pow(dist, 14) - 0.5 / pow(dist, 8)) * (future[i].position[k] - new_r[k]);
                                assert(isnan(f[3*j + k]) == false);
                                assert(isnan(f[3*i + k]) == false);
                            }
                        }
                        
                    }
                }
            }
        }
        return f;
    }

    /**
     * @brief calculate the distance between two points for arbitrary boundary combinations
     * 
     * @param p1 
     * @param p2 
     * @param n 
     * @return double 
     */
    double distance(vecd p1, vecd p2){
        // distance variable
        double dist = 0;

        for (int i = 0; i < 3; i++){
            dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }

        return pow(dist, 1.0/2.0);
    }

    /**
     * @brief 
     * 
     * @return vecd 
     */
    vecd minimal_distance(){
        vecd output (P.N * P.N);

        // loop over output for particles
        for (int i = 0; i < P.N; i++){
            // loop over all particles
            for (int j = 0; j < P.N; j++){
                if (j > i){
                    // only want difference between different particles
                    vecd new_r (3);
                    for (int k = 0; k < 3; k++){
                        if (fabs(future[i].position[k] - future[j].position[k]) < double(P.L[0]) / 2){
                            new_r[k] = future[j].position[k];
                        }
                        else if (future[i].position[k] - future[j].position[k] > double(P.L[0]) / 2){
                            new_r[k] = future[j].position[k] + P.L[0];
                        }
                        else if (future[i].position[k] - future[j].position[k] < - double(P.L[0]) / 2){
                            new_r[k] = future[j].position[k] - P.L[0];
                        }
                    }
                    output[P.N * i + j] = distance(future[i].position, new_r);
                }
                else {
                    output[P.N * i + j] = 0;
                }
            }
        }
        return output;
    }

    double lenJones(double r){
        return -48 * (1/pow(r, 13) - 0.5/pow(r, 7));
    }

    double dpotential(){
        double output = 0;
        // loop over output for particles
        for (int i = 0; i < P.N; i++){
            // loop over all particles
            for (int j = i+1; j < P.N; j++){
                vecd r = future[j].position;
                for (int k = 0; k < 3; k++){
                        if (fabs(future[i].position[k] - future[j].position[k]) < double(P.L[0]) / 2.0){
                            r[k] = future[j].position[k];
                        }
                        else if (future[i].position[k] - future[j].position[k] > double(P.L[0]) / 2.0){
                            r[k] = future[j].position[k] + P.L[0];
                        }
                        else if (future[i].position[k] - future[j].position[k] < - double(P.L[0]) / 2.0){
                            r[k] = future[j].position[k] - P.L[0];
                        }
                    assert(fabs(future[i].position[k] - r[k]) <= P.L[0] * 0.5);
                    }
                double rij = distance(future[i].position, r);
                output+= lenJones(rij) * rij;
            }
        }
        return 1.0 / (3.0 * double(P.N) * temp()) * output;
    }

    void table(){
        //vecd output (P.N * P.N);
        
        // loop over output for particles
        for (int i = 0; i < P.N; i++){
            // loop over all particles
            for (int j = i; j < P.N; j++){
                if (j > i){
                    
                    // only want difference between different particles
                    vecd new_r (3);
                    for (int k = 0; k < 3; k++){
                        new_r[k] = future[j].position[k];
                        if (fabs(future[i].position[k] - future[j].position[k]) < double(P.L[0]) / 2.0){
                            new_r[k] = future[j].position[k];
                        }
                        else if (future[i].position[k] - future[j].position[k] > double(P.L[0]) / 2.0){
                            new_r[k] = future[j].position[k] + P.L[0];
                        }
                        else if (future[i].position[k] - future[j].position[k] < - double(P.L[0]) / 2.0){
                            new_r[k] = future[j].position[k] - P.L[0];
                        }
                        assert(fabs(future[i].position[k] - new_r[k]) <= P.L[0] * 0.5);
                    }
                    double d = distance(future[i].position, new_r);
                    if (d < P.rstar){
                        future[i].neigh[j] += d;
                        future[j].neigh[i] += d;
                    }
                }
            }
        }
        //return output;
        
    }

    double melting(){
        double rho_k = 0;
        double k = 4 * M_PI / P.a;
        for (int i = 0; i < P.N; i++){
            for (int j = 0; j < 3; j++){
                rho_k += cos(k * future[i].position[j]);
            }
        }
        return rho_k;
    }

    void adjust_temp(){
        // adjusting temperature
        double T = temp();
        double lambda = sqrt(P.T0 / T);
        for (int i = 0; i < P.N; i++){
            for (int k = 0; k < 3; k++){
                future[i].velocity[k] *= lambda;
            }
        }
    }

    public:
        // initialisation of needed quantities
        MD(const Parameters& P, Engine& rng):
        P(P),
        past(P.N),
        present(P.N),
        future(P.N),
        rng(rng),
        maxwell(0, sqrt(kb * P.T0 / P.mass)),
        hi(P.hist_depth)
        {   
            
            // to achieve p_tot = 0 we need a vector for the mean
            vecd mean_velocity {0, 0, 0};
            // initialise the position and velocity of each particle drawn from random distributions
            for (int i = 0; i < P.N; i++){
                for (int k = 0; k < 3; k++){
                    // draw velocity from normal distribution, respectively
                    past[i].velocity[k] = maxwell(rng);
                    // add the velocity to the total, i.e. mean, velocity
                    mean_velocity[k] += past[i].velocity[k];
                }
            }

            // placing particles on fcc lattice structure

            
            // for z coordinate
            for (int j = 0; j < 12; j++){
                // for y
                for (int k = 0; k < 12; k++){
                    // for x
                    for (int l = 0; l < 6; l++){
                        //if (72*j + 6*k + l == 2 || 72*j + 6*k + l == 6) cout << j << " " << k << " " << l << endl;
                        // TODO CLEAN THIS MESS!!
                        past[72*j + 6*k + l].position[2] = 1e-10 + j/2 * P.a + j%2*P.a/2;// z coordinate
                        if (j%2 == 0){
                            past[72*j + 6*k + l].position[0] = 1e-10 + (l * P.a + k%2 * P.a/2);// x coordinate
                            past[72*j + 6*k + l].position[1] = 1e-10 + ((k)/2 * P.a + k%2*P.a/2);// y coordinate
                            //if (72*j + 6*k + l == 2 || 72*j + 6*k + l == 6) cout << 1e-10 + (l * P.a + k%2 * P.a/2) << " " << 1e-10 + ((k+1)/2 * P.a + k%2*P.a/2) << endl;
                        }
                        else {
                            past[72*j + 6*k + l].position[0] = 1e-10 + (l * P.a + k%2 * P.a/2) +(P.a/2 - k%2*P.a);// x coordinate
                            past[72*j + 6*k + l].position[1] = 1e-10 + ((k)/2 * P.a + k%2*P.a/2);// y coordinate
                            //if (72*j + 6*k + l == 2 || 72*j + 6*k + l == 6) cout << 1e-10 + (l * P.a + k%2 * P.a/2) +(P.a/2 - k%2*P.a) << " " << 1e-10 + ((k+1)/2 * P.a + k%2*P.a/2) << endl;
                        }
                        if (past[72*j + 6*k + l].position[1] > P.L[0]) cout << j << " " << k << " " << l << endl;
                    }
                }
            }

            // normalise the mean velocity so that the sum goes to zero
            for (int i = 0; i < 3; i++){
                mean_velocity[i] /= double(P.N);
            }

            // remove the mean from each particle so that the total velocity is 0
            for (int i = 0; i < P.N; i++){
                for (int k = 0; k < 3; k++){
                    past[i].velocity[k] -= mean_velocity[k];
                }
            }

            for (int i = 0; i < hi.size(); i++){
                hi[i] = 0;
            }
            
            //just testing some stuff
             ofstream outfile;
             outfile.open("test.txt");
             for (int i = 0; i < P.N; i++){
                 for (int k = 0; k < 3; k++){
                     outfile << past[i].velocity[k] << " ";
                     assert(isnan(past[i].position[k]) == false);
                     if (k == 2) outfile << endl;
                 }
             }
             outfile.close();
        }

        /**
         * @brief propagation of the system using (normal) verlet
         * 
         * @param t total time of simulation
         * @param tburn burnin period of the run
         * @param h time step
         * 
         * @return vecd idk what to return here yet....
         */
        /* vecd verlet(double t, double tburn, double h, int o){
            
        //     for (int n = 0; n * h < t; n++){
                
        //         if (n == 0){
        //             for (int i = 0; i < P.N; i++){
        //                 for (int k = 0; k < 3; k++){
        //                     present[i].position[k] = past[i].position[k] + h * past[i].velocity[k] / P.mass + h*h / 2.0;
        //                 }
        //             }
        //         }

        //         vecd F = force(o);
        //         // update the "future"
        //         for (int i = 0; i < P.N; i++){
        //             for (int k = 0; k < 3; k++){
        //                 // next position
        //                 future[i].position[k] = 2*present[i].position[k] - past[i].position[k] + h*h * F[3*i + k];
        //                 // we have to calculate the new velocity before applying boundary conditions -> otherwise
        //                 // the velocity will be extremely large...
        //                 future[i].velocity[k] = (future[i].position[k] - past[i].position[k]) / (2 * h);
        //                 // we need to account for periodic boundary conditions
        //                 if (future[i].position[k] < 0) future[i].position[k] += P.L[k];
        //                 else if (future[i].position[k] > P.L[k]) future[i].position[k] -= P.L[k]; 

        //                 // update past/present/future
        //                 past[i].position[k] = present[i].position[k];
        //                 present[i].position[k] = future[i].position[k];

        //             }
        //         }
                
        //         // calculating the wanted stuff after equilibrating the system
        //         if (n * h > tburn){

        //         }
        //     }

        //     return present[0].position;
        // }
        */

        void velocity_verlet(double t, double tburn, double h, int o, int adj){
            // average temp, iterator
            vecd averageTemp;
            vecd averagePres;
            double TP = 0;
            double Tav = 0;
            int count = 0;
            int data_block = 0;
            double pressure = 0;
            double cc = 0;
            // without table
            if (o == 0){
                ofstream melt, vel_dist, tem, histo, msd, pres;
                melt.open("melting_factor_f.txt");
                vel_dist.open("vel_dist_a.txt");
                tem.open("temp_f.txt");
                histo.open("histogram_f.txt");
                msd.open("msd_f.txt");
                pres.open("pressure.txt");
                int z = 0;
                for (int n = 0; n * h < t; n++){
                    if (n * h > tburn) z++;
                    // just adding initial conditions
                    if (n == 0){
                        for (int i = 0; i < P.N; i++){
                            for (int k = 0; k < 3; k++){
                                present[i].velocity[k] = past[i].velocity[k];
                                future[i].velocity[k] = past[i].velocity[k];
                                present[i].position[k] = past[i].position[k];
                                future[i].position[k] = past[i].position[k];
                                assert(isnan(present[i].position[k]) == false);
                                assert(isnan(future[i].position[k]) == false);
                            }
                        }
                        tem << 0 << " " << temp() << endl;
                        melt << 0 << " " << melting() << endl;
                        msd << 0 << " " << MSD() << endl;
                    }

                    
                    
                    //int c = 0;
                    vecd F = force(o, z); // this has 3N dimensions
                    for (int i = 0; i < P.N; i++){
                        for (int k = 0; k < 3; k++){
                            future[i].velocity[k] += h/(2.0*P.mass) * F[3*i + k];
                            future[i].position[k] = future[i].position[k] + h*future[i].velocity[k];
                            if (future[i].position[k] > P.L[k]) future[i].position[k] -= P.L[k];
                            else if (future[i].position[k] < 0) future[i].position[k] += P.L[k];
                        }
                    }
                    F = force(o, z);
                    for (int i = 0; i < P.N; i++){
                        for (int k = 0; k < 3; k++){
                            future[i].velocity[k] = future[i].velocity[k] + h/(2.0*P.mass) * F[3*i + k];
                            //present[i].velocity[k] = future[i].velocity[k];
                            //present[i].position[k] = future[i].position[k];
                        }
                    }
                    
                    
                    
                    // adjust temperature
                    if (n%20==0 && adj==1 && n>0) adjust_temp();

                    // for (int i = 0; i < P.N; i++){
                    //     for (int k = 0; k < 3; k++){
                    //         present[i].velocity[k] = future[i].velocity[k];
                    //         present[i].position[k] = future[i].position[k];
                    //     }
                    // }

                    // after equilibrating, going to use data blocking
                    if (n * h > tburn){
                        if (data_block == 0){
                            data_block = 40;
                            if (count > 0) {
                                averageTemp.push_back(Tav / double(data_block + 1));
                                averagePres.push_back(TP / double(data_block + 1));
                            }
                            Tav = temp();
                            TP = 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * P.T0) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3));
                            count++;
                            // length of each data block
                        }
                        
                        else if (data_block > 0){
                            Tav += temp();
                            TP += 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * P.T0) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3));
                            data_block--;
                        }

                    }

                    melt << (n+1) * h << " " << melting() << endl;
                    tem << (n+1) * h << " " << temp() << endl;
                    msd << (n+1) * h << " " << MSD() << endl;
                    pres << (n+1) * h << " " << 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * P.T0) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3)) << " " << 1 - dpotential()  << endl;
                    pressure+= 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * P.T0) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3));
                    cc++;
                }

                for (int i = 0; i < P.N; i++){
                    double v = 0;
                    for (int k = 0; k < 3; k++){
                        vel_dist << future[i].velocity[k] << " ";
                        v += pow(future[i].velocity[k], 2);
                    }
                    vel_dist << sqrt(v) << endl;
                }
                

                for (int i = 0; i < hi.size(); i++){
                    histo << hi[i] / (4*M_PI * P.dx * pow((i+1) * P.dx, 2) * z) << endl;
                }
                melt.close();
                tem.close();
                msd.close();
                pres.close();
                vel_dist.close();
                histo.close();
            }



            // with table
            else {
                // all the output we have
                ofstream melt_tab, vel_dist_table, temp_table, histogram_tab, msd_tab, pres_tab;
                melt_tab.open("melting_factor_tab_16_.txt");
                // open files
            
                vel_dist_table.open("vel_dist_tab_16_e.txt");
                temp_table.open("temp_table_16_e.txt");

                histogram_tab.open("histogram_tab_16_.txt");
                msd_tab.open("msd_tab_.txt");
                pres_tab.open("pressure_tab_e.txt");

                int z = 0;
                int check = 0;
                for (int n = 0; n * h < t; n++){
                    if (n * h > tburn) z++;
                    if (check == 0){
                        check = o;
                        for (int i = 0; i < P.N; i++){
                            for (int k = 0; k < P.N; k++){
                                future[i].neigh[k] = 0;
                            }
                        }
                        
                        table();
                        
                        // just adding initial conditions
                        if (n == 0){
                            for (int i = 0; i < P.N; i++){
                                for (int k = 0; k < 3; k++){
                                    present[i].velocity[k] = past[i].velocity[k];
                                    future[i].velocity[k] = past[i].velocity[k];
                                    present[i].position[k] = past[i].position[k];
                                    future[i].position[k] = past[i].position[k];
                                }
                            }
                            melt_tab << 0 << " " << melting() << endl;
                            temp_table << 0 << " " << temp() << endl;
                            msd_tab << 0 << " " << MSD() << endl;
                        }
                        
                        //int c = 0;
                        
                        vecd F = force(o, z); // this has 3N dimensions

                        for (int i = 0; i < P.N; i++){
                            for (int k = 0; k < 3; k++){
                                present[i].velocity[k] += h/(2.0*P.mass) * F[3*i + k];
                                future[i].position[k] = present[i].position[k] + h*present[i].velocity[k];
                                if (future[i].position[k] > P.L[k]) future[i].position[k] -= P.L[k];
                                else if (future[i].position[k] < 0) future[i].position[k] += P.L[k];
                                // if (fabs(future[i].position[k]) > P.L[0] && c==0){
                                //     cout << "here" << endl;
                                //     cout << future[i].position[k] << endl; 
                                //     c++;
                                // }
                            }
                        }
                        
                        F = force(o, z);
                        for (int i = 0; i < P.N; i++){
                            for (int k = 0; k < 3; k++){
                                future[i].velocity[k] = present[i].velocity[k] + h/(2.0*P.mass) * F[3*i + k];
                                //present[i].velocity[k] = future[i].velocity[k];
                                //present[i].position[k] = future[i].position[k];
                            }
                        }
                    }

                    
                    else {
                        check--;

                        vecd F = force(o, z); // this has 3N dimensions
                        for (int i = 0; i < P.N; i++){
                            for (int k = 0; k < 3; k++){
                                present[i].velocity[k] += h/(2*P.mass) * F[3*i + k];
                                future[i].position[k] = present[i].position[k] + h*present[i].velocity[k];
                                if (future[i].position[k] > P.L[k]) future[i].position[k] -= P.L[k];
                                else if (future[i].position[k] < 0) future[i].position[k] += P.L[k];
                                // if (fabs(future[i].position[k]) > P.L[0] && c==0){
                                //     cout << "here" << endl;
                                //     cout << future[i].position[k] << endl; 
                                //     c++;
                                // }
                            }
                        }
                        F = force(o, z);
                        for (int i = 0; i < P.N; i++){
                            for (int k = 0; k < 3; k++){
                                future[i].velocity[k] = present[i].velocity[k] + h/(2*P.mass) * F[3*i + k];
                                //present[i].velocity[k] = future[i].velocity[k];
                                //present[i].position[k] = future[i].position[k];
                            }
                        }
                        
                    }

                    // adjust temperature
                    if (n%20==0 && adj==1 && n>0) adjust_temp();

                    for (int i = 0; i < P.N; i++){
                        for (int k = 0; k < 3; k++){
                            present[i].velocity[k] = future[i].velocity[k];
                            present[i].position[k] = future[i].position[k];
                        }
                    }
                    // after equilibrating, going to use data blocking
                    if (n * h > tburn){
                        if (data_block == 0){
                            data_block = 40;
                            if (count > 0) {
                                averageTemp.push_back(Tav / double(data_block + 1));
                                averagePres.push_back(TP / double(data_block + 1));
                            }
                            Tav = temp();
                            TP = 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * temp()) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3));
                            count++;
                            // length of each data block
                        }
                        
                        else if (data_block > 0){
                            Tav += temp();
                            TP += 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * temp()) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3));
                            data_block--;
                        }
                    }


                    melt_tab << (n+1) * h << " " << melting() << endl;
                    temp_table << (n+1) * h << " " << temp() << endl;
                    msd_tab << (n+1) * h << " " << MSD() << endl;
                    pres_tab << (n+1) * h << " " << 1 - dpotential() - 2*M_PI*P.rho/ (3.0 * temp()) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3)) << " " << 1 - dpotential() << endl;
                    pressure+=1 - dpotential() - 2*M_PI*P.rho/ (3.0 * temp()) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3));
                    cc++;
                }
                for (int i = 0; i < P.N; i++){
                    double v = 0;
                    for (int k = 0; k < 3; k++){
                        vel_dist_table << future[i].velocity[k] << " ";
                        v += pow(future[i].velocity[k], 2);
                    }
                    vel_dist_table << sqrt(v) << endl;
                }

                for (int i = 0; i < hi.size(); i++){
                    histogram_tab << hi[i] / double(4*M_PI * P.dx * pow((i+1) * P.dx, 2) * z) << endl;
                }
                melt_tab.close();
                temp_table.close();
                msd_tab.close();
                pres_tab.close();
                vel_dist_table.close();
                histogram_tab.close();

            }
            
            // average temp and variance
            Tav = 0;
            double dTav = 0;
            for (int i = 0; i < averageTemp.size(); i++){
                Tav += averageTemp[i];
                dTav += pow(averageTemp[i], 2);
            }
            Tav /= double(averageTemp.size());
            dTav /= double(averageTemp.size());

            dTav = sqrt((dTav - pow(Tav, 2)) / double(averageTemp.size()));

            // average pressure and variance
            TP = 0;
            double dTP = 0;
            for (int i = 0; i < averagePres.size(); i++){
                TP += averagePres[i];
                dTP += pow(averagePres[i], 2);
            }
            TP /= double(averagePres.size());
            dTP /= double(averagePres.size());

            dTP = sqrt((dTP - pow(TP, 2)) / double(averagePres.size()));

            cout << "Average temperature is " << Tav << " with variance " << dTav << endl;
            cout << "Initialisation temperature was " << P.T0 << endl;

            cout << "We get the pressure " << TP << " with variance " << dTP << endl;

            cout << 2*M_PI*P.rho/ (3.0 * temp()) * ( - 48.0/9.0 * pow(2.5, -9) + 8.0 * pow(2.5, -3)) << endl;

        }

        bool reverse(){
            for (int i = 0; i < P.N; i++){
                for (int k = 0; k < 3; k++){
                    past[i].position[k] = future[i].position[k];
                    past[i].velocity[k] = -future[i].velocity[k];
                }
            }
            return true;
        }


        /**
         * @brief function to calculate the average kinetic energy per particle
         * 
         * @param particles vector of all particles
         * @return double kinetic energy per particle
         */
        double energy_per_particle(){
            // total kinetic energy
            double T = 0;

            // calculating total energy
            for (int i = 0; i < present.size(); i++){
                for (int j = 0; j < 3; j++){
                    T += present[i].velocity[j] * present[i].velocity[j];
                }
                
            }

            return P.mass*T / double(2 * present.size());
        }

        double temp (){
            double T = 0;
            for (int i = 0; i < P.N; i++){
                for (int k = 0; k < 3; k++){
                    T += pow(future[i].velocity[k], 2);
                }
            }
            return P.mass*T/(3*double(P.N));
        }


        vecd print_past(){
            vecd output (past.size() * 3);
            for (int i = 0; i < present.size(); i++){
                for (int j = 0; j < 3; j++){
                    output[3*i + j] = past[i].position[j];
                }
            }
            return output;
        }

        vecd print_present(){
            vecd output (present.size() * 3);
            for (int i = 0; i < present.size(); i++){
                for (int j = 0; j < 3; j++){
                    output[3*i + j] = present[i].position[j];
                }
            }
            return output;
        }
        vecd print_future(){
            vecd output (future.size() * 3);
            for (int i = 0; i < present.size(); i++){
                for (int j = 0; j < 3; j++){
                    output[3*i + j] = future[i].position[j];
                }
            }
            return output;
        }

        /**
         * @brief Calculating the Mean Squared Displacement of the particles
         * 
         * @return double MSD averaged over particles
         */
        double MSD(){
            // assuming here that past stays as the initial distribution
            double msd = 0;
            for (int i = 0; i < P.N; i++){
                for (int k = 0; k < 3; k++){
                    // we use the image mirroring again
                    if (fabs(future[i].position[k] - past[i].position[k]) < P.L[0] / 2.0){
                        msd += pow(future[i].position[k] - past[i].position[k], 2);
                    }
                    else if ((past[i].position[k] - future[i].position[k]) > P.L[0] / 2.0){
                        msd += pow(future[i].position[k] + P.L[0] - past[i].position[k], 2);
                    }
                    else if ((past[i].position[k] - future[i].position[k]) < -P.L[0] / 2.0 ){
                        msd += pow(future[i].position[k] - P.L[0] - past[i].position[k], 2);
                    }
                }
            }

            return msd / double(P.N);
        }

};


template<typename Engine>
class ClusterWriter {
  const MD<Engine>& md;
public:
  ClusterWriter(const MD<Engine>& md) : md(md) { }

  friend std::ostream& operator << (std::ostream& out, const ClusterWriter& SW) {
    const auto& sites = SW.md.past;
    for(int i = 0; i < sites.size(); i++){
        for (int k = 0; k < 3; k++){
            out << sites[i].position[k] << " ";
        }
    } 
    return out;
  }
};




int main(int argc, char* argv[]){
    // Load up default parameters
    Parameters P;

    // Set up command-line overrides
    CLI::App app{"Molecular Dynamics of Argon (MD)"};

    app.add_option("-m,--mass",    P.mass,      "mass of particles");
    app.add_option("-N,--particles",  P.N,      "Number of particles");
    app.add_option("-L,--length",     P.L,  "Dimensions of box");
    app.add_option("-T,--temperature",     P.T0,  "initialisation temperature");

    // Output parameters
    std::string output = "";
    std::string method = ""; 
    double burnin = 0, until = 100, every = 0.032;
    double rho = 0.55;
    int n = 0;


    app.add_option("-o, --output", output, "Output type");
    app.add_option("-M, --method", method, "method used");
    app.add_option("-r, --density", rho, "density of system");
    app.add_option("-b,--burnin",        burnin,        "Time to run before starting measurements");
    app.add_option("-u,--until",         until,         "Time to run for once measurements started");
    app.add_option("-e,--every",         every,         "Measurement interval");
    app.add_option("-n, --book", n, "book keeping step size");

    CLI11_PARSE(app, argc, argv);

    std::mt19937 rng((std::random_device())());
    P.rho = rho;
    // assuming cube box we get
    P.L[0] = P.L[1] = P.L[2] = double(pow(double(P.N)/rho, 1.0/3.0));

    //P.N = 4;

    double M3 = P.N/4;
    double M = pow(M3, 1.0/3.0);
    P.a = double(P.L[0]) / double(M);

    if(output[0] == 't') output = "time";
    else if(output[0] == 'T') output = "temp";
    else if(output[0] == 'a') output = "adjust_temp";

    double min_pos = 0;
    double max_pos = 2.5;
    P.hist_depth = 0;
    while (min_pos <= max_pos){
        min_pos+= P.dx;
        P.hist_depth++;
    }


    
    cout << "############################################" << endl;
    cout << "N \t" << P.N << endl;
    cout << "V \t" << P.L[0] << "x" << P.L[1] << "x" << P.L[2] << endl;
    cout << "rho \t" << rho << endl;
    cout << "a \t" << P.a << endl;
    cout << "t_max \t" << until << " , burnin " << burnin << " , h " << every << endl;
    cout << "Output \t" << output << endl;
    cout << "Method \t" << method << endl;
    cout << "############################################" << endl;

    if (output == "time"){

        auto t1 = high_resolution_clock::now();
        MD md(P, rng);
        md.velocity_verlet(until, burnin, every, 0, 0);
        auto t2 = high_resolution_clock::now();

        duration<double, std::milli> ms_double = t2 - t1;

        auto t3 = high_resolution_clock::now();
        MD mdd(P, rng);
        mdd.velocity_verlet(until, burnin, every, n, 0);
        auto t4 = high_resolution_clock::now();

        duration<double, std::milli> ms_double_2 = t4 - t3;


        cout << "Without table: " << ms_double.count() << "ms" << " With table " << ms_double_2.count() << "ms" << endl;
    }
    else if (output == "temp"){
        MD mdd(P, rng);
        mdd.velocity_verlet(until, burnin, every, 0, 0);
    }
    else if (output == "adjust_temp"){
        // auto t1 = high_resolution_clock::now();
        MD md(P, rng);
        md.velocity_verlet(until, burnin, every, 0, 1);
        // auto t2 = high_resolution_clock::now();

        // duration<double, std::milli> ms_double = t2 - t1;

        //auto t3 = high_resolution_clock::now();
        MD mdd(P, rng);
        mdd.velocity_verlet(until, burnin, every, n, 1);
        //auto t4 = high_resolution_clock::now();

        //duration<double, std::milli> ms_double_2 = t4 - t3;


        //cout << "Without table: " << ms_double.count() << "ms" << " With table " << ms_double_2.count() << "ms" << endl;

        cout << "Lattice constant a is " << P.a << endl;
    }
    else if (output == "w"){
        MD md(P, rng);
        md.velocity_verlet(until, burnin, every, 0, 0);
    }
    // in case I need to debug some more....
    if (false){
        MD md(P, rng);
        vecd initial = md.print_past();
        cout << "before" << endl;
        md.velocity_verlet(until, burnin, every, n, 0);
        cout << "after" << endl;
        bool work = md.reverse();
        cout << "reverse" << endl;
        md.velocity_verlet(until, burnin, every, n, 0);
        cout << "after reverse" << endl;
        vecd final = md.print_present();

        ofstream outfile;
        outfile.open("position_comparison.txt");
        for (int i = 0; i < initial.size(); i++){
            outfile << initial[i] << " " << final[i] << endl;
        }

        if (work) cout << "nice" << endl;
    }

    return 0;
}