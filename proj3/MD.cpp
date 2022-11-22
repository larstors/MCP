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

#include "CLI11.hpp"



using namespace std;
using vecd = std::vector<double>;
using vec = std::vector<int>;



struct Parameters {
    double      mass = 1.0;     // assuming the particles have the same mass
    vec         L {10, 20, 30}; // dimensions of box
    double      T0 = 1;         // initial temperature
    unsigned    N = 100;        // number of particles
};

template<typename Engine>
class MD {

    struct Particle{
        vecd position = vecd(3);    // position of partilce
        vecd momentum = vecd(3);    // momentum of particle

    };


    const long double kb = 1;//1.38e-23;

    Parameters P; // A local copy of the model parameters
    std::vector<Particle> past, present, future; // particles at t-dt, t, t+dt
    Engine& rng; // Source of noise: this is a reference as there should only be one of these!
    std::normal_distribution<double> maxwell;

    /**
     * @brief Force on each particle in each coordinate using Lenard-Jones potential (could probs move this into 
     *        public as it doesnt really need to stay private, no?)
     * 
     * @return vecd 
     */
    vecd force(){
        // vector with force for each particle and each component
        vecd f(P.N * 3);



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
    double distance(vecd p1, vecd p2, vec n){
        // distance variable
        double dist = 0;

        for (int i = 0; i < 3; i++){
            dist = pow(p1[i] - p2[i] - n[i] * P.L[i], 2);
        }

        return dist;
    }

    vecd minimal_distance(){
        vecd output;

        // loop over output for particles
        for (int i = 0; i < P.N; i++){
            // loop over all particles
            for (int j = 0; j < P.N; j++){
                // only want difference between different particles
                if (i != j){
                    
                }
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
        maxwell(0, sqrt(kb * P.T0 / P.mass))
        {   
            assert(P.N/4==0);

            int M3 = P.N/4;
            double M = pow(M3, 1.0/3.0);
            // to achieve p_tot = 0 we need a vector for the mean
            vecd mean_momentum {0, 0, 0};
            // initialise the position and momentum of each particle drawn from random distributions
            for (unsigned i = 0; i < P.N; i++){
                for (unsigned k = 0; k < 3; k++){
                    // draw momentum from normal distribution, respectively
                    past[i].momentum[k] = maxwell(rng);
                    // add the momentum to the total, i.e. mean, momentum
                    mean_momentum[k] += past[i].momentum[k];
                }
            }

            // placing particles on fcc lattice structure

            double a = double(P.L[0]) / double(M);

            cout << a << " can fit " << double(P.L[0]) / a << endl;

            // for z coordinate
            for (int j = 0; j < 12; j++){
                // for y
                for (int k = 0; k < 12; k++){
                    // for x
                    for (int l = 0; l < 6; l++){
                        
                        
                        past[72*j + 6*k + l].position[2] = 1e-10 + j/2 * a + j%2*a/2;// z coordinate
                        if (j%2 == 0){
                            past[72*j + 6*k + l].position[0] = 1e-10 + ((0+2)/2)%2*(l * a + k%2 * a/2) + 0%2*(a/2 - k%2*a);// x coordinate
                            past[72*j + 6*k + l].position[1] = 1e-10 + ((0+2)/2)%2*(k/2 * a + k%2*a/2);// y coordinate
                        }
                        else {
                            past[72*j + 6*k + l].position[0] = 1e-10 + ((1+2)/2)%2*(l * a + k%2 * a/2) + 1%2*(a/2 - k%2*a);// x coordinate
                            past[72*j + 6*k + l].position[1] = 1e-10 + ((1+2)/2)%2*(k/2 * a + k%2*a/2);// y coordinate
                        }
                        if (l == 1 && k == 0 && j == 1){
                            cout << past[72*j + 6*k + l].position[0] << endl;
                        }
                    }
                }
            }

            // normalise the mean momentum so that the sum goes to zero
            for (int i = 0; i < 3; i++){
                mean_momentum[i] /= double(P.N);
            }

            // remove the mean from each particle so that the total momentum is 0
            for (unsigned i = 0; i < P.N; i++){
                for (unsigned k = 0; k < 3; k++){
                    past[i].momentum[k] -= mean_momentum[k];
                }
            }
            
            // just testing some stuff
            ofstream outfile;
            outfile.open("test.txt");
            for (unsigned i = 0; i < P.N; i++){
                for (unsigned k = 0; k < 3; k++){
                    outfile << past[i].position[k] << " ";
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
        vecd verlet(double t, double tburn, double h){
            
            for (unsigned n = 0; n * h < t; n++){
                
                if (n == 0){
                    for (unsigned i = 0; i < P.N; i++){
                        for (unsigned k = 0; k < 3; k++){
                            present[i].position[k] = past[i].position[k] + h * past[i].momentum[k] / P.mass + h*h / 2.0;
                        }
                    }
                }

                vecd F = force();
                // update the "future"
                for (unsigned i = 0; i < P.N; i++){
                    for (unsigned k = 0; k < 3; k++){
                        // next position
                        future[i].position[k] = 2*present[i].position[k] - past[i].position[k] + h*h * F[3*i + k];
                        // we have to calculate the new velocity before applying boundary conditions -> otherwise
                        // the velocity will be extremely large...
                        future[i].momentum[k] = P.mass*(future[i].position[k] - past[i].position[k]) / (2 * h);
                        // we need to account for periodic boundary conditions
                        if (future[i].position[k] < 0) future[i].position[k] += P.L[k];
                        else if (future[i].position[k] > P.L[k]) future[i].position[k] -= P.L[k]; 

                        // update past/present/future
                        past[i].position[k] = present[i].position[k];
                        present[i].position[k] = future[i].position[k];

                    }
                }
                
                // calculating the wanted stuff after equilibrating the system
                if (n * h > tburn){

                }
            }

            return present[0].position;
        }

        vecd velocity_verlet(double t, double tburn, double h){

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
                    T += present[i].momentum[j] * present[i].momentum[j];
                }
                
            }

            return T / double(2 * P.mass * present.size());
        }




};






int main(int argc, char* argv[]){
    // Load up default parameters
    Parameters P;

    // Set up command-line overrides
    CLI::App app{"Condensed Run-and-Tumble model (crumble)"};

    app.add_option("-m,--mass",    P.mass,      "mass of particles");
    app.add_option("-N,--particles",  P.N,      "Number of particles");
    app.add_option("-L,--length",     P.L,  "Dimensions of box");
    app.add_option("-T,--temperature",     P.T0,  "initialisation temperature");

    // Output parameters
    std::string output = "";
    std::string method = ""; 
    double burnin = 1000, until = 5000, every = 2.5;
    double rho = 0.55;


    app.add_option("-o, --output", output, "Output type");
    app.add_option("-M, --method", method, "method used");
    app.add_option("-r, --density", rho, "density of system");
    app.add_option("-b,--burnin",        burnin,        "Time to run before starting measurements");
    app.add_option("-u,--until",         until,         "Time to run for once measurements started");
    app.add_option("-e,--every",         every,         "Measurement interval");

    CLI11_PARSE(app, argc, argv);

    std::mt19937 rng((std::random_device())());

    // assuming cube box we get
    P.L[0] = P.L[1] = P.L[2] = int(pow(double(P.N)/rho, 1.0/3.0));

    MD md(P, rng);



    return 0;
}