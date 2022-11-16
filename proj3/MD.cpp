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

    vecd force(){
        // vector with force for each particle and each component
        vecd f(P.N * 3);



        return f;
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
            
            // to achieve p_tot = 0 we need a vector for the mean
            vecd mean_momentum {0, 0, 0};
            // initialise the position and momentum of each particle drawn from random distributions
            for (unsigned i = 0; i < P.N; i++){
                for (unsigned k = 0; k < 3; k++){
                    // draw position and momentum from uniform continuous and normal distribution, respectively
                    past[i].momentum[k] = maxwell(rng);
                    past[i].position[k] = uniform_real_distribution<double> (0, P.L[k])(rng);
                    // add the momentum to the total, i.e. mean, momentum
                    mean_momentum[k] += past[i].momentum[k];
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

            }



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


    app.add_option("-o, --output", output, "Output type");
    app.add_option("-M, --method", method, "method used");

    app.add_option("-b,--burnin",        burnin,        "Time to run before starting measurements");
    app.add_option("-u,--until",         until,         "Time to run for once measurements started");
    app.add_option("-e,--every",         every,         "Measurement interval");

    CLI11_PARSE(app, argc, argv);

    std::mt19937 rng((std::random_device())());



    MD md(P, rng);



    return 0;
}