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
using vecd = std::vector<long double>;
using vec = std::vector<int>;

template<typename Engine> class ClusterWriter;


struct Parameters {
    double      mass = 1.0;     // assuming the particles have the same mass
    vec         L {10, 20, 30}; // dimensions of box
    double      T0 = 1;         // initial temperature
    int    N = 100;        // number of particles
};

template<typename Engine>
class MD {

    friend class ClusterWriter<Engine>;

    struct Particle{
        vecd position = vecd(3);    // position of partilce
        vecd velocity = vecd(3);    // velocity of particle
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

        vecd dist = minimal_distance();

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
                            new_r[k] = future[j].position[k] - P.L[0];
                        }
                        else if (future[i].position[k] - future[j].position[k] < double(P.L[0]) / 2){
                            new_r[k] = future[j].position[k] + P.L[0];
                        }
                        if (dist[P.N*i + j] > 1e-5 && dist[P.N*i + j] < 2.5){
                            f[3*i + k] += (1 / pow(dist[P.N*i + j], 14) - 0.5 / pow(dist[P.N*i + j], 8)) * (future[i].position[k] - new_r[k]);
                            f[3*j + k] -= (1 / pow(dist[P.N*i + j], 14) - 0.5 / pow(dist[P.N*i + j], 8)) * (future[i].position[k] - new_r[k]);
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
                    if (distance(future[i].position, new_r) < 1e-5){
                         cout << "WTF " << i << " " << j << " " << distance(future[i].position, new_r) << endl;
                         for (int m = 0; m < 3; m++){
                            cout << future[i].position[m] << " " << future[j].position[m]  << " "<< new_r[m] << endl;
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
        return 4 * (1/pow(r, 12) - 1/pow(r, 6));
    }

    vecd potential(vecd r){
        vecd output (P.N * P.N);

        // loop over output for particles
        for (int i = 0; i < P.N; i++){
            // loop over all particles
            for (int j = 0; j < P.N; j++){
                if (r[P.N * i + j] > 2.5){
                    output[P.N * i + j] = 0;
                    }
                else {
                    if (j > i){
                        output[P.N * i + j] = lenJones(r[P.N * i + j]);
                    }
                    else if (i == j){
                        output[P.N * i + j] = 0;
                    }
                    else {
                        output[P.N * i + j] = lenJones(r[P.N * j + i]);
                    }
                }
            }
        }
        return output;
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
            
            // just testing some stuff
            ofstream outfile;
            outfile.open("test.txt");
            for (int i = 0; i < P.N; i++){
                for (int k = 0; k < 3; k++){
                    outfile << past[i].position[2-k] << " ";
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
            
            for (int n = 0; n * h < t; n++){
                
                if (n == 0){
                    for (int i = 0; i < P.N; i++){
                        for (int k = 0; k < 3; k++){
                            present[i].position[k] = past[i].position[k] + h * past[i].velocity[k] / P.mass + h*h / 2.0;
                        }
                    }
                }

                vecd F = force();
                // update the "future"
                for (int i = 0; i < P.N; i++){
                    for (int k = 0; k < 3; k++){
                        // next position
                        future[i].position[k] = 2*present[i].position[k] - past[i].position[k] + h*h * F[3*i + k];
                        // we have to calculate the new velocity before applying boundary conditions -> otherwise
                        // the velocity will be extremely large...
                        future[i].velocity[k] = (future[i].position[k] - past[i].position[k]) / (2 * h);
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

        void velocity_verlet(double t, double tburn, double h){
            for (int n = 0; n * h < t; n++){
                // just adding initial conditions
                if (n == 0){
                    for (int i = 0; i < P.N; i++){
                        for (int k = 0; k < 3; k++){
                            present[i].velocity[k] = past[i].velocity[k];
                            present[i].position[k] = past[i].position[k];
                            future[i].position[k] = past[i].position[k];
                        }
                    }
                }
                
                //int c = 0;

                vecd F = force(); // this has 3N dimensions
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
                F = force();
                for (int i = 0; i < P.N; i++){
                    for (int k = 0; k < 3; k++){
                        future[i].velocity[k] = present[i].velocity[k] + h/(2*P.mass) * F[3*i + k];
                    }
                }
            }


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
    CLI::App app{"Condensed Run-and-Tumble model (crumble)"};

    app.add_option("-m,--mass",    P.mass,      "mass of particles");
    app.add_option("-N,--particles",  P.N,      "Number of particles");
    app.add_option("-L,--length",     P.L,  "Dimensions of box");
    app.add_option("-T,--temperature",     P.T0,  "initialisation temperature");

    // Output parameters
    std::string output = "";
    std::string method = ""; 
    double burnin = 0, until = 1, every = 0.001;
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

    vecd initial = md.print_past();
    cout << "before" << endl;
    md.velocity_verlet(until, burnin, every);
    cout << "after" << endl;
    bool work = md.reverse();
    cout << "reverse" << endl;
    md.velocity_verlet(until, burnin, every);
    cout << "after reverse" << endl;
    vecd final = md.print_present();

    ofstream outfile;
    outfile.open("position_comparison.txt");
    for (int i = 0; i < initial.size(); i++){
        outfile << initial[i] << " " << final[i] << endl;
    }

    if (work) cout << "nice" << endl;


    return 0;
}