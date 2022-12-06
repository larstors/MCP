#include<stdlib.h>
#include<tgmath.h>
#include<math.h>
#include<stdio.h>
#include<time.h>


// ####################### GAUSS-SEIDEL ##########################
/**
 * @brief Function to find x in a linear equation Ax = b using the Gauss-Seidel algorithm
 * 
 * @param n dimension of system
 * @param A Matrix
 * @param b vector
 * @param x vector to find
 * @param k_max max number of iterations
 */
void lgs_gs_solve(int n, double A[], double b[], double x[], int k_max){
	// iterating until max number of iterations
	for (int k = 0; k < k_max; k++){
		// actual calculation
		for (int i = 0; i < n; i++){
			x[i] = b[i];

			for (int j = 0; j < i; j++){
				x[i] -= A[i*n + j] * x[j];
			}

			for (int j = i+1; j < n; j++){
				x[i] -= A[i*n + j] * x[j];
			}

			x[i] *= 1.0 / A[i*n + i];
		}

	}

	return;
}

/**
 * @brief Function to find x in a linear equation Ax = b using the SOR algorithm
 * 
 * @param n dimension of system
 * @param A Matrix
 * @param b vector
 * @param x vector to find
 * @param k_max max number of iterations
 * @param w weight of iteration
 */
void lgs_SOR_solve(int n, double A[], double b[], double x[], int k_max, double w){
	// iterating until max number of iterations
	for (int k = 0; k < k_max; k++){
		// actual calculation
		for (int i = 0; i < n; i++){
			x[i] = b[i];

			for (int j = 0; j < i; j++){
				x[i] -= A[i*n + j] * x[j];
			}

			for (int j = i+1; j < n; j++){
				x[i] -= A[i*n + j] * x[j];
			}

			x[i] *= 1.0 / A[i*n + i];
		}

	}

	return;
}


// ####################### INVERSE OF MATRIXJACOBI ALGORITHM ##########################

/**
 * @brief Function to find x in a linear equation Ax = b using the jacobi algorithm
 * 
 * @param n dimension of system
 * @param A Matrix
 * @param b vector
 * @param x vector to find
 * @param k_max max number of iterations
 */
void lgs_jacobi_solve(int n, double A[], double b[], double x[], int k_max){
	// array for storing values
	double *x_int = (double*)malloc(sizeof(double)*n);
	//filling it
	for (int i = 0; i < n; i++){
		x_int[i] = x[i];
	}

	// iterating until max number of iterations
	for (int k = 0; k < k_max; k++){
		// doing the calculations
		for (int i = 0; i < n; i++){
			x[i] = b[i];
			for (int j = 0; j < n; j++){
				if (i != j){
					x[i] -= A[i*n + j] * x_int[j];
				}
			}
			x[i] *= 1.0 / A[i*n + i];
		}

		// setting values of to x_int for next iteration
		for (int i = 0; i < n; i++){
			x_int[i] = x[i];
		}
	}

	free(x_int);
	return;
}