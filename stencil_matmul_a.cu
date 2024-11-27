/**********************************************************************/
/* stencil_matmul.cu                                                  */
/* Author: Elliott Kauffman                                           */
/* performs 2d stencil on 2 matrices then multiplies them             */
/**********************************************************************/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <ostream>

using namespace std;

#define DSIZE 64
#define RADIUS 1
#define BLOCK_SIZE 32

__global__ void mat_mul (int* A, int* B, int* C){

   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;

   if((row < DSIZE) && (column < DSIZE)){
       int sum = 0;
       for(int k = 0; k < DSIZE; k++) {
           sum += A[row*DSIZE + k] * B[k*DSIZE + column];
       }
       C[row * DSIZE + column] = sum;
   }
}


__global__ void stencil_2d(int *in, int *out) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int column = threadIdx.x + blockIdx.x * blockDim.x;

    int result = in[row * DSIZE + column];
    if ((row < RADIUS) || ((DSIZE - row) <= RADIUS)) {
        out[row * DSIZE + column] = result;
        return;
    }
    else if ((column < RADIUS) || ((DSIZE - column) <= RADIUS)) {
        out[row * DSIZE + column] = result;
        return;
    }
    else {
        for (int k = -RADIUS; k <= RADIUS; k++) {
            if (k==0) continue;
            result += in[(row + k)*DSIZE + column];
            result += in[row*DSIZE + (column + k)];
        }
    }
	out[row * DSIZE + column] = result;
    return;
}

int stencil_errorcheck(int *original, int *modified) {
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            if (i < RADIUS || (DSIZE-i) <= RADIUS) {
                if (modified[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, original[i*DSIZE+j], 1);
                    return -1;
                }
            }
            else if (j < RADIUS || (DSIZE-j) <= RADIUS) {
                if (modified[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, original[i*DSIZE+j], 1);
                    return -1;
                }
            }
            else {
                int expectedValue = original[i*DSIZE+j];
                for (int k = 1; k <= RADIUS; k++) {
                    if (i+k <= DSIZE) expectedValue += original[(i+k)*DSIZE+j];
                    if (i-k >= 0) expectedValue += original[(i-k)*DSIZE+j];
                    if (j+k <= DSIZE) expectedValue += original[i*DSIZE+j+k];
                    if (j-k >= 0) expectedValue += original[i*DSIZE+j-k];
                }
                if (modified[i*DSIZE+j] != expectedValue) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, expectedValue, 1);
                    return -1;
                }
            }
        }
    }
    return 0;
}

int matmul_errorcheck(int *A, int *B, int *C) {

    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            int result = 0;
            for (int k = 0; k < DSIZE; k++) {
                result += A[i*DSIZE+k] * B[k*DSIZE+j];
            }
            if (C[i*DSIZE + j]!=result) {
                printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, result, 1);
                return -1;
            }
        }
    }
    return 0;
}

int printMatrix(int *A, int limit = 8) {
    std::cout<<"-              -\n";
    for (int i = 0; i < limit; i++) {
        std::cout<<"| ";
        for (int j = 0; j < limit; j++) {
            std::cout<<A[i*DSIZE+j]<<" ";
        }
        std::cout<<" |\n";
    }
    std::cout<<"-              -\n\n";
    return 0;
}

int main() {

    // initialize matrices
    int *h_A, *h_B, *h_A_stencilled, *h_B_stencilled, *h_C; // host copies of matrices
    int *d_A, *d_B, *d_A_stencilled, *d_B_stencilled, *d_C; // device copies of matrices

    int size = DSIZE * DSIZE * sizeof(int);
    // allocate space for host copies
    h_A = (int *)malloc(size);
    h_B = (int *)malloc(size);
    h_A_stencilled = (int *)malloc(size);
    h_B_stencilled = (int *)malloc(size);
    h_C = (int *)malloc(size);

    // filling initial values of matrices
    srand(time(nullptr));
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            *(h_A+(i*DSIZE+j)) = rand()%2;
            *(h_B+(i*DSIZE+j)) = rand()%2;
            *(h_A_stencilled+(i*DSIZE+j)) = 0;
            *(h_B_stencilled+(i*DSIZE+j)) = 0;
            *(h_C+(i*DSIZE+j)) = 0;
        }
    }

    // allocate space for device copies
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_A_stencilled, size);
    cudaMalloc((void **)&d_B_stencilled, size);
    cudaMalloc((void **)&d_C, size);

    // copy memory from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_stencilled, h_A_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_stencilled, h_B_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // specify block and grid dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 gridSize(DSIZE/BLOCK_SIZE, DSIZE/BLOCK_SIZE);
    
    // launch kernels for stencilling
    stencil_2d<<<gridSize, blockSize>>>(d_A, d_A_stencilled);
    stencil_2d<<<gridSize, blockSize>>>(d_B, d_B_stencilled);

    // copy stencil results back to host
    cudaMemcpy(h_A_stencilled, d_A_stencilled, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_stencilled, d_B_stencilled, size, cudaMemcpyDeviceToHost);

    // perform error check for stencils
    stencil_errorcheck(h_A, h_A_stencilled);
    stencil_errorcheck(h_B, h_B_stencilled);
    
    // copy memory from host to device (probably redundant)
    cudaMemcpy(d_A_stencilled, h_A_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_stencilled, h_B_stencilled, size, cudaMemcpyHostToDevice);
    
    // launch matrix multiplication kernel
    mat_mul<<<gridSize, blockSize>>>(d_A_stencilled, d_B_stencilled, d_C);

    // copy multiplication results back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // perform error check for matrix multiplication
    matmul_errorcheck(h_A_stencilled, h_B_stencilled, h_C);

    // print results
    std::cout<<"Printing 8x8 top left corner of each matrix:\n";
    std::cout<<"h_A = \n";
    printMatrix(h_A);
    std::cout<<"h_B= \n";
    printMatrix(h_B);
    std::cout<<"h_A_stencilled = \n";
    printMatrix(h_A_stencilled);
    std::cout<<"h_B_stencilled = \n";
    printMatrix(h_B_stencilled);
    std::cout<<"h_C = \n";
    printMatrix(h_C);

    // free the memory
    free(h_A);
    free(h_B);
    free(h_A_stencilled);
    free(h_B_stencilled);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_A_stencilled);
    cudaFree(d_B_stencilled);
    cudaFree(d_C);
}
