/**********************************************************************/
/* stencil_matmul_d.cu                                                */
/* Author: Elliott Kauffman                                           */
/* performs 2d stencil on 2 matrices then multiplies them             */
/**********************************************************************/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <ostream>

using namespace std;

#define DSIZE 4096
#define RADIUS 3
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
    int *A, *B, *A_stencilled, *B_stencilled, *C;

    // nondefault CUDA streams
    cudaStream_t streamA, streamB;

    // These are used for timing
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;

    // start timing
    t0 = clock();

    // create nondefault CUDA streams
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);


    int size = DSIZE * DSIZE * sizeof(int);
    // allocate memory
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&A_stencilled, size);
    cudaMallocManaged(&B_stencilled, size);
    cudaMallocManaged(&C, size);

    // filling initial values of matrices
    srand(time(nullptr));
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            *(A+(i*DSIZE+j)) = rand()%2;
            *(B+(i*DSIZE+j)) = rand()%2;
            *(A_stencilled+(i*DSIZE+j)) = 0;
            *(B_stencilled+(i*DSIZE+j)) = 0;
            *(C+(i*DSIZE+j)) = 0;
        }
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // specify block and grid dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 gridSize(DSIZE/BLOCK_SIZE, DSIZE/BLOCK_SIZE);
    
    // launch kernels for stencilling
    stencil_2d<<<gridSize, blockSize, 0, streamA>>>(A, A_stencilled);
    cudaDeviceSynchronize();
    stencil_2d<<<gridSize, blockSize, 0, streamB>>>(B, B_stencilled);
    cudaDeviceSynchronize();

    //synchronize streams
    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    // launch matrix multiplication kernel
    mat_mul<<<gridSize, blockSize, 0, streamA>>>(A_stencilled, B_stencilled, C);
    cudaDeviceSynchronize();

    //synchronize streams
    cudaStreamSynchronize(streamA);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // perform error check for stencils
    stencil_errorcheck(A, A_stencilled);
    stencil_errorcheck(B, B_stencilled);

    // perform error check for matrix multiplication
    matmul_errorcheck(A_stencilled, B_stencilled, C);

    // print results
    std::cout<<"Printing 8x8 top left corner of each matrix:\n";
    std::cout<<"A = \n";
    printMatrix(A);
    std::cout<<"B= \n";
    printMatrix(B);
    std::cout<<"A_stencilled = \n";
    printMatrix(A_stencilled);
    std::cout<<"B_stencilled = \n";
    printMatrix(B_stencilled);
    std::cout<<"C = \n";
    printMatrix(C);

    // free the memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(A_stencilled);
    cudaFree(B_stencilled);
    cudaFree(C);

    // Destroy streams
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);

}
