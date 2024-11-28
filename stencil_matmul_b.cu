/**********************************************************************/
/* stencil_matmul_b.cu                                                */
/* Author: Elliott Kauffman                                           */
/* performs 2d stencil on 2 matrices then multiplies them             */
/**********************************************************************/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <ostream>

using namespace std;

#define DSIZE 1024
#define RADIUS 3
#define BLOCK_SIZE 32

// error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

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

    // calculate indices
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
    int lindex_x = threadIdx.x + RADIUS;
    int lindex_y = threadIdx.y + RADIUS;


    // shared memory
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // Initialize central block elements
    if (gindex_x < DSIZE && gindex_y < DSIZE) {
        temp[lindex_x][lindex_y] = in[gindex_y + DSIZE * gindex_x];
    } else {
        temp[lindex_x][lindex_y] = 0; // Zero-pad for out-of-bound elements
    }

    // Fill halo regions
    if (threadIdx.x < RADIUS) {
        // Left halo
        int halo_gx = gindex_x - RADIUS;
        if (halo_gx >= 0) {
            temp[lindex_x - RADIUS][lindex_y] = in[gindex_y + DSIZE * halo_gx];
        } else {
            temp[lindex_x - RADIUS][lindex_y] = 0; // Zero-pad
        }

        // Right halo
        halo_gx = gindex_x + BLOCK_SIZE;
        if (halo_gx < DSIZE) {
            temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_y + DSIZE * halo_gx];
        } else {
            temp[lindex_x + BLOCK_SIZE][lindex_y] = 0; // Zero-pad
        }
    }

    if (threadIdx.y < RADIUS) {
        // Top halo
        int halo_gy = gindex_y - RADIUS;
        if (halo_gy >= 0) {
            temp[lindex_x][lindex_y - RADIUS] = in[halo_gy + DSIZE * gindex_x];
        } else {
            temp[lindex_x][lindex_y - RADIUS] = 0; // Zero-pad
        }

        // Bottom halo
        halo_gy = gindex_y + BLOCK_SIZE;
        if (halo_gy < DSIZE) {
            temp[lindex_x][lindex_y + BLOCK_SIZE] = in[halo_gy + DSIZE * gindex_x];
        } else {
            temp[lindex_x][lindex_y + BLOCK_SIZE] = 0; // Zero-pad
        }
    }

    // sync threads
    __syncthreads();

    // fill values for halo elements
    if ((gindex_x < RADIUS) || ((DSIZE - gindex_x) <= RADIUS) || (gindex_y < RADIUS) || ((DSIZE - gindex_y) <= RADIUS)) {
        out[gindex_x * DSIZE + gindex_y] = in[gindex_y + DSIZE * gindex_x];
        return;
    }

    // Perform stencil computation for valid global indices
    if (gindex_x >= RADIUS && gindex_x < DSIZE - RADIUS &&
        gindex_y >= RADIUS && gindex_y < DSIZE - RADIUS) {
        int result = 0;
        for (int k = -RADIUS; k <= RADIUS; k++) {
            result += temp[lindex_x + k][lindex_y];    // Row stencil
            if (k!=0) // avoid double counting
                result += temp[lindex_x][lindex_y + k];    // Column stencil
        }
        out[gindex_y + DSIZE * gindex_x] = result;
    }
    return;
}

int stencil_errorcheck(int *original, int *modified) {
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            if (i < RADIUS || (DSIZE-i) <= RADIUS) {
                if (modified[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, modified[i*DSIZE+j], original[i*DSIZE+j]);
                    return -1;
                }
            }
            else if (j < RADIUS || (DSIZE-j) <= RADIUS) {
                if (modified[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, modified[i*DSIZE+j], original[i*DSIZE+j]);
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
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, modified[i*DSIZE+j], expectedValue);
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
                printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[i*DSIZE+j], result);
                return -1;
            }
        }
    }
    return 0;
}

int printMatrix(int *A, int limit = 4) {
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

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);

    // initialize matrices
    int *h_A, *h_B, *h_A_stencilled, *h_B_stencilled, *h_C; // host copies of matrices
    int *d_A, *d_B, *d_A_stencilled, *d_B_stencilled, *d_C; // device copies of matrices

    // These are used for timing
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;

    // start timing
    t0 = clock();

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

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // allocate space for device copies
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_A_stencilled, size);
    cudaMalloc((void **)&d_B_stencilled, size);
    cudaMalloc((void **)&d_C, size);
    cudaCheckErrors("");

    // copy memory from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_stencilled, h_A_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_stencilled, h_B_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("");

    // specify block and grid dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 gridSize(DSIZE/BLOCK_SIZE, DSIZE/BLOCK_SIZE);
   
    // launch kernels for stencilling
    printf("Launching kernel for stencil #1\n");
    stencil_2d<<<gridSize, blockSize>>>(d_A, d_A_stencilled);
    cudaCheckErrors("");
    printf("Launching kernel for stencil #2\n");
    stencil_2d<<<gridSize, blockSize>>>(d_B, d_B_stencilled);
    cudaCheckErrors("");

    // copy stencil results back to host
    cudaMemcpy(h_A_stencilled, d_A_stencilled, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_stencilled, d_B_stencilled, size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("");

    // copy memory from host to device (probably redundant)
    cudaMemcpy(d_A_stencilled, h_A_stencilled, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_stencilled, h_B_stencilled, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("");
    
    // launch matrix multiplication kernel
    printf("Launching kernel for matrix multiplication\n");
    mat_mul<<<gridSize, blockSize>>>(d_A_stencilled, d_B_stencilled, d_C);
    cudaCheckErrors("");

    // copy multiplication results back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("");

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // perform error check for stencils
    stencil_errorcheck(h_A, h_A_stencilled);
    stencil_errorcheck(h_B, h_B_stencilled);

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
