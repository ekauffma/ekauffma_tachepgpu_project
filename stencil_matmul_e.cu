/**********************************************************************/
/* stencil_matmul_e.cu                                                */
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

__global__ void mat_mul(int *A, int *B, int *C) {
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
    int lindex_x = threadIdx.x;
    int lindex_y = threadIdx.y;

    // Shared memory
    __shared__ int tempA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tempB[BLOCK_SIZE][BLOCK_SIZE];

    long long result = 0;

    for (int i = 0; i < (((DSIZE - 1) / BLOCK_SIZE) + 1); i++) {
        if ((gindex_y < DSIZE) && (lindex_x + i * BLOCK_SIZE) < DSIZE) {
            tempA[lindex_y][lindex_x] = A[(gindex_y * DSIZE) + lindex_x + (i * BLOCK_SIZE)];
        } else {
            tempA[lindex_y][lindex_x] = 0;
        }
        if ((gindex_x < DSIZE) && (lindex_y + i * BLOCK_SIZE) < DSIZE) {
            tempB[lindex_y][lindex_x] = B[(lindex_y + i * BLOCK_SIZE) * DSIZE + gindex_x];
        } else {
            tempB[lindex_y][lindex_x] = 0;
        }
        __syncthreads();

        // Compute partial matrix product
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            result += tempA[lindex_y][j] * tempB[j][lindex_x];
        }
        __syncthreads(); // Synchronize again
    }

    if (gindex_y < DSIZE && gindex_x < DSIZE) {
        C[gindex_y * DSIZE + gindex_x] = (int)result;
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
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i, j, modified[i*DSIZE+j], original[i*DSIZE+j]);
                    return -1;
                }
            }
            else if (j < RADIUS || (DSIZE-j) <= RADIUS) {
                if (modified[i*DSIZE+j] != original[i*DSIZE+j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i, j, modified[i*DSIZE+j], original[i*DSIZE+j]);
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
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i, j, modified[i*DSIZE+j], expectedValue);
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
                printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[i*DSIZE + j], result);
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
    //stencil_errorcheck(A, A_stencilled);
   // stencil_errorcheck(B, B_stencilled);

    // perform error check for matrix multiplication
    //matmul_errorcheck(A_stencilled, B_stencilled, C);

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
