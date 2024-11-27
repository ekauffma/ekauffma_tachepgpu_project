/**********************************************************************/
/* stencil_matmul.cc                                                  */
/* Author: Elliott Kauffman                                           */
/* performs 2d stencil on 2 matrices then multiplies them             */
/**********************************************************************/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <ostream>

const int DSIZE = 4;
const int RADIUS = 1;

int stencil_2d(int in[DSIZE][DSIZE], int out[DSIZE][DSIZE]) {
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            int result = in[i][j];
            out[i][j] = result;
            if (i < RADIUS || (DSIZE-i) <= RADIUS) {
                continue;
            }
            if (j < RADIUS || (DSIZE-j) <= RADIUS) {
                continue;
            }
            for (int k = 1; k <= RADIUS; k++) {
                if (i+k <= DSIZE) {
                    result += in[i+k][j];
                }
                if (i-k >= 0) {
                    result += in[i-k][j];
                }
                if (j+k <= DSIZE) {
                    result += in[i][j+k];
                }
                if (j-k >= 0) {
                    result += in[i][j-k];
                }
            }
            out[i][j] = result;
        }
    }
    return 0;
}

int stencil_errorcheck(int original[DSIZE][DSIZE], int modified[DSIZE][DSIZE]) {
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            if (i < RADIUS || (DSIZE-i) <= RADIUS) {
                if (modified[i][j] != original[i][j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, original[i][j], 1);
                    return -1;
                }
            }
            else if (j < RADIUS || (DSIZE-j) <= RADIUS) {
                if (modified[i][j] != original[i][j]) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, original[i][j], 1);
                    return -1;
                }
            }
            else {
                int expectedValue = original[i][j];
                for (int k = 1; k <= RADIUS; k++) {
                    if (i+k <= DSIZE) expectedValue += original[i+k][j];
                    if (i-k >= 0) expectedValue += original[i-k][j];
                    if (j+k <= DSIZE) expectedValue += original[i][j+k];
                    if (j-k >= 0) expectedValue += original[i][j-k];
                }
                if (modified[i][j] != expectedValue) {
                    printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, expectedValue, 1);
                    return -1;
                }
            }
        }
    }
    return 0;
}

int mat_mul(int A[DSIZE][DSIZE], int B[DSIZE][DSIZE], int C[DSIZE][DSIZE]) {
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            int result = 0;
            for (int k = 0; k < DSIZE; k++) {
                result += A[i][k] * B[k][j];
            }
            C[i][j] = result;
        }
    }
    return 0;
}

int printMatrix(int A[DSIZE][DSIZE], int limit = 8) {
    std::cout<<"-              -\n";
    for (int i = 0; i < limit; i++) {
        std::cout<<"| ";
        for (int j = 0; j < limit; j++) {
            std::cout<<A[i][j]<<" ";
        }
        std::cout<<" |\n";
    }
    std::cout<<"-              -\n\n";
    return 0;
}


int main() {

    std::cout<<"Start of program\n";
    int (*h_A)[DSIZE] = new int[DSIZE][DSIZE];
    int (*h_B)[DSIZE] = new int[DSIZE][DSIZE];
    int (*h_A_stencilled)[DSIZE] = new int[DSIZE][DSIZE];
    int (*h_B_stencilled)[DSIZE] = new int[DSIZE][DSIZE];
    int (*h_C)[DSIZE] = new int[DSIZE][DSIZE];    
    std::cout<<"Initialized matrices\n";

    srand(time(nullptr));
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            h_A[i][j] = rand()%2;
            h_B[i][j] = rand()%2;
            h_A_stencilled[i][j] = 0;
            h_B_stencilled[i][j] = 0;
            h_C[i][j] = 0;
        }
    }
    std::cout<<"Initialized matrix values\n";

    stencil_2d(h_A, h_A_stencilled);
    std::cout<<"Performed stencil on matrix A\n";
    stencil_2d(h_B, h_B_stencilled);
    std::cout<<"Performed stencil on matrix B\n";
    stencil_errorcheck(h_A, h_A_stencilled);
    std::cout<<"Performed error check on matrix A\n";
    stencil_errorcheck(h_B, h_B_stencilled);
    std::cout<<"Performed error check on matrix B\n";
    mat_mul(h_A_stencilled, h_B_stencilled, h_C);
    std::cout<<"Finished multiplying matries and storing results in C\n";
    
    std::cout<<"Printing 8x8 top left corner of each matrix:\n";
    std::cout<<"h_A = \n";
    printMatrix(h_A, 4);
    std::cout<<"h_B= \n";
    printMatrix(h_B, 4);
    std::cout<<"h_A_stencilled = \n";
    printMatrix(h_A_stencilled, 4);
    std::cout<<"h_B_stencilled = \n";
    printMatrix(h_B_stencilled, 4);
    std::cout<<"h_C = \n";
    printMatrix(h_C, 4);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_stencilled;
    delete[] h_B_stencilled;
    delete[] h_C;
}
