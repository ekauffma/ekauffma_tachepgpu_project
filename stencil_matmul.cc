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

const int DSIZE = 512;
const int RADIUS = 2;

int stencil_2d(int *in, int *out) {
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            int result = in[i][j];
            for (int k = 0; k < RADIUS; k++) {
                if (i+k < DSIZE) result += in[i+k][j];
                if (i-k > 0) result += in[i-k][j];
                if (j+k < DSIZE) result += in[i][j+k];
                if (j-k > 0) result += in[i][j-k];
            out[i][j] = result;
        }
    }
    return 0;
}

int main() {

    int h_A[DSIZE][DSIZE] = {};
    int h_B[DSIZE][DSIZE] = {};
    int h_A_stencilled[DSIZE][DSIZE] = {};
    int h_B_stencilled[DSIZE][DSIZE] = {};
    int h_C[DSIZE][DSIZE] = {};

    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            h_A[i][j] = rand()%11;
            h_B[i][j] = rand()%11;
            h_A_stencilled[i][j] = 0;
            h_B_stencilled[i][j] = 0;
            h_C[i][j] = 0;
        }
    }

    stencil_2d(h_A, h_A_stencilled);
    stencil_2d(h_B, h_B_stencilled);

    for (int i = 0; i < 50; i++) {
        std::cout << h_A[i][i] << ", ";
    }
}
