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

int main() {

    int h_A[DSIZE * DSIZE] = {};
    int h_B[DSIZE * DSIZE] = {};
    int h_C[DSIZE * DSIZE] = {};

    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = rand()%11;
        h_B[i] = rand()%11;
        h_C[i] = 0;
    }

    for (int i = 0; i < 50; i++) {
        std::cout << h_A[i] << ", ";
    }
}
