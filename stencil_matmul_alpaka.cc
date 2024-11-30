/**********************************************************************/
/* stencil_matmul_alpaka.cc                                           */
/* Author: Elliott Kauffman                                           */
/* performs 2d stencil on 2 matrices then multiplies them             */
/**********************************************************************/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <ostream>

#include <alpaka/alpaka.hpp>

#define DSIZE 1024
#define RADIUS 3

using Device = alpaka::DevCpu;
using Platform = alpaka::Platform<Device>;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;

struct VectorAddKernel3D {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ in1,
                                T const* __restrict__ in2,
                                T* __restrict__ out,
                                Vec3D size) const {
    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
      auto index = (ndindex[0] * size[1] + ndindex[1]) * size[2] + ndindex[2];
      out[index] = in1[index] + in2[index];
    }
  }
};

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

    alpaka::PlatformCpu host_platform;
    alpaka::DevCpu host = alpaka::getDevByIdx(host_platform, 0u);
    std::cout << "Host platform: " << alpaka::core::demangled<alpaka::PlatformCpu> << '\n';
    std::cout << "Found 1 device:\n";
    std::cout << "  - " << alpaka::getName(host) << "\n\n";

    Platform platform;
    std::vector<Device> devices = alpaka::getDevs(platform);

    std::cout << "Accelerator platform: " << alpaka::core::demangled<Platform> << '\n';
    std::cout << "Found " << devices.size() << " device(s):\n";
    for (auto const& device : devices)
        std::cout << "  - " << alpaka::getName(device) << '\n';
    std::cout << '\n';

    Device device = alpaka::getDevByIdx(platform, 0u);
    std::cout << "Device: " << alpaka::getName(device) << '\n';

    // initialize matrices
    auto h_A = alpaka::allocMappedBuf<int, uint32_t>(host, platform, uint32_t(DSIZE*DSIZE));
    auto h_B = alpaka::allocMappedBuf<int, uint32_t>(host, platform, uint32_t(DSIZE*DSIZE));
    auto h_C = alpaka::allocMappedBuf<int, uint32_t>(host, platform, uint32_t(DSIZE*DSIZE));

    // filling initial values of matrices
    srand(time(nullptr));
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            h_A[i*DSIZE+j] = rand()%2;
            h_B[i*DSIZE+j] = rand()%2;
            h_C[i*DSIZE+j] = 0;
        }
    }

    // run the test the given device
    auto queue = Queue{device};

    // allocate input and output buffers on the device
    auto d_A = alpaka::allocAsyncBuf<int, uint32_t>(queue, uint32_t(DSIZE*DSIZE));
    auto d_B = alpaka::allocAsyncBuf<int, uint32_t>(queue, uint32_t(DSIZE*DSIZE));
    auto d_C = alpaka::allocAsyncBuf<int, uint32_t>(queue, uint32_t(DSIZE*DSIZE));

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::memcpy(queue, d_A, h_A);
    alpaka::memcpy(queue, d_B, h_B);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::memset(queue, d_C, 0x00);

}
