#include <cassert>
#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>

#include "config.h"
#include "WorkDiv.hpp"

#define DSIZE 1024
#define RADIUS 3

struct Stencil2d {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator() (TAcc const& acc,
                                 T const* __restrict__ A,
                                 T* __restrict__ A_stencilled,
                                 Vec2D size) const {

    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
      auto i = ndindex[0];
      auto j = ndindex[1];

      // Initialize result with the current element
      int result = A[i * size[1] + j];

      // Apply stencil only if within bounds
      if (i >= RADIUS && i < size[0] - RADIUS && j >= RADIUS && j < size[1] - RADIUS) {
        for (int k = 1; k <= RADIUS; ++k) {
          result += A[(i + k) * size[1] + j]; // Below
          result += A[(i - k) * size[1] + j]; // Above
          result += A[i * size[1] + (j + k)]; // Right
          result += A[i * size[1] + (j - k)]; // Left
        }
   :   }

      // Store the result
      A_stencilled[i * size[1] + j] = result;
    }
  }
};

struct MatMul {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator() (TAcc const& acc,
                                 T const* __restrict__ A,
                                 T const* __restrict__ B,
                                 T* __restrict__ C,
                                 Vec2D size) const {

    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {

      auto i = ndindex[0];
      auto j = ndindex[1];

      int result = 0;
      for (uint32_t k = 0; k < size[0]; k++) {
        result += A[i*size[1]+k] * B[k*size[1]+j];
      }
      C[i*size[1]+j] = result;
    }
  }
};

void testStencilMatMul(Host host, Platform platform, Device device) {

  // These are used for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // start timing
  t0 = clock();

  // 3-dimensional and linearised buffer size
  constexpr Vec2D ndsize = {DSIZE, DSIZE};
  constexpr uint32_t size = ndsize.prod();

  // initialize matrices
  auto h_A = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
  auto h_B = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
  auto h_A_stencilled = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
  auto h_B_stencilled = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
  auto h_C = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);

  // fill the input buffers with random data, and the output buffer with zeros
  srand(time(nullptr));
  for (uint32_t i = 0; i < size; ++i) {
    h_A[i] = rand()%2;
    h_B[i] = rand()%2;
    h_A_stencilled[i] = 0;
    h_B_stencilled[i] = 0;
    h_C[i] = 0.;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // run the test the given device
  auto queue = Queue{device};

  // allocate input and output buffers on the device
  auto d_A = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
  auto d_B = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
  auto d_A_stencilled = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
  auto d_B_stencilled = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
  auto d_C = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);

  // copy the input data to the device; the size is known from the buffer objects
  alpaka::memcpy(queue, d_A, h_A);
  alpaka::memcpy(queue, d_B, h_B);

  // fill the output buffers with zeros; the size is known from the buffer objects
  alpaka::memset(queue, d_A_stencilled, 0x00);
  alpaka::memset(queue, d_B_stencilled, 0x00);
  alpaka::memset(queue, d_C, 0x00);

  auto div = makeWorkDiv<Acc2D>({32, 1},{32, 1});
  std::cout << "Running Stencil2d for Matrix A with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc2D>(
      queue, div, Stencil2d{}, d_A.data(), d_A_stencilled.data(), ndsize);
  
  // copy the results from the device to the host
  alpaka::memcpy(queue, h_A_stencilled, d_A_stencilled);

  // wait for all the operations to complete
  alpaka::wait(queue);

  //check stencil
  printf("Checking stencil operation:\n");
  for (int i = 0; i < DSIZE; ++i) {
    for (int j = 0; j < DSIZE; ++j) {
       if (i < RADIUS || (DSIZE-i) <= RADIUS) {
          if (h_A[i*DSIZE+j] != h_A_stencilled[i*DSIZE+j]) {
            printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_A_stencilled[i*DSIZE+j], h_A[i*DSIZE+j]);
          }
        }
        else if (j < RADIUS || (DSIZE-j) <= RADIUS) {
          if (h_A[i*DSIZE+j] != h_A_stencilled[i*DSIZE+j]) {
            printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_A_stencilled[i*DSIZE+j], h_A[i*DSIZE+j]);
          }
        }
        else {
          int expectedValue = h_A[i*DSIZE+j];
          for (int k = 1; k <= RADIUS; k++) {
            if (i+k <= DSIZE) expectedValue += h_A[(i+k)*DSIZE+j];
            if (i-k >= 0) expectedValue += h_A[(i-k)*DSIZE+j];
            if (j+k <= DSIZE) expectedValue += h_A[i*DSIZE+j+k];
            if (j-k >= 0) expectedValue += h_A[i*DSIZE+j-k];
          }
        if (h_A_stencilled[i*DSIZE+j] != expectedValue) {
          printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_A_stencilled[i*DSIZE+j], expectedValue);
        }
      }
    }
  }

  std::cout << "Running Stencil2d for Matrix B with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc2D>(
      queue, div, Stencil2d{}, d_B.data(), d_B_stencilled.data(), ndsize);

  // copy the results from the device to the host
  alpaka::memcpy(queue, h_B_stencilled, d_B_stencilled);

  // wait for all the operations to complete
  alpaka::wait(queue);

  std::cout << "Running MatMul with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc2D>(
      queue, div, MatMul{}, d_A_stencilled.data(), d_B_stencilled.data(), d_C.data(), ndsize);

  // copy the results from the device to the host
  alpaka::memcpy(queue, h_C, d_C);

  // wait for all the operations to complete
  alpaka::wait(queue);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // perform error check for matrix multiplication
  printf("Checking matrix multiplication operation:\n");
  for (int i = 0; i < DSIZE; i++) {
    for (int j = 0; j < DSIZE; j++) {
      int result = 0;
      for (int k = 0; k < DSIZE; k++) {
        result += h_A_stencilled[i*DSIZE+k] * h_B_stencilled[k*DSIZE+j];
      }
      if (h_C[i*DSIZE + j]!=result) {
        printf("    Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_C[i*DSIZE+j], result);
      }
    }
  }
}

int main() {
  // initialise the accelerator platform
  Platform platform;

  // require at least one device
  std::uint32_t n = alpaka::getDevCount(platform);
  if (n == 0) {
    exit(EXIT_FAILURE);
  }

  // use the single host device
  HostPlatform host_platform;
  Host host = alpaka::getDevByIdx(host_platform, 0u);
  std::cout << "Host:   " << alpaka::getName(host) << '\n';

  // use the first device
  Device device = alpaka::getDevByIdx(platform, 0u);
  std::cout << "Device: " << alpaka::getName(device) << '\n';

  testStencilMatMul(host, platform, device);
}
