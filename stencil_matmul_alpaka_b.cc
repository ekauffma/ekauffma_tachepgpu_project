#include <cassert>
#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>

#include "config.h"
#include "WorkDiv.hpp"

#define DSIZE 64
#define RADIUS 1

struct Stencil2d {
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC void operator() (TAcc const& acc,
                                 T const* __restrict__ A,
                                 T* __restrict__ A_stencilled,
                                 Vec2D size) const {

    for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
      auto index = (ndindex[0] * size[1] + ndindex[1]);
      A_stencilled[index] = 2;
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
