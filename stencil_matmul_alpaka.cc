#include <cassert>
#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>

#include "config.h"
#include "WorkDiv.hpp"

#define DSIZE 8

struct MatMul {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator() (TAcc const& acc,
                                   T const* __restrict__ A,
                                   T const* __restrict__ B,
                                   T* __restrict__ C,
                                   uint32_t size) const {

        uint32_t size_1d = std::sqrt(size);

        for (auto i : alpaka::uniformElements(acc, size_1d)) {
            for (auto j : alpaka::uniformElements(acc, size_1d)) {
                int result = 0;
                for (auto k : alpaka::uniformElements(acc, size_1d)) {
                    result += A[i*size_1d+k] * B[k*size_1d+j];
                }
                C[i*size_1d+j] = result;
            }
        }
    }
};

void testMatMul(Host host, Platform platform, Device device) {
  constexpr uint32_t size = DSIZE * DSIZE;

  // run the test the given device
  auto queue = Queue{device};

  auto div = makeWorkDiv<Acc1D>(1, 1);

  // initialize matrices
  auto h_A = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
  auto h_B = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
  auto h_C = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);

  // filling initial values of matrices
  srand(time(nullptr));
  for (int i = 0; i < DSIZE; i++) {
    for (int j = 0; j < DSIZE; j++) {
      h_A[i*DSIZE+j] = rand()%2;
      h_B[i*DSIZE+j] = rand()%2;
      h_C[i*DSIZE+j] = 0;
    }
  }

  // allocate input and output buffers on the device
  auto d_A = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
  auto d_B = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
  auto d_C = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);

  // copy the input data to the device; the size is known from the buffer objects
  alpaka::memcpy(queue, d_A, h_A);
  alpaka::memcpy(queue, d_B, h_B);

  // fill the output buffer with zeros; the size is known from the buffer objects
  alpaka::memset(queue, d_C, 0x00);

  std::cout << "Testing MatMul with scalar indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
  alpaka::exec<Acc1D>(
      queue, div, MatMul{}, d_A.data(), d_B.data(), d_C.data(), size);

  // copy the results from the device to the host
  alpaka::memcpy(queue, h_C, d_C);
                                                
  // wait for all the operations to complete
  alpaka::wait(queue);

  // perform error check for matrix multiplication
  for (int i = 0; i < DSIZE; i++) {
    for (int j = 0; j < DSIZE; j++) {
      int result = 0;
      for (int k = 0; k < DSIZE; k++) {
        result += h_A[i*DSIZE+k] * h_B[k*DSIZE+j];
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

  testMatMul(host, platform, device);
}
