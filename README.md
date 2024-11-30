# TAC-HEP GPU Project

## C++ and CPU Profiling

I wrote C++ code to perform the task: `stencil_matmul.cc`.
I compiled using `g++ stencil_matmul.cc -o stencil_matmul` and got the following result after running `./stencil_matmul`:

```
h_A =
-                  -
| 1 1 1 0 1 0 0 1  |
| 1 0 1 1 1 1 1 0  |
| 1 0 0 1 0 0 0 0  |
| 1 0 0 1 1 1 0 0  |
| 1 0 0 1 0 1 1 1  |
| 1 1 0 0 0 1 0 0  |
| 0 1 1 0 1 1 1 1  |
| 1 0 1 1 0 1 0 0  |
-                  -

h_B=
-                  -
| 0 1 1 0 1 1 1 0  |
| 0 1 0 1 0 0 0 1  |
| 0 1 0 0 1 1 0 0  |
| 1 0 1 1 1 1 0 1  |
| 0 1 0 0 0 0 0 1  |
| 1 1 0 0 1 1 1 1  |
| 0 0 1 0 1 1 0 0  |
| 0 0 0 0 0 0 0 0  |
-                  -

h_A_stencilled =
-                  -
| 1 1 1 0 1 0 0 1  |
| 1 0 1 1 1 1 1 0  |
| 1 0 0 1 0 0 0 0  |
| 1 0 0 6 4 6 4 3  |
| 1 0 0 4 5 7 4 4  |
| 1 1 0 5 3 5 4 4  |
| 0 1 1 7 4 7 5 5  |
| 1 0 1 5 4 5 3 5  |
-                  -

h_B_stencilled =
-                  -
| 0 1 1 0 1 1 1 0  |
| 0 1 0 1 0 0 0 1  |
| 0 1 0 0 1 1 0 0  |
| 1 0 1 5 6 6 4 5  |
| 0 1 0 2 4 5 3 5  |
| 1 1 0 4 5 6 4 6  |
| 0 0 1 3 5 4 3 4  |
| 0 0 0 1 3 3 2 3  |
-                  -

h_C =
-                                               -
| 107 126 120 1066 1067 1112 1124 1132          |
| 138 146 133 1174 1155 1217 1243 1200          |
| 107 120 122 1030 1030 1059 1079 1068          |
| 1083 1232 1122 10053 10097 10671 10808 10521  |
| 1104 1210 1114 10005 9893 10566 10558 10399   |
| 1049 1169 1100 9633 9550 10198 10275 10135    |
| 1032 1186 1104 9832 9767 10472 10472 10401    |
| 1052 1180 1096 9680 9622 10308 10335 10255    |
-                                               -
```

I then compiled with optimizations enabled and debug symbols included: `g++ stencil_matmul.cc -O2 -g -o stencil_matmul_cpp_debug`.
After running the collect program using the following command: `vtune -collect hotspots -result-dir vtune_results ./stencil_matmul_cpp_debug`,
I examined the results using `vtune -report hotspots -result-dir vtune_results` and got the following output with my matrix size set to 1024 and radius set to 3:
```
vtune: Using result path `/afs/cern.ch/user/e/ekauffma/ekauffma_tachepgpu_project/vtune_results'
vtune: Executing actions 75 % Generating a report                              Function    CPU Time  CPU Time:Effective Time  CPU Time:Spin Time  CPU Time:Overhead Time  Module                    Function (Full)                                       Source File        Start Address
----------  --------  -----------------------  ------------------  ----------------------  ------------------------  ----------------------------------------------------  -----------------  -------------
mat_mul       4.850s                   4.850s                  0s                      0s  stencil_matmul_cpp_debug  mat_mul(int (*)[1024], int (*)[1024], int (*)[1024])  stencil_matmul.cc  0x4016e0
rand          0.010s                   0.010s                  0s                      0s  libc.so.6                 rand                                                  [Unknown]          0x41ed0
stencil_2d    0.010s                   0.010s                  0s                      0s  stencil_matmul_cpp_debug  stencil_2d(int (*)[1024], int (*)[1024])              stencil_matmul.cc  0x401520
vtune: Executing actions 100 % done
```

I ran for a variety of matrix sizes:

| Matrix Size | mat_mul time    | mat_mul percentage | stencil_2d time | stencil_2d percentage |
|-------------|-----------------|--------------------|-----------------|-----------------------|
| 256         | 0.010s          | 100.0%             | -               | -                     |
| 512         | 0.210s          | 100.0%             | -               | -                     |
| 1024        | 4.810s          | 99.6%              | 0.010s          | 0.2%                  |
| 2048        | 44.310s         | 99.8%              | 0.030s          | 0.1%                  |
| 4096        | 395.678s        | 99.9%              | 0.070s          | 0.0%                  |

It looks like the stencil operation is negligible in comparison to matrix multiplication. So we need to focus on speeding up matrix multiplication when optimizing in CUDA.


## Porting to CUDA

I then wrote CUDA code without any optimization or usage of shared memory: `stencil_matmul_a.cu`.
I compiled using `nvcc stencil_matmul_a.cu -o stencil_matmul_a`.
For profiling, I compiled using `nvcc -o stencil_matmul_a -lineinfo stencil_matmul_a.cu`, then ran

```
nsys profile -o profile_a ./stencil_matmul_a
nsys stats profile_a.nsys-rep
```

The timing results for the kernels are in the tables below. First, varying matrix size (keeping BLOCK_SIZE=32):

| Matrix Size | mat_mul time | mat_mul percentage    | stencil_2d time | stencil_2d percentage | total compute time |
|-------------|--------------|-----------------------|-----------------|-----------------------|--------------------|
| 256         | 1.2256e-5s   | 64.5%                 | 6.752e-6s       | 35.5%                 | 0.334071s          |
| 512         | 7.61067e-4s  | 95.8%                 | 3.3535e-5s      | 4.2%                  | 0.505172s          |
| 1024        | 5.55579e-3s  | 97.9%                 | 1.1654e-4s      | 2.1%                  | 0.972020s          |
| 2048        | 0.03482s     | 98.9%                 | 3.8457e-4s      | 1.1%                  | 0.892507s          |
| 4096        | 0.26800s     | 99.4%                 | 1.7442e-3s      | 0.6%                  | 1.324413s          |

Then varying block size (keeping DSIZE=1024):

| Block Size  | mat_mul time | mat_mul percentage    | stencil_2d time | stencil_2d percentage | total compute time |
|-------------|--------------|-----------------------|-----------------|-----------------------|--------------------|
| 8           | 7.20828e-3s  | 97.5%                 | 1.83899e-4s     | 2.5%                  | 1.063913s          |
| 16          | 9.03191e-3s  | 97.8%                 | 2.06203e-4s     | 2.2%                  | 0.491856s          |
| 32          | 5.43969e-3s  | 96.8%                 | 1.77564e-4s     | 3.2%                  | 0.966627s          |

We can see that this is way faster in CUDA than in C++. The stencil operation scales better than the matrix multiplication operation.
Varying the block size doesn't see a huge performance increase.

Before using managed memory, I will switch to using some shared memory, like we did in the Week 4 assignment. This is `stencil_matmul_b.cu`, which is compiled and tested in the same way as above. Here are the profiling results:

Varying matrix size:

| Matrix Size | mat_mul time | mat_mul percentage    | stencil_2d time | stencil_2d percentage | total compute time |
|-------------|--------------|-----------------------|-----------------|-----------------------|--------------------|
| 256         | 1.2621e-4s   | 65.4%                 | 6.9910e-5s      | 34.6%                 | 0.368899s          |
| 512         | 6.9941e-4s   | 81.4%                 | 1.6012e-4s      | 18.6%                 | 0.295196s          |
| 1024        | 6.4961e-3s   | 90.3%                 | 6.9989e-4s      | 9.7%                  | 0.325992s          |
| 2048        | 0.046062s    | 95.0%                 | 2.4288e-3s      | 5.0%                  | 0.342977s          |
| 4096        | 0.289980s    | 97.0%                 | 8.9117e-3s      | 3.0%                  | 0.798047s          |

Varying block size:

| Block Size  | mat_mul time | mat_mul percentage    | stencil_2d time | stencil_2d percentage | total compute time |
|-------------|--------------|-----------------------|-----------------|-----------------------|--------------------|
| 8           | 0.013367s    | 97.3%                 | 3.7493e-4s      | 2.7%                  | 0.322107s          |
| 16          | 5.7204e-3s   | 95.6%                 | 2.6102e-4s      | 4.4%                  | 0.305001s          |
| 32          | 6.4959e-3s   | 90.3%                 | 7.0088e-4s      | 9.7%                  | 0.280805s          |

It looks like this version of the stencil operation has worse performance. This might be due to how indices need to be handled for the halo cases, considering that we also need to do the matrix operation. I will revert back to version a before implementing managed memory. Before implementing managed memory, the profiling of the memory copies for `DSIZE=1024` is shown below:

| Time (%) | Total Time (ns) | Count | Avg (ns)  |  Med (ns) |  Min (ns) |  Max (ns) | StdDev (ns)   | Operation                    |
|----------|-----------------|-------|-----------|-----------|-----------|-----------|---------------|------------------------------|
| 65.7     | 2,543,165       | 7     | 363,309.3 | 333,591.0 | 328,439   | 533,426   | 75,364.8      | [CUDA memcpy Host-to-Device] |
| 34.3     | 1,330,429       | 3     | 443,476.3 | 444,532.0 | 431,349   | 454,548   | 11,635.5      | [CUDA memcpy Device-to-Host] |

| Total (MB) | Count | Avg (MB) | Med (MB) | Min (MB) | Max (MB) | StdDev (MB) |          Operation           |
|------------|-------|----------|----------|----------|----------|-------------|------------------------------|
| 29.360     | 7     | 4.194    | 4.194    | 4.194    | 4.194    | 0.000       | [CUDA memcpy Host-to-Device] |
| 12.583     | 3     | 4.194    | 4.194    | 4.194    | 4.194    | 0.000       | [CUDA memcpy Device-to-Host] |

Now to implement managed memory, which is done in `stencil_matmul_c.cu`. Again, for profiling, this program is compiled and ran as above.
The kernels won't see any performance increase, so I will not recreate those timing tables, but the profiling of the memory copies is shown below:

| Time (%) | Total Time (ns) | Count | Avg (ns)  | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation                            |
|----------|-----------------|-------|-----------|----------|----------|----------|-------------|--------------------------------------|
| 53.8     | 2,146,181       | 236   | 9,094.0   | 3,263.0  | 2,430    |  81,119  | 15,705.0    | [CUDA memcpy Unified Host-to-Device] |
| 46.2     | 1,843,521       | 120   | 15,362.7  | 4,176.0  | 1,919    |  91,871  | 24,108.0    | [CUDA memcpy Unified Device-to-Host] |
 
| Total (MB) | Count | Avg (MB) | Med (MB) | Min (MB) | Max (MB) | StdDev (MB) | Operation                             |
|------------|-------|----------|----------|----------|----------|-------------|---------------------------------------|
| 20.972     | 120   | 0.175    | 0.033    | 0.004    | 1.044    | 0.302       | [CUDA memcpy Unified Device-to-Host]  |
| 20.972     | 236   | 0.089    | 0.012    | 0.004    | 1.036    | 0.207       | [CUDA memcpy Unified Host-to-Device]  |

The total compute time sped up dramatically. Here are those results for different matrix sizes:

| Matrix Size | total compute time |
|-------------|--------------------|
| 256         | 0.040985s          |
| 512         | 0.036200s          |
| 1024        | 0.020276s          |
| 2048        | 0.128740s          |
| 4096        | 0.418634s          |

At larger matrix sizes, the total compute time becomes similar to that of the explicit memory copies, but the improvement at small matrix sizes is huge.

## Optimizing performance in CUDA

I've implemented non-default CUDA streams by creating two streams to run the stencil operations on matrix A and matrix B concurrently.
This is done in `stencil_matmul_d.cu`. I've already implemented shared memory in the stencil operation, but for the next step I'll try with matrix multiplication, as that is the current bottleneck. Here are the total compute times for Version D:

| Matrix Size | total compute time |
|-------------|--------------------|
| 256         | 0.029293s          |
| 512         | 0.030861s          |
| 1024        | 0.016853s          |
| 2048        | 0.121651s          |
| 4096        | 0.407939s          |

Again there appears to be some performance increase for the smaller matrix data points but not much for the larger matrices, since the matrix multiplication bottleneck dominates there.

The version `stencil_matmul_e.cu` implements shared memory in both the `stencil_2d` and `mat_mul` kernels as well as managed memory and nondefault streams.

The timing is shown below:

Varying matrix size:

| Matrix Size | mat_mul time | mat_mul percentage    | stencil_2d time | stencil_2d percentage | total compute time |
|-------------|--------------|-----------------------|-----------------|-----------------------|--------------------|
| 256         | 6.6463e-04s  | 31.4%                 | 1.4489e-3s      | 68.6%                 | 0.056233s          |
| 512         | 1.0914e-03s  | 24.1%                 | 3.4324e-3s      | 75.9 %                | 0.055880s          |
| 1024        | 6.8867e-03s  | 39.5%                 | 0.010569s       | 60.5%                 | 0.073639s          |
| 2048        | 0.04417s     | 46.4%                 | 0.050926s       | 53.6%                 | 0.096272s          |
| 4096        | 0.24612s     | 64.5%                 | 0.135693s       | 35.5%                 | 0.358389s          |

Varying block size:

| Block Size  | mat_mul time | mat_mul percentage    | stencil_2d time | stencil_2d percentage | total compute time |
|-------------|--------------|-----------------------|-----------------|-----------------------|--------------------|
| 8           | 0.01044s     | 51.0%                 | 0.01042s        | 49.0%                 | 0.056380s          |
| 16          | 6.7337e-3s   | 44.0%                 | 8.5807e-3s      | 56.0%                 | 0.053103s          |
| 32          | 0.01044s     | 48.7%                 | 0.01102s        | 51.3%                 | 0.023246s          |

The performance increase in the matrix multiplication kernel is now pretty obvious. This is the best version of the CUDA code and thus will be used to convert to Alpaka. For reference here are the profiling results for the memory operations:

| Time (%) | Total Time (ns) | Count  | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation                            |
|----------|-----------------|--------|----------|----------|----------|----------|-------------|--------------------------------------|
| 63.5     | 3,214,334       | 699    | 4,598.5  | 3,550.0  | 2,399    | 72,128   | 4,491.1     | [CUDA memcpy Unified Host-to-Device] |
| 36.5     | 1,847,984       | 120    | 15,399.9 | 4,079.5  | 1,919    | 87,136   | 24,084.3    | [CUDA memcpy Unified Device-to-Host] |

| Total (MB) | Count | Avg (MB) | Med (MB) | Min (MB) | Max (MB) | StdDev (MB) | Operation                              |
|------------|-------|----------|----------|----------|----------|-------------|----------------------------------------|
| 20.972     | 120   | 0.175    | 0.033    | 0.004    |  1.044   | 0.302       | [CUDA memcpy Unified Device-to-Host]   |
| 20.972     | 699   | 0.030    | 0.016    | 0.004    |  0.918   | 0.059       | [CUDA memcpy Unified Host-to-Device]   |



## Making use of Alpaka

I installed Alpaka through the git repository using the following instructions:

```
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/install/ ..
cmake --install .
cmake -Dalpaka_ACC_GPU_CUDA_ENABLE=ON ...
```

I copied the `config.h` and `WorkDiv.hpp` files from the tutorial shown in lecture: [https://github.com/fwyzard/intro_to_alpaka/tree/master/alpaka](https://github.com/fwyzard/intro_to_alpaka/tree/master/alpaka).

My first implementation in Alpaka is located at `stencil_matmul_alpaka_a.cc`. Here I was not able to implement any parallelization utilizing the GPUs, and as such this code only works with 1 block and for matrix sizes under 32x32. I will work on parallelization next.
I compiled the program using `nvcc -x cu -std=c++20 -O2 -g --expt-relaxed-constexpr -I /afs/hep.wisc.edu/home/ekauffma/install/include -DALPAKA_ACC_GPU_CUDA_ENABLED stencil_matmul_alpaka_a.cc -o stencil_matmul_alpaka` and then ran using `./stencil_matmul_alpaka_a`.

My second implementation in Alpaka is located at `stencil_matmul_alpaka_b.cc`. I needed to use the 2D methods in order to get the kernels to work properly. These implementations don't use shared memory or any of the CUDA optimizations, but the program still runs pretty quickly. Here are the compute timing results, with the same settings used in previous timings:

| Matrix Size | total compute time |
|-------------|--------------------|
| 256         | 0.022629s          |
| 512         | 0.033952s          |
| 1024        | 0.127945s          |
| 2048        | 0.647309s          |
| 4096        | 4.275024s          |

| Block Size  | total compute time |
|-------------|--------------------|
| 8           | 0.422716s          |
| 16          | 0.199956s          |
| 32          | 0.107521s          |

Interestingly, block size seems to have a much larger effect here.




