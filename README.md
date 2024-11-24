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

| Matrix Size | stencil_2d time | stencil_2d percentage | mat_mul time | mat_mul percentage |
|-------------|-----------------|-----------------------|--------------|--------------------|
| 256         | 0.010s          | 100.0%                | -            | -                  |
| 512         | 0.210s          | 100.0%                | -            | -                  |
| 1024        | 4.810s          | 99.6%                 | 0.010s       | 0.2%               |
| 2048        | 44.310s         | 99.8%                 | 0.030s       | 0.1%               |
| 4096        | 395.678s        | 99.9%                 | 0.070s       | 0.0%               |

It looks like the stencil operation is negligible in comparison to matrix multiplication. So we need to focus on speeding up matrix multiplication when optimizing in CUDA.


## Porting to CUDA

## Optimizing performance in CUDA

## Making use of Alpaka
