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

## Porting to CUDA

## Optimizing performance in CUDA

## Making use of Alpaka
