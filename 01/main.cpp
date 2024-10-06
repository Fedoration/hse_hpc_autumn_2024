#include <iostream>
#include "gemm_laputin.h"

int main() {
    int N = 3;
    double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double B[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double C[N * N];
    double C_sequential[N * N];

    // Sequential multiplication
    sequential_matrix_multiplication(N, A, B, C_sequential);

    // Parallel multiplication with OpenMP
    dgemm(N, A, B, C);

    // Test if the results are the same
    if (test(N, C, C_sequential)) {
        std::cout << "The matrices are the same!" << std::endl;
    } else {
        std::cout << "The matrices are different!" << std::endl;
    }

    return 0;
}
