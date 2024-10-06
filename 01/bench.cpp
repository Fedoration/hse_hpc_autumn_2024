#include <iostream>
#include <omp.h>
#include <vector>
#include "gemm_laputin.h"

int main() {
    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    std::vector<int> matrix_sizes = {500, 1000, 1500};

    for (int N : matrix_sizes) {
        double* A = new double[N * N];
        double* B = new double[N * N];
        double* C = new double[N * N];

        init_matrix(N, A);
        init_matrix(N, B);
        init_matrix_zero(N, C);

        std::cout << "Matrix size: " << N << "x" << N << std::endl;

        for (int n_threads : thread_counts) {
            double start = omp_get_wtime();

            omp_set_num_threads(n_threads);
            dgemm(N, A, B, C);

            double stop = omp_get_wtime();
            double elapsed_time = stop - start;

            std::cout << "Runtime "
            << n_threads << "/" << max_threads << " cores"
            << ": " << elapsed_time << " seconds" << std::endl;
        }

        delete[] A;
        delete[] B;
        delete[] C;

        std::cout << "----------------------------------" << std::endl;
    }
    
    return 0;
}
