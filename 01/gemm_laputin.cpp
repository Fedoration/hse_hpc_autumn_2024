#include "gemm_laputin.h"
#include <omp.h>

void init_matrix(int N, double* matrix) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = rand() % 100;
    }
}

void init_matrix_zero(int N, double* matrix) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = 0.0;
    }
}

void sequential_matrix_multiplication(int N, double *A, double *B, double *C) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}

void dgemm(int N, double *A, double *B, double *C) {
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}

void custom_dgemm(int N, double *A, double *B, double *C) {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        for (int j = thread_id; j < N; j += total_threads) {
            for (int i = 0; i < N; ++i) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[k * N + i] * B[j * N + k];
                }
                C[j * N + i] = sum;
            }
        }
    }
}

bool test(int N, double* matrix1, double* matrix2) {
    for (int i = 0; i < N * N; ++i) {
        if (matrix1[i] != matrix2[i]) {
            return false;
        }
    }
    return true;
}
