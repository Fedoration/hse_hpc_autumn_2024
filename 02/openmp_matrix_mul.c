#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>


void matrixMulOpenMP(const double* A, const double* B, double* C, int N) {
    #pragma omp target teams distribute parallel for collapse(2) map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}


void matrixMulCPU(const double* A, const double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}


int compareMatrices(const double* C1, const double* C2, int N, double epsilon) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            printf("Несовпадение найдено в элементе %d: C1 = %f, C2 = %f\n", i, C1[i], C2[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    const int N = 1024;

    double* A = (double*)malloc(N * N * sizeof(double));
    double* B = (double*)malloc(N * N * sizeof(double));
    double* C_gpu = (double*)malloc(N * N * sizeof(double));
    double* C_cpu = (double*)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C_gpu == NULL || C_cpu == NULL) {
        printf("Memory alloc error.\n");
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C_gpu[i] = 0.0;
        C_cpu[i] = 0.0;
    }

    // Вычисления на CPU
    double cpu_start = omp_get_wtime();
    matrixMulCPU(A, B, C_cpu, N);
    double cpu_end = omp_get_wtime();
    printf("CPU Runtime: %f s.\n", cpu_end - cpu_start);

    // Вычисления на GPU
    double gpu_start = omp_get_wtime();
    matrixMulOpenMP(A, B, C_gpu, N);
    double gpu_end = omp_get_wtime();
    printf("GPU Runtime: %f s.\n", gpu_end - gpu_start);

    // Сравнение результатов
    if (compareMatrices(C_cpu, C_gpu, N, 1e-6)) {
        printf("OpenMP test PASSED!\n");
    } else {
        printf("OpenMP test FAILED!\n");
    }

    // Освобождение памяти
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    return 0;
}
