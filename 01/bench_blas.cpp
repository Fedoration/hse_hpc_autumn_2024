#include <iostream>
#include <omp.h>
#include <cblas.h>
#include "gemm_laputin.h"

void blas_dgemm(int N, double *A, double *B, double *C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}

int main() {
    int N = 1000;
    double *A = new double[N * N];
    double *B = new double[N * N];
    double *C_sequential = new double[N * N];
    double *C_parallel = new double[N * N];
    double *C_blas = new double[N * N];

    // Инициализация матриц
    init_matrix(N, A);
    init_matrix(N, B);
    init_matrix_zero(N, C_sequential);
    init_matrix_zero(N, C_parallel);
    init_matrix_zero(N, C_blas);

    double times;
    // Запуск последовательной версии
    times = omp_get_wtime();
    
    sequential_matrix_multiplication(N, A, B, C_sequential);
    
    times = omp_get_wtime() - times;
    std::cout << "Время выполенения последовательной версии: " << times << std::endl;

    // Запуск параллельной версии
    times = omp_get_wtime();

    dgemm(N, A, B, C_parallel);
    
    times = omp_get_wtime() - times;
    std::cout << "Время выполенения параллельной версии: " << times << std::endl;

    // Запуск OpenBlas версии
    times = omp_get_wtime();

    blas_dgemm(N, A, B, C_blas);

    times = omp_get_wtime() - times;
    std::cout << "Время выполенения OpenBlas версии: " << times << std::endl;

    if (test(N, C_sequential, C_parallel)) {
        std::cout << "Результаты совпадают! C_parallel" << std::endl;
    } else {
        std::cout << "Результаты отличаются! Неправильный ответ C_parralel" << std::endl;
    }

    if (test(N, C_sequential, C_blas)) {
        std::cout << "Результаты совпадают! C_blas" << std::endl;
    } else {
        std::cout << "Результаты отличаются! Неправильный ответ C_blas" << std::endl;
    }

    // Освобождение ресурсов
    delete[] A;
    delete[] B;
    delete[] C_sequential;
    delete[] C_parallel;
    delete[] C_blas;
 
    return 0;
}
