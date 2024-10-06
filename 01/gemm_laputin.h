#pragma once
#include <iostream>
#include <omp.h>

void init_matrix(int N, double* matrix);
void init_matrix_zero(int N, double* matrix);
void sequential_matrix_multiplication(int N, double *A, double *B, double *C);
void dgemm(int N, double *A, double *B, double *C);
void custom_dgemm(int N, double *A, double *B, double *C);
bool test(int N, double* matrix1, double* matrix2);
