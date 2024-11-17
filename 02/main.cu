#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>

#define BLOCK_SIZE 16


__global__ void matrixMultiplyGlobal(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[k * N + row] * B[col * N + k];
        }
        C[col * N + row] = sum;
    }
}


void matrixMultiplyCPU(const double* A, const double* B, double* C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[k * N + row] * B[col * N + k];
            }
            C[col * N + row] = sum;
        }
    }
}


bool validateResult(const double* gpuResult, const double* cpuResult, int N, double epsilon = 1e-6) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(gpuResult[i] - cpuResult[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": GPU result = " << gpuResult[i]
                      << ", CPU result = " << cpuResult[i] << "\n";
            return false;
        }
    }
    return true;
}


void testGlobalMemoryMatrixMultiply(int N) {
    size_t size = N * N * sizeof(double);

    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    cudaMallocHost(&h_C_cpu, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiplyGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Global memory multiplication time: " << milliseconds << " ms\n";
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Проверка результата
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    if (validateResult(h_C, h_C_cpu, N)) {
        std::cout << "Global memory matrix multiplication test PASSED!\n";
    } else {
        std::cout << "Global memory matrix multiplication test FAILED!\n";
    }

    // Очистка памяти
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void testPinnedMemoryMatrixMultiply(int N) {
    size_t size = N * N * sizeof(double);

    double *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C_gpu, size);
    cudaMallocHost(&h_C_cpu, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiplyGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Pinned memory multiplication time: " << milliseconds << " ms\n";

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Проверка результата
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    if (validateResult(h_C_gpu, h_C_cpu, N)) {
        std::cout << "Pinned memory test PASSED!\n";
    } else {
        std::cout << "Pinned memory test FAILED!\n";
    }

    // Очистка памяти
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_gpu);
    cudaFreeHost(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void testUnifiedMemoryMatrixMultiply(int N) {
    size_t size = N * N * sizeof(double);

    double *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiplyGlobal<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Unified memory multiplication time: " << milliseconds << " ms\n";

    cudaDeviceSynchronize();

    // Проверка результата
    double *C_cpu = new double[N * N];
    matrixMultiplyCPU(A, B, C_cpu, N);
    if (validateResult(C, C_cpu, N)) {
        std::cout << "Unified memory test PASSED!\n";
    } else {
        std::cout << "Unified memory test FAILED!\n";
    }

    // Очистка памяти
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] C_cpu;
}


void testStreamsMatrixMultiply(int N, int numStreams) {
    size_t size = N * N * sizeof(double);
    size_t chunkSize = (N / numStreams) * N;

    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    cudaMallocHost(&h_C_cpu, size);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < numStreams; ++i) {
        size_t offset = i * chunkSize;
        cudaMemcpyAsync(d_A + offset, h_A + offset, chunkSize * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B + offset, h_B + offset, chunkSize * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixMultiplyGlobal<<<blocksPerGrid, threadsPerBlock, 0, streams[0]>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Streams multiplication time: " << milliseconds << " ms\n";

    for (int i = 0; i < numStreams; ++i) {
        size_t offset = i * chunkSize;
        cudaMemcpyAsync(h_C + offset, d_C + offset, chunkSize * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Проверка результата
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    if (validateResult(h_C, h_C_cpu, N)) {
        std::cout << "Streams test PASSED!\n";
    } else {
        std::cout << "Streams test FAILED!\n";
    }

    // Очистка памяти
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


__global__ void matrixMulShared(double* A, double* B, double* C, int N) {
    __shared__ double Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    double Cvalue = 0.0f;

    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < N && t * BLOCK_SIZE + tx < N)
            Asub[ty][tx] = A[(t * BLOCK_SIZE + tx) * N + row];
        else
            Asub[ty][tx] = 0.0f;

        if (col < N && t * BLOCK_SIZE + ty < N)
            Bsub[ty][tx] = B[col * N + t * BLOCK_SIZE + ty]; 
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            Cvalue += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[col * N + row] = Cvalue;
}


void testSharedMemoryMatrixMultiply(int N) {
    size_t size = N * N * sizeof(double);

    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    cudaMallocHost(&h_C_cpu, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Shared memory multiplication time: " << milliseconds << " ms\n";

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Проверка результата
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    if (validateResult(h_C, h_C_cpu, N)) {
        std::cout << "Shared memory test PASSED!\n";
    } else {
        std::cout << "Shared memory test FAILED!\n";
    }

    // Очистка памяти
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void checkCudaStatus(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasStatus(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}


void testCublasMatrixMultiply(int N) {
    size_t size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    cudaMallocHost(&h_C_cpu, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle), "Failed to create cuBLAS handle");

    const float alpha = 1.0f; 
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Выполняем умножение матриц: C = alpha * A * B + beta * C
    checkCublasStatus(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,  
                    N, N, N,                   
                    &alpha,                    
                    d_A, N,                    
                    d_B, N,                    
                    &beta,                     
                    d_C, N),                   
        "Failed to execute cublasSgemm"
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Cublas multiplication time: " << milliseconds << " ms\n"; 

    checkCudaStatus(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy C to host");

    checkCublasStatus(cublasDestroy(handle), "Failed to destroy cuBLAS handle");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    int N = 1024;

    testGlobalMemoryMatrixMultiply(N);
    testPinnedMemoryMatrixMultiply(N);
    testUnifiedMemoryMatrixMultiply(N);
    testStreamsMatrixMultiply(N, 4);
    testSharedMemoryMatrixMultiply(N);
    testCublasMatrixMultiply(N);
}
