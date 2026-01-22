%%cuda
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 1000000
#define BLOCK_SIZE 256

// CUDA-ядро обработки массива
__global__ void multiplyKernel(float* d_arr, float scalar, int n) {

    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n)
        d_arr[idx] *= scalar;
}

int main() {
    float* h_arr = new float[N];
    float scalar = 2.0f;

    for (int i = 0; i < N; i++)
        h_arr[i] = 1.0f;

    float* d_arr;
    cudaMalloc(&d_arr, N * sizeof(float));

    // Копирование данных с CPU на GPU
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start = std::chrono::high_resolution_clock::now();

    // Запуск CUDA-ядра
    multiplyKernel<<<blocks, BLOCK_SIZE>>>(d_arr, scalar, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    // Копирование результата обратно
    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std :: cout << "GPU time: " << duration.count() << " microseconds" << std :: endl;

    cudaFree(d_arr);
    delete[] h_arr;
    return 0;
}
