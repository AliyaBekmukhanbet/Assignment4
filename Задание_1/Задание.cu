%%cuda 
#include <cuda_runtime.h>   // CUDA Runtime API
#include <iostream>
#include <chrono>

#define N 100000            // Размер массива
#define BLOCK_SIZE 256      // Количество потоков в блоке

// CUDA-ядро для суммирования элементов массива
__global__ void sumKernel(float* d_arr, float* d_sum, int n) {

    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if (idx < n) {
        // Атомарное сложение в глобальной памяти
        atomicAdd(d_sum, d_arr[idx]);
    }
}

int main() {
    float* h_arr = new float[N];   // Массив в памяти CPU
    float h_sum = 0.0f;            // Результат суммы

    // Инициализация массива
    for (int i = 0; i < N; i++)
        h_arr[i] = 1.0f;

    float *d_arr, *d_sum;          // Указатели на память GPU

    // Выделение памяти в global memory GPU
    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    // Копирование массива с CPU на GPU
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    // Инициализация суммы на GPU нулём
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    // Расчёт количества CUDA-блоков
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Начало измерения времени GPU-вычислений
    auto start = std::chrono::high_resolution_clock::now();

    // Запуск CUDA-ядра
    sumKernel<<<blocks, BLOCK_SIZE>>>(d_arr, d_sum, N);

    // Ожидание завершения всех потоков GPU
    cudaDeviceSynchronize();

    // Конец измерения времени
    auto end = std::chrono::high_resolution_clock::now();

    // Копирование результата обратно на CPU
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Вычисление времени выполнения
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Вывод результатов
    std::cout << "GPU sum: " << h_sum << std::endl;
    std::cout << "GPU time: " << duration.count() << " microseconds" << std::endl;

    // Освобождение памяти GPU
    cudaFree(d_arr);
    cudaFree(d_sum);

    // Освобождение памяти CPU
    delete[] h_arr;

    return 0;
}
