%%cuda
#include <cuda_runtime.h>   // CUDA Runtime API
#include <iostream>
#include <chrono>

#define N 1000000           // Размер массива
#define BLOCK_SIZE 256      // Число потоков в блоке

// CUDA-ядро для вычисления префиксной суммы внутри блока
__global__ void prefixScan(float* d_in, float* d_out, int n) {

    // Разделяемая память для текущего блока
    __shared__ float temp[BLOCK_SIZE];

    // Локальный индекс потока в блоке
    int tid = threadIdx.x;

    // Глобальный индекс элемента массива
    int idx = blockIdx.x * blockDim.x + tid;

    // Загрузка данных из global memory в shared memory
    if (idx < n)
        temp[tid] = d_in[idx];
    else
        temp[tid] = 0.0f;

    // Синхронизация всех потоков блока
    __syncthreads();

    // Параллельное префиксное суммирование (inclusive scan)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {

        float val = 0.0f;

        // Чтение элемента с заданным смещением
        if (tid >= offset)
            val = temp[tid - offset];

        // Синхронизация перед записью
        __syncthreads();

        // Добавление значения к текущему элементу
        temp[tid] += val;

        // Синхронизация после обновления
        __syncthreads();
    }

    // Запись результата из shared memory обратно в global memory
    if (idx < n)
        d_out[idx] = temp[tid];
}

int main() {
    float* h_in = new float[N];    // Входной массив CPU
    float* h_out = new float[N];   // Выходной массив CPU

    // Инициализация входного массива
    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;

    float *d_in, *d_out;           // Указатели на память GPU

    // Выделение памяти в global memory GPU
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // Копирование данных с CPU на GPU
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // Расчёт количества CUDA-блоков
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Начало измерения времени GPU-вычислений
    auto start = std::chrono::high_resolution_clock::now();

    // Запуск CUDA-ядра
    prefixScan<<<blocks, BLOCK_SIZE>>>(d_in, d_out, N);

    // Ожидание завершения выполнения всех потоков
    cudaDeviceSynchronize();

    // Конец измерения времени
    auto end = std::chrono::high_resolution_clock::now();

    // Копирование результата обратно на CPU
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Вычисление времени выполнения
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Вывод результата и времени выполнения
    std::cout << "GPU last element (block scan): " << h_out[N - 1] << std::endl;
    std::cout << "GPU time: " << duration.count() << " microseconds" << std::endl;

    // Освобождение памяти GPU
    cudaFree(d_in);
    cudaFree(d_out);

    // Освобождение памяти CPU
    delete[] h_in;
    delete[] h_out;

    return 0;
}
