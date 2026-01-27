%%cuda 
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 40000000LL       // размер массива (можно увеличить до 100–400 млн)

int main(int argc, char** argv) {
    int rank, size;
    long long i;
    double start_time, end_time, local_sum = 0.0, global_sum = 0.0;
    
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Определяем размер локальной части для каждого процесса
    long long local_n = N / size;
    long long remainder = N % size;
    
    // Последние remainder процессов получают на 1 элемент больше
    if (rank < remainder) {
        local_n++;
    }

    // Вычисляем смещение (начальный индекс для текущего процесса)
    long long local_start = rank * (N / size) + (rank < remainder ? rank : remainder);

    // Выделяем память под локальную часть массива
    double* local_array = (double*)malloc(local_n * sizeof(double));
    if (local_array == NULL) {
        fprintf(stderr, "Процесс %d: ошибка выделения памяти\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Заполняем локальный массив (например, значениями от 1.0 до N)
    // Можно заменить на чтение из файла или генерацию по-другому
    for (i = 0; i < local_n; i++) {
        long long global_idx = local_start + i;
        local_array[i] = (double)(global_idx + 1);  // 1 + 2 + 3 + ... + N
    }

    // ------------------- ЗАМЕР ВРЕМЕНИ НАЧИНАЕТСЯ -------------------
    MPI_Barrier(MPI_COMM_WORLD);  // синхронизируем все процессы перед замером
    start_time = MPI_Wtime();

    // Локальная сумма
    for (i = 0; i < local_n; i++) {
        local_sum += local_array[i];
    }

    // Собираем суммы со всех процессов
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ------------------- ЗАМЕР ВРЕМЕНИ ЗАКАНЧИВАЕТСЯ -------------------
    end_time = MPI_Wtime();

    // Выводим результат только процесс 0
    if (rank == 0) {
        double expected = (double)N * (N + 1) / 2.0;
        printf("Количество процессов: %d\n", size);
        printf("Сумма элементов массива: %.0f\n", global_sum);
        printf("Ожидаемая сумма:      %.0f\n", expected);
        printf("Время выполнения:     %.4f секунд\n", end_time - start_time);
        printf("Размер массива:       %lld элементов (%.1f млн)\n\n", N, N/1e6);
    }

    // Освобождаем память
    free(local_array);

    MPI_Finalize();
    return 0;
}
