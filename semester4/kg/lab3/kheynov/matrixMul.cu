/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// Подключение системных библиотек
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Утилиты нужные для работы CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

/**
 * Перемножение матриц на GPU (CUDA ядра): C = A * B
 * wA это размерность матрицы A, а wB это размерность матрицы B
 * C == результат перемножения
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
    float *B, int wA,
    int wB) {

  // Индекс блока CUDA
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Индекс Thread'a (да, я не люблю называть их "нитями")
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Индекс первой "подматрицы" матрицы А, которая будет обрабатываться блоком
  int aBegin = wA * BLOCK_SIZE * by;

  // Индекс первой "подматрицы" матрицы А, которая будет обрабатываться блоком
  int aEnd   = aBegin + wA - 1;

  // Размер шага, с которым мы будем итерироваться по подматрицам А
  int aStep  = BLOCK_SIZE;

  // Индекс первой "подматрицы" матрицы B, которая будет обрабатываться блоком
  int bBegin = BLOCK_SIZE * bx;

  // Размер шага, с которым мы будем итерироваться по подматрицам B
  int bStep  = BLOCK_SIZE * wB;

  // Переменная Csub хранит элемент блока подматрицы, который был посчитан в thread'e
  float Csub = 0;

  // Проходим в цикле по всем подматрицам A и B
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {

    // Выделяем массив As в общей памяти чтобы хранить подматрицу A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Аналогично для подматрицы А
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Загружаем матрицы из видеопамяти в общую. Каждый поток загружает один элемент из каждой матрицы
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    //Синхронизируемся чтобы убедиться что все матрицы загрузились
    __syncthreads();


    // Перемножаем матрицы, каждый поток вычисляет один элемент из блока подматрицы

#pragma unroll // Оптимизация компилятора, позволяет провернуть следующее:
      /**
        for ( int i = 0; i < 5; i++ )
          b[i] = i;
        
        =============
        
        b[0] = 0;
        b[1] = 1;
        b[2] = 2;
        b[3] = 3;
        b[4] = 4; 
       */

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx]; // Собственно перемножение подматриц
    }

    // Синхронизируемся чтобы убедиться что мы всё посчитали 
    // прежде чем загружать следующие подматрицы
    __syncthreads();
  }

  // Записываем блок подматрицы в видеопамять
  // Каждый элемент отдельным thread'ом 
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {  // метод для заполнения массива значениями
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Простенький тест перемножения матриц
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
  
  // Выделение памяти на ОЗУ (не видеопамять) для хранения матриц А и В
  unsigned int size_A = dimsA.x * dimsA.y; //Размер матрицы А
  unsigned int mem_size_A = sizeof(float) * size_A; //Размер выделяемой памяти
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y; // Аналогично для матрицы В
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;

  // Инициализируем память ОЗУ, заполняем матрицы
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  // Выделяем память на GPU
  float *d_A, *d_B, *d_C;

  // Выделяем память в ОЗУ для результирующей матрицы С
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) { // Выводим ошибку если не получилось выделить память
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

  // Подключаем CUDA Event'ы для подсчета времени вычисления
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Копируем память из ОЗУ на GPU (матрицы)
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  // Устанавливаем параметры запуска (блоки, сетка блоков)
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  printf("Computing result using CUDA Kernel...\n");

  // Выполняем "разогревочные" прогоны перемножения
  if (block_size == 16) {
    MatrixMulCUDA<16>
        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  } else {
    MatrixMulCUDA<32>
        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  // Запускаем таймер
  checkCudaErrors(cudaEventRecord(start, stream));

  // Количество итераций
  int nIter = 300;

  for (int j = 0; j < nIter; j++) { //Выполняем вычисления
    if (block_size == 16) {
      MatrixMulCUDA<16>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else {
      MatrixMulCUDA<32>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }

  // Останавливаем таймер
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Ждем пока таймер остановится
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop)); // Вычисляем затраченное время

  // Вычисляем и выводим производительность
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Копируем матрицу с результатом вычисления с GPU на ОЗУ
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: ");
  bool correct = true;

  // Вычисляем ошибку по формуле
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // Машинный "ноль"

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Освобождаем память
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance "
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}


/**
 * Программа main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // Эта функция выберет лучший из доступных видеоадаптеров
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
  dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

  // Ширина матрицы А
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // Высота матрицы А
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // Ширина матрицы В
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // Высота матрицы В
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) { //Если ширина матрицы А и высота матрицы В не совпадает, выдаем ошибку, мы не сможем 
  // перемножить такие матрицы
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart()); //запускаем измеритель производительности
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB); //Выполняем вычисления
  checkCudaErrors(cudaProfilerStop()); //Останавливаем измеритель производительности

  exit(matrix_result);
}
