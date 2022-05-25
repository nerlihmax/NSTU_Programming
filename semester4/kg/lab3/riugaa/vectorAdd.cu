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
 * Сложение векторов C = A + B.
 */

#include <stdio.h>

// CUDA библиотеки
#include <cuda_runtime.h>
#include <helper_cuda.h>

/**
 * Вычисляет сумму векторов А и B и записывает в вектор С. Все векторы имеют одинаковое число элементов (numElements)
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; //Индекс нити в блоке и сетке блоков

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

int main(void) {
  // Код ошибки, которая может возникнуть во время работы
  cudaError_t err = cudaSuccess;

  // Вывод длины вектора, вычисление размера памяти под этот вектор
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Выделяем ОЗУ для вектора А
  float *h_A = (float *)malloc(size);

  // Выделяем ОЗУ для вектора В
  float *h_B = (float *)malloc(size);

  // Выделяем ОЗУ для выходного вектора С
  float *h_C = (float *)malloc(size);

  // Проверяем что выделение памяти прошло успешно
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Заполняем входные векторы A и B
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Выделяем видеопамять под вектор A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Выделяем видеопамять под вектор B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Выделяем видеопамять под вектор С
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Копируем входные векторы A и B в видеопамять
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); //видеопамять под вектор A

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); // Видеопамять под вектор В

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Запускаем функцию сложения векторов на GPU
  int threadsPerBlock = 256; //Количество нитей на блок
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock; // Количество блоков в сетке
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) { //Вывод сообщения об ошибке если она была
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Копируем результат сложения векторов из видеопамяти в ОЗУ
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Проверяем правильность сложения
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Освобождаем видеопамять для вектора А
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);// Освобождаем видеопамять для вектора В

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Освобождаем память из ОЗУ
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
