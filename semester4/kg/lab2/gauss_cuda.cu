#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h> 
#include <stdio.h>
#include <time.h>

#define MAX_THREADS_PER_BLOCK 1024


#define startSize 5
#define maxSize  5000
#define step 5

int N = (maxSize - startSize) / step + 1; //quantity of elements

float* fillUpMatrix(int size) //Filling up matrix with elements in range [-10, 10]
{
    if (size <= 0)
        return nullptr;

    float* matrix = (float*)malloc(size * size * sizeof(float));
    int min = -10, max = 10;
    srand(time(0));

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            matrix[i * size + j] = rand() % (max - min + 1) + min;
            if (i == j && matrix[i * size + j] == 0)
                matrix[i * size + j]++;
        }
    }

    return matrix;
}

__global__ void multiplyDetWithElement(long double* det, float* matrix, int currentDiagonalElemIndex)
{
    *det *= matrix[currentDiagonalElemIndex];
}

__global__ void fillCoefsArray(float* coefs, float* matrix, int size, int currentDiagonalElemIndex, int startNumber = 0)
{
    int i = startNumber + blockDim.x * blockIdx.x + threadIdx.x;
    int elemToZeroIndex = currentDiagonalElemIndex + size * (i + 1);

    coefs[(elemToZeroIndex / size) - 1] = -matrix[elemToZeroIndex] / matrix[currentDiagonalElemIndex];
}

__global__ void multiplyElemWithCoef(float* matrix, int size, int currentDiagonalElemRow, float* coefs, int startNumber = 0)
{
    int number = startNumber + blockDim.x * blockIdx.x + threadIdx.x;
    int columnsCount = size - currentDiagonalElemRow;

    int row = currentDiagonalElemRow + 1 + (number / columnsCount);
    int column = currentDiagonalElemRow + (number % columnsCount);

    matrix[row * size + column] += coefs[row - 1] * matrix[currentDiagonalElemRow * size + column];
}

void getNumberOfBlocksAndThreads(int elemsCount, int* blocks, int* threads, int* remains)
{
    if (elemsCount < MAX_THREADS_PER_BLOCK)
    {
        *blocks = 1;
        *threads = elemsCount;
    }
    else
    {
        *blocks = elemsCount / MAX_THREADS_PER_BLOCK;
        *threads = MAX_THREADS_PER_BLOCK;
    }
    *remains = elemsCount - *blocks * *threads;
}

cudaError_t allocateMemory(float* matrix, float** gpuMatrix, int size, float** gpuCoefs, long double** gpuDet)
{
    cudaError_t status = cudaMalloc(gpuMatrix, size * size * sizeof(float));
    status = cudaMemcpy(*gpuMatrix, matrix, size * size * sizeof(float), cudaMemcpyHostToDevice);

    status = cudaMalloc(gpuCoefs, (size - 1) * sizeof(float));

    status = cudaMalloc(gpuDet, 1 * sizeof(long double));
    long double det = 1;
    status = cudaMemcpy(*gpuDet, &det, 1 * sizeof(long double), cudaMemcpyHostToDevice);

    return status;
}

long double gaussMethod(float* matrix, int size)
{
    long double det = 1;

    long double* _det = nullptr;
    float* _matrix = nullptr;
    float* _coefs = nullptr;

    if (allocateMemory(matrix, &_matrix, size, &_coefs, &_det) != cudaSuccess)
    {
        printf("Allocate memory error!\n");
        goto freeMemory;
    }

    for (int i = 0; i < size; i++) {

        int curDiagonalElemIndex = i * size + i;
        multiplyDetWithElement <<< 1, 1 >>> (_det, _matrix, curDiagonalElemIndex);


        int blocksCount, threadsCount, remains;
        getNumberOfBlocksAndThreads(size - i - 1, &blocksCount, &threadsCount, &remains);

        fillCoefsArray <<< blocksCount, threadsCount >>> (_coefs, _matrix, size, curDiagonalElemIndex);
        fillCoefsArray <<< 1, remains >>> (_coefs, _matrix, size, curDiagonalElemIndex, blocksCount * threadsCount);

        int elemsCount = (size - 1 - i) * (size - i);
        getNumberOfBlocksAndThreads(elemsCount, &blocksCount, &threadsCount, &remains);

        cudaDeviceSynchronize();

        multiplyElemWithCoef <<<blocksCount, threadsCount >>> (_matrix, size, i, _coefs);
        multiplyElemWithCoef << <1, remains >> > (_matrix, size, i, _coefs, blocksCount * threadsCount);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(&det, &_det[0], sizeof(long double), cudaMemcpyDeviceToHost);

    freeMemory:
    cudaFree(_matrix);
    cudaFree(_det);
    cudaFree(_coefs);
    return det;
}

using namespace std;

int main()
{
    printf("Starting calculation...\n");
    for (int size = startSize, i = 0; size <= maxSize; size += step, i++)
    {
        float* matrix = fillUpMatrix(size);

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        //------
        cudaEventRecord(start, 0);
        gaussMethod(matrix, size);
        cudaEventRecord(stop, 0);
        //------

        cudaEventSynchronize(stop);
        
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);

        free(matrix);
        printf("%d ", size);
        printf("%f\n", time);
    }
    return 0;
}
