#include <limits.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace std;

#define THREADS_PER_BLOCK 64

struct Result {
    int value;
    int number;
    int dozens;
    bool operator==(Result res) { return value == res.value; }
};

vector<int> eratosphenesSieveSequential(int min, int max, int *count) {
    if (max < 2) {
        return vector<int>{};
    }
    vector<int> primeNumbers{};

    primeNumbers.push_back(2);

    int curCount = 1;
    int minBorder = -1;

    for (int n = 3; n < max; n += 2) {
        bool isPrime = true;

        for (int i = 1; i < curCount; i++) {
            if (primeNumbers[i] * primeNumbers[i] > n) break;
            if (n % primeNumbers[i] == 0) {
                isPrime = false;
                break;
            }
        }

        if (!isPrime) continue;

        if (n < min) minBorder = curCount;
        primeNumbers.push_back(n);
    }

    primeNumbers.erase(primeNumbers.begin(), primeNumbers.begin() + minBorder);

    int finalCount = primeNumbers.size();

    *count = finalCount;
    return primeNumbers;
}

void findMinMax(int number, Result &min, Result &max) {
    int dozens = 10;

    while (number / dozens != 0) {
        int n = (number / dozens) * (number % dozens);

        if (n < min.value) {
            Result _min = {n, number, dozens};
            min = _min;
        }
        if (n > max.value) {
            Result _max = {n, number, dozens};
            max = _max;
        }

        dozens *= 10;
    }
}

void solveSequential(int number1, int number2, Result &min, Result &max) {
    int primesCount;
    vector<int> primes =
        eratosphenesSieveSequential(number1, number2, &primesCount);

    Result _min = {INT_MAX, -1, -1}, _max = {INT_MIN, -1, -1};

    for (unsigned int i = 0; i < primesCount; i++) {
        findMinMax(primes[i], _min, _max);
    }

    min = _min;
    max = _max;
}

__device__ Result *d_min;
__device__ Result *d_max;
__device__ int d_primesCount;

__global__ void eratosphenesSieveParallel(int *primes, int primesStartCount,
                                          int min, int max,
                                          int elemsPerThread) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int id = blockId * THREADS_PER_BLOCK + threadIdx.x;

    if (id == 0) atomicExch(&d_primesCount, primesStartCount);

    for (int i = id * elemsPerThread; i < id * elemsPerThread + elemsPerThread;
         i++) {
        bool isPrime = true;
        int number = primes[primesStartCount - 1] + 2 + 2 * i;

        if (number > max || number < min) continue;

        for (int i = 1; i < primesStartCount; i++) {
            if (number % primes[i] == 0) {
                isPrime = false;
                break;
            }
        }

        if (isPrime) {
            int writeIndex = atomicAdd(&d_primesCount, 1);
            primes[writeIndex] = number;
        }
    }
}

__global__ void findMinMaxCuda(int *primes, int primesCount, int min, int max,
                               int elementsPerThread) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int id = blockId * THREADS_PER_BLOCK + threadIdx.x;
    if (id == 0) {
        *d_min = {INT_MAX, -1, -1};
        *d_max = {INT_MIN, -1, -1};
    }

    Result myMin = {INT_MAX, -1, -1}, myMax = {INT_MIN, -1, -1};

    for (int i = id * elementsPerThread;
         i < id * elementsPerThread + elementsPerThread && i < primesCount;
         i++) {
        int number = primes[i];
        if (number < min || number > max) continue;

        int dozens = 10;

        while (number / dozens != 0) {
            int n = (number / dozens) * (number % dozens);

            if (n < myMin.value) {
                Result _myMin = {n, number, dozens};
                myMin = _myMin;
            }
            if (n > myMax.value) {
                Result _myMax = {n, number, dozens};
                myMax = _myMax;
            }

            dozens *= 10;
        }
    }

    if (myMin.value != INT_MAX) atomicMin(&(d_min->value), myMin.value);
    if (myMax.value != INT_MIN) atomicMax(&(d_max->value), myMax.value);

    __syncthreads();

    if (d_min->value == myMin.value) *d_min = myMin;
    if (d_max->value == myMax.value) *d_max = myMax;
}

void getNumberOfBlocksAndThreads(int elemsCount, dim3 *blocks, dim3 *threads) {
    dim3 _blocks, _threads;
    _blocks.x = sqrt((elemsCount - 1) / THREADS_PER_BLOCK + 1) + 1;
    _blocks.y = _blocks.x;

    _threads.x =
        (elemsCount < THREADS_PER_BLOCK ? elemsCount : THREADS_PER_BLOCK);

    *blocks = _blocks;
    *threads = _threads;
}

void solveParallel(int number1, int number2, Result &h_min, Result &h_max) {
    int primesCount;
    int *primes =
        &eratosphenesSieveSequential(0, sqrt(number2), &primesCount)[0];

    int theoreticalPrimesCount =
        ((number2 - number1) / log(sqrt(number2))) * 1.2;
    if (theoreticalPrimesCount < 500) theoreticalPrimesCount = 500;

    int *d_primes;
    cudaMalloc(&d_primes, (primesCount + theoreticalPrimesCount) * sizeof(int));
    cudaMemcpy(d_primes, primes, primesCount * sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 blocks, threads;
    int elems = number2 / 2 + 1;

    int elemsPerThread = (elems > 1000 ? 1000 : 1);
    getNumberOfBlocksAndThreads(elems / elemsPerThread + 1, &blocks, &threads);

    eratosphenesSieveParallel<<<blocks, threads>>>(
        d_primes, primesCount, number1, number2, elemsPerThread);
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&primesCount, d_primesCount, sizeof(int));

    Result *_d_min, *_d_max;
    cudaMalloc(&_d_min, sizeof(Result));
    cudaMemcpyToSymbol(d_min, &_d_min, sizeof(Result *));

    cudaMalloc(&_d_max, sizeof(Result));
    cudaMemcpyToSymbol(d_max, &_d_max, sizeof(Result *));

    int elementsPerThread = 10;
    getNumberOfBlocksAndThreads(primesCount / elementsPerThread, &blocks,
                                &threads);

    findMinMaxCuda<<<blocks, threads>>>(d_primes, primesCount, number1, number2,
                                        elementsPerThread);
    cudaDeviceSynchronize();

    Result *_h_min = (Result *)malloc(sizeof(Result));
    Result *_h_max = (Result *)malloc(sizeof(Result));

    cudaMemcpy(_h_min, _d_min, sizeof(Result), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_max, _d_max, sizeof(Result), cudaMemcpyDeviceToHost);

    h_min = *_h_min;
    h_max = *_h_max;

    cudaFree(d_primes);
    cudaFree(_d_min);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Args: number#1, number#2\n");
        return -1;
    }

    int number1 = strtol(argv[1], NULL, 10);
    int number2 = strtol(argv[2], NULL, 10);

    if (number2 < number1) {
        printf("Condition: number1 < number2\n");
        return -1;
    }
	

    auto execute = [number1, number2](
                       void (*func)(int, int, Result &, Result &), Result &min,
                       Result &max) {
        struct timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start);

        func(number1, number2, min, max);

        clock_gettime(CLOCK_REALTIME, &end);

        printf("min value: %d = %d * %d (from %d)\n", min.value,
               min.number / min.dozens, min.number % min.dozens, min.number);
        printf("max value: %d = %d * %d (from %d)\n", max.value,
               max.number / max.dozens, max.number % max.dozens, max.number);
        printf("time: %f c\n\n",
               (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) * 1.0 / 1000000000);
    };

    Result min1, max1, min2, max2;

    printf("Parallel calculation:\n");
    execute(solveParallel, min1, max1);

    printf("Not parallel calculation:\n");
    execute(solveNotParallel, min2, max2);

    std::cout << "The results are "
              << (min1 == min2 && max1 == max2 ? "" : "not ") << "equal\n";
    return 0;
}
