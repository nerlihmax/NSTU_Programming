#include <math.h>
#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define NANOSEC_IN_SEC 1000000000

const char OUTPUT_FILE[] = "result.txt";

struct ArrayPart {
    size_t startId;
    size_t size;
    ArrayPart(){};
    ArrayPart(size_t startId, size_t size) : startId(startId), size(size){};
};

ArrayPart merge(vector<int>& array, ArrayPart sortedPart1,
                ArrayPart sortedPart2) {
    ArrayPart mergedPart = {sortedPart1.startId,
                            sortedPart1.size + sortedPart2.size};

    int* arrayPart = &array[sortedPart1.startId];
    vector<int> sortedArray;
    sortedArray.reserve(mergedPart.size);

    auto shrinkPart = [](ArrayPart& part) {
        part.startId++;
        part.size--;
    };

    while (sortedPart1.size > 0 || sortedPart2.size > 0) {
        if (sortedPart1.size == 0) {
            memcpy(arrayPart, &sortedArray[0],
                   sortedArray.size() * sizeof(int));
            break;
        } else if (sortedPart2.size == 0) {  // move remains of part1 to the
                                             // end, then memcpy sortedArray
            memcpy(arrayPart + sortedArray.size(), &array[sortedPart1.startId],
                   sortedPart1.size * sizeof(int));
            memcpy(arrayPart, &sortedArray[0],
                   sortedArray.size() * sizeof(int));
            break;
        }

        ArrayPart* selectedPart =
            (array[sortedPart1.startId] > array[sortedPart2.startId]
                 ? &sortedPart2
                 : &sortedPart1);

        sortedArray.push_back(array[selectedPart->startId]);
        shrinkPart(*selectedPart);
    }

    return mergedPart;
}

void insertionSort(int* a, int n, int step) {
    for (int i = step; i < n; i += step) {
        int key = a[i];
        int j = i - step;
        while (j >= 0 && a[j] > key) {
            a[j + step] = a[j];
            j -= step;
        }
        a[j + step] = key;
    }
}

void shellSortPart(vector<int>& array, ArrayPart part) {
    for (int step = part.size / 2; step > 0; step /= 2) {
        for (int i = 0; i < step; i++) {
            int start = i + part.startId;
            insertionSort(&(array[start]), part.size - i, step);
        }
    }
}

ArrayPart calculateArrayPartForProcess(size_t arraySize, int processCount,
                                       int processRank) {
    size_t elemsPerThread = arraySize / processCount + 1;
    ArrayPart part = {(size_t)processRank * elemsPerThread, elemsPerThread};

    if (part.startId + part.size > arraySize)
        part.size = arraySize - part.startId;

    return part;
}

void shellSortParallel(vector<int>& array, int processCount, int processRank) {
    size_t partSize =
        (processRank == 0 ? calculateArrayPartForProcess(
                                array.size(), processCount, processRank)
                                .size
                          : array.size());
    ArrayPart part = {0, partSize};

    shellSortPart(array, part);

    vector<int> partSizes, offsets;

    for (int i = 0; i < processCount; i++) {
        partSizes.push_back(
            calculateArrayPartForProcess(array.size(), processCount, i).size);
        offsets.push_back(i * partSizes[0]);
    }

    vector<int> _array(array.size());
    MPI_Gatherv(&array.front(), (int)partSize, MPI_INT, &_array.front(),
                &partSizes.front(), &offsets.front(), MPI_INT, 0,
                MPI_COMM_WORLD);

    if (processRank != 0) return;

    for (int i = 1; i < processCount; i++) {
        ArrayPart processPart =
            calculateArrayPartForProcess(_array.size(), processCount, i);
        part = merge(_array, part, processPart);
    }

    array = _array;
}

void writeResultsToFile(vector<int>& array, float time, vector<int>& arrayMPI,
                        float timeMPI) {
    std::ofstream fout(OUTPUT_FILE);
    if (!fout) {
        printf("fail to open result.txt\n");
        return;
    }

    fout << "The results are " << (array == arrayMPI ? "equal" : "different")
         << "\n\n";

    fout << "Parallel calculation time:\t" << timeMPI << "c \n";

    fout << "\n\nSequential calculation time:\t" << time << "c \n";

	fout << "\nParallel result array" << endl;
    for (auto elem : arrayMPI) fout << elem << " ";
	fout << "\n\nSequential result array" << endl;
    for (auto elem : array) fout << elem << " ";
    fout << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Required argument: size\n");
        return -1;
    }

    const long size = strtol(argv[1], NULL, 10);

    if (size <= 0) {
        printf("Wrong size\n");
        return -1;
    }

    MPI_Init(NULL, NULL);
    int processCount, processRank;

    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    vector<int> array1, array2;

    if (processRank == 0) {
        array1 = vector<int>(size);
        srand(time(0));
        int min = -10000, max = 10000;
        std::generate(array1.begin(), array1.end(),
                      [min, max]() { return rand() % (max - min + 1) + min; });

        array2 = vector<int>(array1);

        for (int i = 1; i < processCount; i++) {
            ArrayPart processPart =
                calculateArrayPartForProcess(array2.size(), processCount, i);
            int size = processPart.size;

            MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&array2[processPart.startId], size, MPI_INT, i, 0,
                     MPI_COMM_WORLD);
        }
    } else {
        int partSize;
        MPI_Recv(&partSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        array2.resize(partSize);
        MPI_Recv(&array2.front(), partSize, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    struct timespec start, end;
    float time1 = 0, time2 = 0;

    // sequential
    if (processRank == 0) {
        clock_gettime(CLOCK_REALTIME, &start);

        shellSortPart(array1, {0, (size_t)size});

        clock_gettime(CLOCK_REALTIME, &end);
        time1 = (end.tv_sec - start.tv_sec) +
                (end.tv_nsec - start.tv_nsec) * 1.0 / NANOSEC_IN_SEC;
    }

    // parallel
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_REALTIME, &start);

    shellSortParallel(array2, processCount, processRank);

    clock_gettime(CLOCK_REALTIME, &end);
    time2 = (end.tv_sec - start.tv_sec) +
            (end.tv_nsec - start.tv_nsec) * 1.0 / NANOSEC_IN_SEC;

    if (processRank == 0) {
        writeResultsToFile(array1, time1, array2, time2);
    }

    MPI_Finalize();
    return 0;
}
