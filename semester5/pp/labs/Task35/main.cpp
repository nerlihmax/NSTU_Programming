#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ctime>
#include <fstream>
#include <iostream>

#define RED "\e[31m"
#define GREEN "\e[32m"
#define PLAIN "\e[0m"

#define NANOSEC_TO_SEC 1000000000.0

using namespace std;

const char PROGRAM_FILE[] = "kernel.cl";
const char KERNEL_FUNC[] = "calculateLengths";
const char OUTPUT_FILE[] = "result.txt";
const int LENGTHS_ARRAY_SIZE = 5000;

struct OpenCLProgramBuilder {
    struct Info {
        cl_program program = NULL;
        cl_context context = NULL;
        cl_command_queue queue = NULL;
        cl_device_id device_id;
        cl_platform_id cpPlatform;

        cl_int error;
    };

    void build(const char **sourceCode, Info &info) {
        info.error = clGetPlatformIDs(1, &info.cpPlatform, NULL);
        info.error = clGetDeviceIDs(info.cpPlatform, CL_DEVICE_TYPE_GPU, 1,
                                    &info.device_id, NULL);
        info.context =
            clCreateContext(0, 1, &info.device_id, NULL, NULL, &info.error);
        info.queue =
            clCreateCommandQueue(info.context, info.device_id, 0, &info.error);
        info.program = clCreateProgramWithSource(info.context, 1, sourceCode,
                                                 NULL, &info.error);
        info.error = clBuildProgram(info.program, 0, NULL, NULL, NULL, NULL);
    }

    void Dispose(Info &info) {
        if (info.program != NULL) clReleaseProgram(info.program);
        if (info.queue != NULL) clReleaseCommandQueue(info.queue);
        if (info.context != NULL) clReleaseContext(info.context);
    }
};

void readFileToStr(const char *name, char **buffer) {
    std::ifstream fin(name);
    std::string content((std::istreambuf_iterator<char>(fin)),
                        (std::istreambuf_iterator<char>()));

    *buffer = (char *)calloc(content.length() + 1, sizeof(char));
    strcpy(*buffer, content.data());
}

void calculateWithOpenCL(int firstNumber, int lastNumber,
                         int *digitsWithLength) {
    char *sourceCode;
    readFileToStr(PROGRAM_FILE, &sourceCode);

    OpenCLProgramBuilder::Info info;
    OpenCLProgramBuilder program;
    program.build((const char **)&sourceCode, info);

    if (info.error != CL_SUCCESS) {
        printf("Build program error %d\n", info.error);
        return;
    };

    int elemsPerThread = 100;

    size_t localSize = 64;
    size_t globalSize = ceil((float)(lastNumber - firstNumber + 1) /
                             elemsPerThread / localSize) *
                        localSize;

    cl_int error;
    cl_kernel kernel = clCreateKernel(info.program, KERNEL_FUNC, &error);

    cl_mem d_digitsWithLength = clCreateBuffer(
        info.context, CL_MEM_USE_HOST_PTR, LENGTHS_ARRAY_SIZE * sizeof(int),
        digitsWithLength, NULL);

    error |= clSetKernelArg(kernel, 0, sizeof(int), &firstNumber);
    error |= clSetKernelArg(kernel, 1, sizeof(int), &lastNumber);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_digitsWithLength);
    error |= clSetKernelArg(kernel, 3, sizeof(int), &elemsPerThread);

    clEnqueueNDRangeKernel(info.queue, kernel, 1, NULL, &globalSize, &localSize,
                           0, NULL, NULL);

    clFinish(info.queue);

    if (error != CL_SUCCESS) {
        printf("Execute kernel error!\n", error);
        return;
    };

    clEnqueueReadBuffer(info.queue, d_digitsWithLength, CL_TRUE, 0,
                        LENGTHS_ARRAY_SIZE * sizeof(int), digitsWithLength, 0,
                        NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseMemObject(d_digitsWithLength);
    program.Dispose(info);
    free(sourceCode);
}

void calculateWithoutOpenCL(int firstNumber, int lastNumber,
                            int *digitsWithLength) {
    for (int i = firstNumber; i <= lastNumber; i++) {
        unsigned int n = i;
        int length = 1;

        while (n != 1) {
            n = (n % 2 == 0 ? n / 2 : 3 * n + 1);
            length++;
        }

        digitsWithLength[length]++;
    }
}

void writeResultsToFile(int *resWithOpenCL, int *resWithoutOpenCL,
                        float timeWithOpenCL, float timeWithoutOpenCL) {
    std::ofstream fout(OUTPUT_FILE);

    bool resultsEqual = true;
    for (int i = 0; i < LENGTHS_ARRAY_SIZE; i++) {
        if (resWithoutOpenCL[i] != resWithOpenCL[i]) {
            resultsEqual = false;
            break;
        }
    }
    cout << "=============" << endl;
    cout << "The results are " << (resultsEqual ? GREEN : RED)
         << (resultsEqual ? "equal" : "different") << PLAIN << endl;
    cout << "Parallel | Sequential calculation time: \n"
         << (timeWithOpenCL < timeWithoutOpenCL ? GREEN : RED) << timeWithOpenCL
         << PLAIN << " c | "
         << (timeWithOpenCL > timeWithoutOpenCL ? GREEN : RED)
         << timeWithoutOpenCL << PLAIN << " c\n";

    cout << "\n ============= \n\n";
    printf("Writing results to file...\n");
    fout << "Length: amount of digits #1 | amount of digits #2\n";

    for (int i = 0; i < LENGTHS_ARRAY_SIZE; i++) {
        if (resWithOpenCL[i] != 0 && resWithoutOpenCL[i] != 0)
            fout << i << " : " << resWithOpenCL[i] << " | "
                 << resWithoutOpenCL[i] << "\n";
    }

    fout << "The unspecified amounts of numbers for the length are 0\n";

    fout.close();
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << RED << "Required args: number#1, number#2" << PLAIN << endl;
        return -1;
    }

    long firstNumber = strtol(argv[1], NULL, 10);
    long secondNumber = strtol(argv[2], NULL, 10);

    if (firstNumber > secondNumber || firstNumber <= 1 || secondNumber <= 1) {
        cout << RED << "Condition: number#2 > number#1 > 1" << PLAIN << endl;
        return -1;
    }

    int *digitsWithLength1 = (int *)calloc(LENGTHS_ARRAY_SIZE, sizeof(int));
    int *digitsWithLength2 = (int *)calloc(LENGTHS_ARRAY_SIZE, sizeof(int));

    auto execute = [firstNumber, secondNumber](void (*func)(int, int, int *),
                                               int *digitsWithLength) {
        struct timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start);
        func(firstNumber, secondNumber, digitsWithLength);
        clock_gettime(CLOCK_REALTIME, &end);
        return (end.tv_sec - start.tv_sec) +
               (end.tv_nsec - start.tv_nsec) / NANOSEC_TO_SEC;
    };

    printf("Calculation parallel\n");
    float time1 = execute(calculateWithOpenCL, digitsWithLength1);

    printf("Calculation sequentially\n");
    float time2 = execute(calculateWithoutOpenCL, digitsWithLength2);

    writeResultsToFile(digitsWithLength1, digitsWithLength2, time1, time2);

    free(digitsWithLength1);
    free(digitsWithLength2);
    return 0;
}
