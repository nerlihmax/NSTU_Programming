#include <iostream>
using namespace std;

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int c;
    int *dev_c;
    cudaMallocManaged(&dev_c, sizeof(int));
    add<<<1, 1>>>(1, 2, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cout << c << endl;
    cudaFree(dev_c);
    return 0;
}
