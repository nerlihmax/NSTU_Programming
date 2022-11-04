__kernel void calculateLengths(int firstNumber, int lastNumber,
                               __global int *digitsWithLength,
                               unsigned int elemsPerThread) {
    int id = get_global_id(0);

    for (int i = id * elemsPerThread; i < id * elemsPerThread + elemsPerThread;
         i++) {
        unsigned int n = firstNumber + i;

        if (n > lastNumber) break;

        int length = 1;

        while (n != 1) {
            n = (n % 2 == 0 ? n / 2 : 3 * n + 1);
            length++;
        }

        atomic_add(&digitsWithLength[length], 1);
    }
}
