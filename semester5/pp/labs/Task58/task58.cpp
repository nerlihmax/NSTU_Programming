#include <array>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <vector>

using namespace std;

#define RED "\e[31m"
#define GREEN "\e[32m"
#define NANOSEC_TO_SEC 1000000000.0

vector<int> eratosthenesSieve(long maxNumber) {
  if (maxNumber < 2)
    return vector<int>{};

  vector<int> primeNumbers{};

  primeNumbers.push_back(2);

  for (long n = 3; n < maxNumber; n += 2) {
    bool isPrime = true;

    for (int i = 1; i < primeNumbers.size(); i++) {
      if (n % primeNumbers[i] == 0) {
        isPrime = false;
        break;
      }
      if (pow(primeNumbers[i], 2) > n)
        break;
    }

    if (!isPrime)
      continue;

    primeNumbers.push_back(n);
  }

  primeNumbers.shrink_to_fit();
  return primeNumbers;
}

struct Result {
  int number;
  array<int, 4> primeNumbers;
  bool operator==(Result res) { return number == res.number; }
};

Result factorize(long number, bool isParallel) {
  vector<int> primes = eratosthenesSieve(number < 287 ? 8 : sqrt(number));
  int primesCount = primes.size();

  Result result = {INT_MAX, {-1, -1, -1, -1}};

  auto getSum = [&primes](int i, int j, int k = -1, int s = -1) {
    return pow(primes[i], 2) + pow(primes[j], 3) +
           (k < 0 ? 0 : pow(primes[k], 4)) + (s < 0 ? 0 : pow(primes[s], 5));
  };

#pragma omp parallel for if (isParallel) shared(number, primesCount, primes)
  for (int i = 0; i < primesCount; i++) {
    Result threadResult = {INT_MAX, {-1, -1, -1, -1}};

    for (int j = 0; j < primesCount; j++) {
      int preSum = getSum(i, j);
      if (preSum > number && preSum > threadResult.number)
        break;

      for (int k = 0; k < primesCount; k++) {
        preSum = getSum(i, j, k);
        if (preSum > number && preSum > threadResult.number)
          break;

        for (int s = 0; s < primesCount; s++) {
          if (i == j || i == k || i == s || j == k || j == s || k == s)
            continue;

          int sum = getSum(i, j, k, s);

          if (sum <= number)
            continue;

          if (sum < threadResult.number)
            threadResult = {sum, {primes[i], primes[j], primes[k], primes[s]}};
          else
            break;
        }
      }
    }

#pragma omp critical
    if (threadResult.number < result.number) {
      result = threadResult;
    }
  }

  primes.clear();
  return result;
}

Result factorizeWithBenchmark(long number, bool isParallel) {
  timespec start, end;

  // measuring computing time
  clock_gettime(CLOCK_REALTIME, &start);
  Result res = factorize(number, isParallel);
  clock_gettime(CLOCK_REALTIME, &end);

  printf("\e[32m%d\e[0m = \e[33m %d^2 \e[0m + \e[34m %d^3 \e[0m + \e[35m %d^4 \e[0m + \e[36m %d^5 \e[0m\n", res.number, res.primeNumbers[0],
         res.primeNumbers[1], res.primeNumbers[2], res.primeNumbers[3]);
  printf("Computing time: %.6f seconds\n",
         (end.tv_sec - start.tv_sec) +
             (end.tv_nsec - start.tv_nsec) / NANOSEC_TO_SEC);
  return res;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Missing input number, exit.\n");
    return -1;
  }

  long number = strtol(argv[1], NULL, 10);

  printf("\nSequential:\n");
  Result res1 = factorizeWithBenchmark(number, false);

  printf("\nParallel:\n");
  Result res2 = factorizeWithBenchmark(number, true);

  cout << "\nResults are " << (res1 == res2 ? GREEN : RED) << (res1 == res2 ? "equal" : "different") << "\e[0m" << endl;
  return 0;
}