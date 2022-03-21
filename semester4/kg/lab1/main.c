#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main()
{
    float det;                     // determinant
    clock_t firstTime, secondTime; // time before and after calculation
    det = 1;

    for (int N = 100; N <= 5000; N += 100)
    {
        float **matrix = calloc(N, sizeof(float *));
        for (int i = 0; i < N; i++)
            matrix[i] = calloc(N, sizeof(float));

        // random matrix generation
        srand(time(0));
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                matrix[i][j] = 1.0f + rand() % 9;
            }
        }

    
        // printf("Calculating determinant \n");
        firstTime = clock();        // get initial time before calculation
        for (int i = 0; i < N; i++) // converting matrix to triangle format
        {
            for (int j = i + 1; j < N; j++)
            {
                float fTmp = matrix[j][i] / matrix[i][i];
                for (int k = i; k < N; k++)
                {
                    matrix[j][k] -= matrix[i][k] * fTmp; // column to nulls
                }
            }
            det *= matrix[i][i]; // determinant
        }
        secondTime = clock();
        secondTime = secondTime - firstTime; // result time
        // printf("Determinant: %14.3f", det);
        printf("%d ", N);
        printf("%f",
               ((float)secondTime) / CLOCKS_PER_SEC);
        printf("\n");

        for (int i = 0; i < N; i++)
            free(matrix[i]);
        free(matrix);
    }

    getchar();
    return 0;
}