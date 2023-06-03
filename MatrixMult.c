#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiply(float *A, float *B, float *C, int size)
{
    // serial matrix multiplication
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < size; k++)
            {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s [matrix size]\n", argv[0]);
        return 1;
    }
    int size = atoi(argv[1]);
    if (size <= 0)
    {
        printf("Error: matrix size must be a positive integer.\n");
        return 1;
    }
    float *A = (float *)malloc(size * size * sizeof(float));
    float *B = (float *)malloc(size * size * sizeof(float));
    float *C = (float *)malloc(size * size * sizeof(float));

    // initialize matrices with random numbers

    srand(100);
    for (int i = 0; i < size * size; i++)
    {
        A[i] = (float)(rand()) / ((float)RAND_MAX / 100);
        B[i] = (float)(rand()) / ((float)RAND_MAX / 100);
    }

    // matrix multiplication
    clock_t start_time = clock();
    matrix_multiply(A, B, C, size);
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double total_time_ms = total_time * 1000.0;

    // output results to file
    char filename[100];
    sprintf(filename, "Matrix_Calulations_of_Size_%s.dat", argv[1]);
    FILE *outfile;
    outfile = fopen(filename, "a");
    fprintf(outfile, "Matrix Multiply CPU Results for Matrix Size %s \n", argv[1]);
    fprintf(outfile, "Execution time: %.6f ms\n", total_time);
    fprintf(outfile, "Matrix A:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            A[i * size + j] = (float)(rand() % 101);
            fprintf(outfile, "%.6f\t", A[i * size + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

    fprintf(outfile, "Matrix B:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            B[i * size + j] = (float)(rand() % 101);
            fprintf(outfile, "%.6f\t", B[i * size + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\nMatrix C:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(outfile, "%f ", C[i * size + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "\n");
    fclose(outfile);

    // Product.out file
    FILE *product;
    product = fopen("Product.out", "a");
    fprintf(product, "CPU Matrix Size %d\n", size);
    fprintf(product, "Execution time: %f ms\n", total_time);
    ;
    fprintf(product, "\nMatrix Output:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(product, "%f ", C[i * size + j]);
        }
        fprintf(product, "\n");
    }
    fprintf(product, "\n");
    fprintf(product, "\n");
    fclose(product);
    // file output for graph
    FILE *csv_file;
    char csv_filename[100];
    sprintf(csv_filename, "CPU.csv", argv[1]);
    csv_file = fopen(csv_filename, "a");
    fprintf(csv_file, "Input Size,Elapsed Time (ms)\n");
    fprintf(csv_file, "%d,%f\n", size, total_time);
    fclose(csv_file);

    free(A);
    free(B);
    free(C);
    return 0;
}
