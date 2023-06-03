#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#define BLOCK_SIZE 16
__global__ void generateRandomNumbers(curandState_t *states, float *numbersA, float *numbersB, int n);
__global__ void matrixMult(float *A, float *B, float *C, int size);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s size\n", argv[0]);
        exit(1);
    }

    int size = atoi(argv[1]);
    float input_size = size * size * sizeof(float);
    if (size <= 0)
    {
        printf("Invalid matrix size: %d\n", size);
        exit(1);
    }

    // Allocate memory for matrices A, B, and C on the host
    float *A = (float *)malloc(input_size);
    float *B = (float *)malloc(input_size);
    float *C = (float *)malloc(input_size);

    // Allocate memory for matrices A, B, and C on the device
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, input_size);
    cudaMalloc(&dev_b, input_size);
    cudaMalloc(&dev_c, input_size);

    // Allocate memory for the curandState_t object on the device
    curandState_t *d_states;
    cudaMalloc(&d_states, size * size * sizeof(curandState_t));

    // Initialize the curandState_t object on the device
    generateRandomNumbers<<<(size * size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, dev_a, dev_b, size * size);

    // Copy matrices A and B from device to host for verification
    cudaMemcpy(A, dev_a, input_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, dev_b, input_size, cudaMemcpyDeviceToHost);

    // Define the grid and block dimensions for the MatrixMultKernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // Create CUDA events to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call the MatrixMultKernel on the device
    cudaEventRecord(start);
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, size);
    cudaEventRecord(stop);

    // Copy matrix C from device to host
    cudaMemcpy(C, dev_c, input_size, cudaMemcpyDeviceToHost);

    // Calculate the elapsed time in seconds
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // file output for calculation
    char filename[100];
    sprintf(filename, "Matrix_Calulations_of_Size_%s.dat", argv[1]);
    FILE *outfile;
    outfile = fopen(filename, "a");
    sprintf(filename, "Matrix_Calulations_of_Size_%s.dat", argv[1]);
    fprintf(outfile, "Execution time: %f ms\n", elapsedTime);
    ;
    fprintf(outfile, "\nMatrix A:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(outfile, "%f ", A[i * size + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\nMatrix B:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(outfile, "%f ", B[i * size + j]);
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
    fprintf(product, "Tiled Matrix Size %d\n",size);
    fprintf(product, "Execution time: %f ms\n", elapsedTime);
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
    sprintf(csv_filename, "Naive.csv", argv[1]);
    csv_file = fopen(csv_filename, "a");
    fprintf(csv_file, "Input Size,Elapsed Time (ms)\n");
    fprintf(csv_file, "%d,%f\n", size, elapsedTime);
    fclose(csv_file);

    // Free memory on the device and host
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(d_states);
    free(A);
    free(B);
    free(C);

    return 0;
}
// CUDA kernel to multiply two matrices in global memory
__global__ void matrixMult(float *A, float *B, float *C, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    if (row < size && col < size)
    {
        float sum = 0.0f;
        for (k = 0; k < size; k++)
        {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// CUDA kernel to generate random numbers in global memory
__global__ void generateRandomNumbers(curandState_t *states, float *numbersA, float *numbersB, int n)
{
    // Get the thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize the curandState_t object for this thread
    curand_init(100, idx, 0, &states[idx]);

    // Generate n random numbers from 0-100 for matrix A and B
    // and store them in the numbersA and B array
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        numbersA[i] = 100.0f * curand_uniform(&states[idx]);
        numbersB[i] = 100.0f * curand_uniform(&states[idx]);
    }
}