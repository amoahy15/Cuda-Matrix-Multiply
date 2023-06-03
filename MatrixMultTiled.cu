#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define TILE_WIDTH 1
__global__ void generateRandomNumbers(curandState_t *states, float *numbersA, float *numbersB, int n);
__global__ void matrixMult(float *M, float *N, float *P, int Width);
int main(int argc, char *argv[])
{
    // Check for the appropriate number of command line arguments
    if (argc != 2)
    {
        printf("Error: You must provide one command line argument corresponding to the row/column length of the matrices to multiply.\n");
        return 1;
    }
    // Parse the command line argument and check if it is a positive number
    int matrixSize = atoi(argv[1]);
    

    // Round the matrix size to the nearest multiple of TILE_WIDTH
    matrixSize = TILE_WIDTH * ((matrixSize + TILE_WIDTH - 1) / TILE_WIDTH);

    if (matrixSize <= 0)
    {
        printf("Error: The command line argument must be a positive integer.\n");
        return 1;
    }

    // Allocate host memory for the matrices
    float *A = (float *)malloc(matrixSize * matrixSize * sizeof(float));
    float *B = (float *)malloc(matrixSize * matrixSize * sizeof(float));
    float *C = (float *)malloc(matrixSize * matrixSize * sizeof(float));

    // Allocate device memory for the matrices
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, matrixSize * matrixSize * sizeof(float));
    cudaMalloc(&dev_b, matrixSize * matrixSize * sizeof(float));
    cudaMalloc(&dev_c, matrixSize * matrixSize * sizeof(float));

    // Allocate memory for the random number generator states
    curandState_t *d_states;
    cudaMalloc(&d_states, matrixSize * matrixSize * sizeof(curandState_t));

    // Generate random numbers for matrices M and N on the device
    generateRandomNumbers<<<ceil(matrixSize * matrixSize / 256.0), 256>>>(d_states, dev_a, dev_b, matrixSize * matrixSize);

    // Copy matrices M and N from device to host
    cudaMemcpy(A, dev_a, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, dev_b, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Create events for timing the matrix multiplication kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the matrix multiplication kernel and time its execution
    dim3 dimGrid(matrixSize / TILE_WIDTH, matrixSize / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    cudaEventRecord(start);
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, matrixSize);
    cudaEventRecord(stop);

    // Copy matrix P from device to host
    cudaMemcpy(C, dev_c, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the execution time of the matrix multiplication kernel
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // file output
    char filename[100];
    sprintf(filename, "Matrix_Calulations_of_Size_%s.dat", argv[1]);
    FILE *outfile;
    outfile = fopen(filename, "a");
    fprintf(outfile, "Matrix Multiply Tiled Results for Matrix size %s  \n", argv[1]);
    fprintf(outfile, "Execution time: %f ms\n", elapsedTime);
    ;
    fprintf(outfile, "\nMatrix A:\n");
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            fprintf(outfile, "%f ", A[i * matrixSize + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\nMatrix B:\n");
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            fprintf(outfile, "%f ", B[i * matrixSize + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\nMatrix C:\n");
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            fprintf(outfile, "%f ", C[i * matrixSize + j]);
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "\n");
    fclose(outfile);

    // Product.out file
    FILE *product;
    product = fopen("Product.out", "a");
    fprintf(product, "Tiled Matrix Size %d\n",matrixSize);
    fprintf(product, "Execution time: %f ms\n", elapsedTime);
    ;
    fprintf(product, "\nMatrix Output:\n");
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            fprintf(product, "%f ", C[i * matrixSize + j]);
        }
        fprintf(product, "\n");
    }
    fprintf(product, "\n");
    fprintf(product, "\n");
    fclose(product);
    // file output for graph
    FILE *csv_file;
    char csv_filename[100];
    sprintf(csv_filename, "Tiled.csv", argv[1]);
    csv_file = fopen(csv_filename, "a");
    fprintf(csv_file, "Input Size,Elapsed Time (ms)\n");
    fprintf(csv_file, "%d,%f\n", matrixSize, elapsedTime);
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
// CUDA kernel for matrix multiplication using shared memory and tiling
__global__ void matrixMult(float *M, float *N, float *P, int Width)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < Width / TILE_WIDTH; ++p)
    {
        // Collaborative loading of M and N tiles into shared memory
        ds_M[ty][tx] = M[Row * Width + p * TILE_WIDTH + tx];
        ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += ds_M[ty][i] * ds_N[i][tx];

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

// CUDA kernel for generating random numbers for matrix A and B
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
