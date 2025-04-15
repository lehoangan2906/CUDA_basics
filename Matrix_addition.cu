#include <stdio.h>


// kernel definition

/* Since the matrices are allocated memory on the GPU with cudaMalloc, 
it gives a 1D array, therefore we need to use pointers to 1D arrays to 
match this linear memory layout.
*/

__global__ void AddTwoMatrices(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && column < N){
        int idx = row * N + column;
        C[idx] = A[idx] + B[idx];   // Use this way of idexing since GPU memory is linear
    }
}


int main(){
    int N = 512; // operate on 512x512 matrices
    dim3 threads_per_block(16, 16);   // 16 x 16 = 256 threads per block
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);    // 2D grid to cover NxN


    // Dynamically allocate host arrays
    float *A = (float *)malloc(N * N * sizeof(float));   // allocate memory on the heap, which can handle large array size (up to system's available RAM). Also make it easy to copy the flatten 1D array to the GPU with cudaMemcpy.
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));


    // Initialize arrays A and B
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            int idx = i * N + j;
            A[idx] = i + 10;
            B[idx] = sinf(i * 0.1f);
        }
    }


    // Create pointers for these arrays on GPU
    float *d_A, *d_B, *d_C;


    // Allocate memory on the GPU for these arrays
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));


    // Copy matrices A and B form host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);


    // Launch kernel
    AddTwoMatrices<<<gridDim, threads_per_block>>>(d_A, d_B, d_C, N);


    // Check for CUDA error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    // Wait until all CUDA threads are executed
    cudaDeviceSynchronize();


    // Copy the resulting vector from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);


    printf("First 10 elements of C:\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++){        
            printf("C[%d][%d] = %f\n", i, j, C[i * N + j]);
        }
    }


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
