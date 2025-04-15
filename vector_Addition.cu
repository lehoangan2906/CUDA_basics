#include <stdio.h>

// Host: the CPU
// Device: The GPU
// Kernel: a function that runs on the GPU and is called from the CPU


// __global__ is use to define a kernel.
__global__ void AddTwoVectors(float A[], float B[], float C[]) {
    int i = threadIdx.x;    // Each thread is given a unique threadID
    C[i] = A[i] + B[i];
}

int main(){
    int N = 1000;
    float A[N], B[N], C[N];

    // Initialize vectors A and B
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 3;
    }


    float *d_A, *d_B, *d_C;     // Device pointers for vectors A, B, and C

    /*
    Note that in CUDA programming, you can't directly use host arrays (like A, B, and C) with kernel launches (<<<number_of_block, thread_per_block >>>).
    
    CUDA kernel operate on device memory, so you need to pass device pointers (d_A, d_B, and d_C) to the kernel for it to operate on.

    Beyond that, we need to allocate memory on the device by using cudaMalloc, and copy data between host and device using cudaMemcpy.
    */

    // Allocate memory on the device for vectors A, B, and C
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));


    // Copy vectors A and B from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);


    // The number of CUDA threads that execute the above kernel can be spefified using <<< >>> notation.
    // Kernel invocation with 1 block and N threads
    AddTwoVectors<<<1, N>>>(d_A, d_B, d_C);

    
    /*

    cudaDeviceSynchronize() is used to synchronize the device and the host thread.

    When this function is called, the host thread will wait until all previously issued CUDA commands on the device are completed before continuing execution.
    */


    // Check for errors in kernel launch (e.g. invalid configuration) 
    cudaError_t error = cudaGetLastError();
    

    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    // Wait until all CUDA threads are executed
    cudaDeviceSynchronize();


    // copy vector C from device to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
