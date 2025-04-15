# Overview

This repository contains a Jupyter Notebook, `cuda_basics.ipynb`, designed to serve as an introductory tutorial on CUDA programming for GPU-accelerated computing. The notebook focuses on fundamental CUDA concepts and demonstrates how to perform parallel computations on NVIDIA GPUs using CUDA C. It includes examples of vector addition (1D data) and matrix addition (2D data), showcasing how to scale from simple to more complex operations.

---
# Intention
The primary intention of this repository is to provide a hands-on learning resource for beginners in CUDA programming. It aims to:
- Introduce core CUDA concepts such as kernels, threads, blocks, and grids.
- Explain how to map these concepts to different data structures (vectors, matrices).
- Demonstrate practical CUDA workflows, including memory allocation, data transfer between host (CPU) and device (GPU), kernel execution, and error handling.
<p>
Highlight the differences in thread organization for 1D (vectors) and 2D (matrices) data, preparing users for more advanced tensor operations in deep learning.
<p>
This notebook is tailored for users running code on Google Colab with GPU support (e.g., Tesla T4), as indicated by the metadata and setup commands.

---
# Content

The cuda_basics.ipynb notebook is structured as a tutorial with a mix of code cells, markdown explanations, and output logs. Below is a breakdown of its content:

1. **Setup and Environment Configuration**
- Installation: Installs the nvcc4jupyter plugin to enable CUDA compilation in Jupyter Notebook on Google Colab.
    - Command: !pip install nvcc4jupyter followed by %load_ext nvcc4jupyter.
- GPU and Compiler Check:
    - Uses !nvidia-smi to display GPU details (e.g., Tesla T4, CUDA Version 12.4).
    - Uses !nvcc --version to verify the CUDA compiler version (e.g., CUDA 12.5).

2. **Terminology and Concepts**
- A markdown section explains key CUDA terms:
    - Kernel: A function that runs on the GPU.
    - Threads: Individual workers executing the kernel on different data.
    - Thread Blocks: Groups of threads (up to 1024 per block).
    - Grid: Collection of all blocks.
- Includes a conceptual diagram (base64-encoded image, truncated in the document) to visualize the thread-block-grid hierarchy.


3. **Vector Addition Example (1D Data)**
- File: `vector_addition_updated.cu`
- Purpose: Demonstrates a simple CUDA program for adding two 1D vectors (`A + B = C`).
- Key Steps:
    - Defines a kernel `AddTwoVectors` that uses 1D thread indexing (`blockIdx.x * blockDim.x + threadIdx.x`).
    - Allocates memory on the host and GPU, copies data to the GPU, launches the kernel, and copies results back.
    - Uses a 1D grid and 1D blocks to match the vector’s linear structure.
- Execution:
    - Vector size: `N = 500000`.
    - Block size: Determined by GPU’s `maxThreadsPerBlock` (typically 1024).
    - Grid size: `(N + threads_per_block - 1) / threads_per_block`.
- Output: Prints the first 10 elements of the result vector `C`, showing correct addition (e.g.,` C[0] = 1.000000, C[1] = 1.099833`).

4. **Matrix Addition Example (2D Data)**
- File: `matrix_addition.cu`
- Purpose: Extends the concept to 2D data by adding two `N x N` matrices (`A + B = C`).
- Key Steps:
    - Defines a kernel `AddTwoMatrices` using 2D thread indexing (`row = blockIdx.y * blockDim.y + threadIdx.y, column = blockIdx.x * blockDim.x + threadIdx.x`).
    - Flattens the 2D matrices into 1D arrays for GPU memory allocation (`cudaMalloc`) and uses pointers (float *A).
    - Uses a 2D grid and 2D blocks (`dim3 threads_per_block(16, 16), dim3 gridDim((N + 15) / 16, (N + 15) / 16)`) to cover the matrix.
- Execution:
    - Matrix size: `N = 512` (512x512 matrix).
    - Block size: 16x16 threads (256 threads per block).
    - Grid size: 32x32 blocks (to cover 512x512 elements).
- Output: Prints the first 10x10 elements of the result matrix C, showing consistent values across columns due to initialization (A[i][j] = i + 10, B[i][j] = sinf(i * 0.1f)).

5. **Thread, Block, and Grid Dimensions for Different Data Types**
- A markdown section explains how to choose thread/block/grid dimensions based on data structure:
    - **1D Vectors**: Use 1D blocks and grids (as in `vector_addition_updated.cu`).
    - **2D Matrices**: Use 2D blocks and grids (as in `matrix_addition.cu`).
    - **3D Tensors**: Use 3D blocks and grids (conceptual, not implemented).
- Provides intuitive analogies (e.g., vectors as a row of boxes, matrices as a checkerboard, tensors as a Rubik’s cube) to explain the mapping.

---
# Conclusion

This repository provides a solid foundation for learning CUDA programming, starting with simple vector operations and progressing to matrix operations. It’s ideal for beginners looking to understand CUDA’s thread hierarchy and memory management, with clear examples and explanations. With minor improvements, it can serve as a comprehensive tutorial for GPU computing in deep learning contexts.
