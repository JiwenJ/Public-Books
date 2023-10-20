#include "header.h"

//The kernel implemented by a global function: called from host, executed in device.
extern "C" __global__ static void sumOfSquares(int *num, int* result, int nSize)
{
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int i;
    shared[tid] = 0;
    for(i = bid * THREAD_NUM + tid; i < nSize;
        i += BLOCK_NUM * THREAD_NUM) {
    	shared[tid] += __mul24(num[i], num[i]);
    }

    __syncthreads();
    if(tid < 128) { shared[tid] += shared[tid + 128]; }
    __syncthreads();
    if(tid < 64) { shared[tid] += shared[tid + 64]; }
    __syncthreads();
    if(tid < 32) { shared[tid] += shared[tid + 32]; }
//    __syncthreads();
    if(tid < 16) { shared[tid] += shared[tid + 16]; }
//    __syncthreads();
    if(tid < 8) { shared[tid] += shared[tid + 8]; }
//    __syncthreads();
    if(tid < 4) { shared[tid] += shared[tid + 4]; }
//    __syncthreads();
    if(tid < 2) { shared[tid] += shared[tid + 2]; }
//    __syncthreads();
    if(tid < 1) { shared[tid] += shared[tid + 1]; }
//    __syncthreads();

	if (tid == 0) result[bid] = shared[0];
}

extern "C" cudaError_t kernel_caller(int nBlocks, int nThreads, int nShared,
		int* gpudata, int* result, int nSize) {
	sumOfSquares<<<nBlocks, nThreads, nShared>>>(gpudata, result, nSize);
#ifdef NDEBUG
	return cudaSuccess;
#else
	cudaThreadSynchronize();
	return cudaGetLastError();
#endif
}
