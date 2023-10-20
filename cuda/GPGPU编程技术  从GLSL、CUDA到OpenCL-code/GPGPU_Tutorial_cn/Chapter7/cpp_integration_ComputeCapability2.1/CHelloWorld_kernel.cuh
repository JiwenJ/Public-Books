#ifndef CHELLOWORLD_KERNEL_CUH
#define CHELLOWORLD_KERNEL_CUH

//include CUDA headers
#include <cuda.h>
#include "cutil.h"

dim3 dimBlock;
dim3 dimGrid;

__global__ void sayHello_kernel(void);

extern "C"
void sayHello_agent(unsigned unBlockSize){
	dimBlock.x = unBlockSize;
	dimBlock.y = 1;
	dimBlock.z = 1;
	dimGrid.x = 1;
	dimGrid.y = 1;
	dimGrid.z = 1;

	//invoke the kernel
	sayHello_kernel<<<1, 8>>>();
	
	CUDA_SAFE_CALL(cudaThreadExit());
}

#include "CHelloWorld_kernel.cu"

#endif
