#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// Simple assignment test
#define N 16

__global__ void kernel(unsigned int *data, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) data[idx] = idx;
}

void getCudaErr(void){
	cudaError_t error=cudaGetLastError();
	std::cout<<cudaGetErrorString(error)<<std::endl;
}

int main(void)
{
    int i;
    unsigned int *d = NULL;
    unsigned int odata[N] = {0};

    cudaMalloc((void**)&d, sizeof(int) * (N-1) );

    kernel<<<1, N>>>(d,N);

getCudaErr();

    cudaMemcpy(&odata[0], d, sizeof(int)*N, cudaMemcpyDeviceToHost);

getCudaErr();

for(int j=0;j<N;j++)
std::cout<<odata[j]<<std::endl;

    // Test to see if the array retrieved from the GPU is correct
    for (i = 0; i < N; i++) {
      if(odata[i] != i) {
	break;
      }
    }
    if(i == N) printf("PASSED\n");
    else printf("FAILED\n");

    cudaFree((void*)d);std::cout<<i<<std::endl;


    return 0;
}

