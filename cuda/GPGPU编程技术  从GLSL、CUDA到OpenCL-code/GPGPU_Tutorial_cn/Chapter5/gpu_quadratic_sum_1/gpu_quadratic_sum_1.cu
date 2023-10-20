/*
 * @brief The first CUDA quadratic sum program.
 * @author Deyuan Qiu
 * @date June 9, 2009
 * @file gpu_quadratic_sum_1.cu
 */

#include <iostream>
#include "cutil.h"

#define DATA_SIZE 1048576

using namespace std;

int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++)	number[i] = rand() % 10;
}

//The kernel implemented by a global function: called from host, executed in device.
__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;
    for(unsigned i = 0; i < DATA_SIZE; i++)	sum += num[i] * num[i];

    *result = sum;
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	GenerateNumbers(data, DATA_SIZE);

	int *gpudata, *result;
	CUDA_SAFE_CALL(cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**) &result, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice));

	//Using only one scalar processer (single-thread).
	sumOfSquares<<<1, 1, 0>>>(gpudata, result);

	int sum = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(gpudata));
	CUDA_SAFE_CALL(cudaFree(result));

	cout<<"sum = "<<sum<<endl;

	return EXIT_SUCCESS;
}
