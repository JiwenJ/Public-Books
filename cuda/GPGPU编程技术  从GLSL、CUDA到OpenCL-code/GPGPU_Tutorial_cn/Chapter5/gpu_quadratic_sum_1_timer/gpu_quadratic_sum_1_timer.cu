/*
 * @brief The first CUDA quadratic sum program with timing and page-locked memory.
 * @author Deyuan Qiu
 * @date June 9, 2009
 * @file gpu_quadratic_sum_1_timer.cu
 */


#include <iostream>
#include "cutil.h"

#define DATA_SIZE 1048576	//data of 4 MB

using namespace std;

void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++)	number[i] = rand() % 10;
}

//The kernel implemented by a global function: called from host, executed in device.
__global__ static void sumOfSquares(int *num, int* result, clock_t* time)
{
    int sum = 0;
    clock_t start = clock();
    for(unsigned i = 0; i < DATA_SIZE; i++)	sum += num[i] * num[i];

    *result = sum;
    *time = clock() - start;
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	int *data, *sum;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&data, DATA_SIZE*sizeof(int)));
	GenerateNumbers(data, DATA_SIZE);
	CUDA_SAFE_CALL(cudaMallocHost((void**)&sum, sizeof(int)));

	int *gpudata, *result;
	clock_t *time;
	CUDA_SAFE_CALL(cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**) &result, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &time, sizeof(clock_t)));
	CUDA_SAFE_CALL(cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice));

	//Using only one scalar processer (single-thread).
	sumOfSquares<<<1, 1, 0>>>(gpudata, result, time);

	clock_t time_used;
	CUDA_SAFE_CALL(cudaMemcpy(sum, result, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost));
	printf("sum: %d\ntime: %d\n", *sum, time_used);

	//Clean up
	CUDA_SAFE_CALL(cudaFree(time));
	CUDA_SAFE_CALL(cudaFree(result));
	CUDA_SAFE_CALL(cudaFree(gpudata));
	CUDA_SAFE_CALL(cudaFreeHost(sum));
	CUDA_SAFE_CALL(cudaFreeHost(data));

	return EXIT_SUCCESS;
}
