/*
 * @brief The forth CUDA quadratic sum program with increased threads.
 * @author Deyuan Qiu
 * @date June 21st, 2009
 * @file gpu_quadratic_sum_4.cu
 */

#include <iostream>
#include "cutil.h"

#define DATA_SIZE 1048576	//data of 4 MB
#define BLOCK_NUM	32
#define THREAD_NUM	256

using namespace std;

void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++)	number[i] = rand() % 10;
}

//The kernel implemented by a global function: called from host, executed in device.
__global__ static void sumOfSquares(int *num, int* result,
    clock_t* time)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int sum = 0;
    int i;
    if(tid == 0) time[bid] = clock();
    for(i = bid * THREAD_NUM + tid; i < DATA_SIZE;
        i += BLOCK_NUM * THREAD_NUM) {
       sum += num[i] * num[i];
    }

    result[bid * THREAD_NUM + tid] = sum;
    if(tid == 0) time[bid + BLOCK_NUM] = clock();
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	//allocate host page-locked memory
	int *data, *sum;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&data, DATA_SIZE*sizeof(int)));
	GenerateNumbers(data, DATA_SIZE);
	CUDA_SAFE_CALL(cudaMallocHost((void**)&sum, BLOCK_NUM*THREAD_NUM*sizeof(int)));
	clock_t *time_used;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&time_used, sizeof(clock_t) * BLOCK_NUM * 2));

	//allocate device memory
	int *gpudata, *result;
	clock_t *time;
	CUDA_SAFE_CALL(cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM * BLOCK_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**) &time, sizeof(clock_t) * BLOCK_NUM * 2));
	CUDA_SAFE_CALL(cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice));

	//Using THREAD_NUM scalar processer.
	sumOfSquares<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpudata, result, time);

	CUDA_SAFE_CALL(cudaMemcpy(sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost));

	//sum up on CPU
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM * BLOCK_NUM; i++)	final_sum += sum[i];

	//calculate the time: minimum start time - maximum end time.
	clock_t min_start, max_end;
	min_start = time_used[0];
	max_end = time_used[BLOCK_NUM];
	for (int i = 1; i < BLOCK_NUM; i++) {
		if (min_start > time_used[i])
			min_start = time_used[i];
		if (max_end < time_used[i + BLOCK_NUM])
			max_end = time_used[i + BLOCK_NUM];
	}

	printf("sum: %d  time: %d\n", final_sum, max_end - min_start);

	//Clean up
	CUDA_SAFE_CALL(cudaFree(time));
	CUDA_SAFE_CALL(cudaFree(result));
	CUDA_SAFE_CALL(cudaFree(gpudata));
	CUDA_SAFE_CALL(cudaFreeHost(sum));
	CUDA_SAFE_CALL(cudaFreeHost(data));
	CUDA_SAFE_CALL(cudaFreeHost(time_used));

	return EXIT_SUCCESS;
}
