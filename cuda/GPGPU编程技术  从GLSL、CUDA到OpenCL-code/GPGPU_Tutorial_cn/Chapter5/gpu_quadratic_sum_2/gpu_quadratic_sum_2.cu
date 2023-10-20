/*
 * @brief The second CUDA quadratic sum program with parallelism.
 * @author Deyuan Qiu
 * @date June 21st, 2009
 * @file gpu_quadratic_sum_2.cu
 */

#include <iostream>
#include "cutil.h"

#define DATA_SIZE 1048576	//data of 4 MB
#define THREAD_NUM   256
#define FREQUENCY	783330	//set the GPU frequency in kHz

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
    const int size = DATA_SIZE / THREAD_NUM;
    int sum = 0;
    int i;
    clock_t start;
    if(tid == 0) start = clock();
    for(i = tid * size; i < (tid + 1) * size; i++) {
       sum += num[i] * num[i];
    }

    result[tid] = sum;
    if(tid == 0) *time = clock() - start;
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	int *data, *sum;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&data, DATA_SIZE*sizeof(int)));
	GenerateNumbers(data, DATA_SIZE);
	CUDA_SAFE_CALL(cudaMallocHost((void**)&sum, THREAD_NUM*sizeof(int)));

	int *gpudata, *result;
	clock_t *time;
	CUDA_SAFE_CALL(cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**) &result, sizeof(int) * THREAD_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**) &time, sizeof(clock_t)));
	CUDA_SAFE_CALL(cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice));

	//Using THREAD_NUM scalar processer.
	sumOfSquares<<<1, THREAD_NUM, 0>>>(gpudata, result, time);

	clock_t time_used;
	CUDA_SAFE_CALL(cudaMemcpy(sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost));

	//sum up on CPU
	int final_sum = 0;
	for (int i = 0; i < THREAD_NUM; i++)	final_sum += sum[i];

	printf("sum: %d  time: %d ms\n", final_sum, time_used/783330);

	//Clean up
	CUDA_SAFE_CALL(cudaFree(time));
	CUDA_SAFE_CALL(cudaFree(result));
	CUDA_SAFE_CALL(cudaFree(gpudata));
	CUDA_SAFE_CALL(cudaFreeHost(sum));
	CUDA_SAFE_CALL(cudaFreeHost(data));

	return EXIT_SUCCESS;
}
