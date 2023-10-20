/*
 * @brief Using two GPUs concurrently for the discrete convolution.
 * @author Deyuan Qiu
 * @date June 28nd, 2009
 * @file multi_gpu.cpp
 */
#include <cuda_runtime.h>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include "../GPUWorker/GPUWorker.h"
#include "../CTimer/CTimer.h"
#include "header.h"

using namespace std;
using namespace boost;

void GenerateNumbers(int *number0, int *number1, int size0, int size1)
{
    for(int i = 0; i < size0; i++)	number0[i] = rand() % 10;
    for(int i = 0; i < size1; i++)	number1[i] = rand() % 10;
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	//allocate host page-locked memory
	int *data0, *data1, *sum0, *sum1;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&data0, DATA_SIZE0*sizeof(int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&data1, DATA_SIZE1*sizeof(int)));
	GenerateNumbers(data0, data1, DATA_SIZE0, DATA_SIZE1);
	CUDA_SAFE_CALL(cudaMallocHost((void**)&sum0, BLOCK_NUM*sizeof(int)));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&sum1, BLOCK_NUM*sizeof(int)));

	//specify two GPUs
	GPUWorker gpu0(0);
	GPUWorker gpu1(1);

	//allocate device memory
	int *gpudata0, *gpudata1, *result0, *result1;
	gpu0.call(bind(cudaMalloc, (void**)(&gpudata0), sizeof(int) * DATA_SIZE0));
	gpu0.call(bind(cudaMalloc, (void**)(&result0), sizeof(int) * BLOCK_NUM));
	gpu1.call(bind(cudaMalloc, (void**)(&gpudata1), sizeof(int) * DATA_SIZE1));
	gpu1.call(bind(cudaMalloc, (void**)(&result1), sizeof(int) * BLOCK_NUM));
	CTimer timer;

	//transfer data to device
	gpu0.callAsync(bind(cudaMemcpy, gpudata0, data0, sizeof(int) * DATA_SIZE0, cudaMemcpyHostToDevice));
	gpu1.callAsync(bind(cudaMemcpy, gpudata1, data1, sizeof(int) * DATA_SIZE1, cudaMemcpyHostToDevice));

	//call global functions
	gpu0.callAsync(bind(kernel_caller, BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int), gpudata0, result0, DATA_SIZE0));
	gpu1.callAsync(bind(kernel_caller, BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int), gpudata1, result1, DATA_SIZE1));
	gpu0.callAsync(bind(cudaMemcpy, sum0, result0, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost));
	gpu1.callAsync(bind(cudaMemcpy, sum1, result1, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost));

	//get timing result
	gpu0.call(bind(cudaThreadSynchronize));
	gpu1.call(bind(cudaThreadSynchronize));
	long lTime = timer.getTime();
	cout<<"time: "<<lTime<<endl;

	//sum up on CPU
	int final_sum0 = 0;
	int final_sum1 = 0;
	for (int i = 0; i < BLOCK_NUM; i++)	final_sum0 += sum0[i];
	for (int i = 0; i < BLOCK_NUM; i++)	final_sum1 += sum1[i];
	int final_sum = final_sum0 + final_sum1;
	cout<<"sum: "<<final_sum<<endl;

	//Clean up
	gpu0.call(bind(cudaFree, result0));
	gpu1.call(bind(cudaFree, result1));
	gpu0.call(bind(cudaFree, gpudata0));
	gpu1.call(bind(cudaFree, gpudata1));
	CUDA_SAFE_CALL(cudaFreeHost(sum0));
	CUDA_SAFE_CALL(cudaFreeHost(sum1));
	CUDA_SAFE_CALL(cudaFreeHost(data0));
	CUDA_SAFE_CALL(cudaFreeHost(data1));

	return EXIT_SUCCESS;
}
