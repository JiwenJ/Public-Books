/*
 * @brief CUDA memory roundtrip.
 * @author Deyuan Qiu
 * @date June 24th, 2009
 * @file cuda_texture_roundtrip.cu
 */

#include <iostream>
#include "cutil.h"

#define DATA_SIZE	8
using namespace std;

//texture variables
texture<int, 1, cudaReadModeElementType> refTex;
cudaArray* cuArray;

//the kernel: invert the input numbers.
__global__ void convolution(unsigned unSizeData, int* pnResult){
	const int idxX = threadIdx.x;
	pnResult[idxX] = unSizeData + 1 - tex1D(refTex, idxX);
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	//prepare data
	unsigned unSizeData = (unsigned)DATA_SIZE;
	unsigned unData = 0;
	int* pnSampler;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&pnSampler, unSizeData * sizeof(int)));
	for(unsigned i=0; i<unSizeData; i++)	pnSampler[i] = ++unData;
	for(unsigned i=0; i<unSizeData; i++) 	cout<<pnSampler[i]<<'\t';	cout<<endl;	//data before roundtrip

	//prepare texture to read from
	cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<int>();
	CUDA_SAFE_CALL(cudaMallocArray(&cuArray, &cuDesc, unSizeData));
	CUDA_SAFE_CALL(cudaMemcpyToArray(cuArray, 0, 0, pnSampler, unSizeData * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(refTex, cuArray));

	//allocate global memory to write to
	int* pnResult;
	CUDA_SAFE_CALL(cudaMalloc((void**)&pnResult, unSizeData * sizeof(int)));

	//call global function
    convolution<<<1, unSizeData>>>(unSizeData, pnResult);

    //fetch result
    CUDA_SAFE_CALL(cudaMemcpy(pnSampler, pnResult, unSizeData * sizeof(int), cudaMemcpyDeviceToHost));
    for(unsigned i=0; i<unSizeData; i++) 	cout<<pnSampler[i]<<'\t';	cout<<endl;	//data after roundtrip

    //garbage collection
    CUDA_SAFE_CALL(cudaUnbindTexture(refTex));
    CUDA_SAFE_CALL(cudaFreeHost(pnSampler));
    CUDA_SAFE_CALL(cudaFreeArray(cuArray));
    CUDA_SAFE_CALL(cudaFree(pnResult));

	return EXIT_SUCCESS;
}
