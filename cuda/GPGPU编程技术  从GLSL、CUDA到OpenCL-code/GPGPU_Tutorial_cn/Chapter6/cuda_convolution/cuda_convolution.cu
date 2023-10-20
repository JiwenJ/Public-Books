/*
 * @brief CUDA-accelerated discrete convolution.
 * @author Deyuan Qiu
 * @date June 24th, 2009
 * @file cuda_convolution.cu
 */

#include <iostream>
#include "cutil.h"

#define WIDTH	1024
#define HEIGHT	1024
#define CHANNEL	4
#define BLOCK_X	16
#define BLOCK_Y	16	//The block of [BLOCK_X x BLOCK_Y] threads.
#define RADIUS	2

#define VectorAdd(a, b)	\
	a.x += b.x;	a.y += b.y; a.z += b.z;	a.w += b.w;

#define VectorDev(a, b, c)	\
	a.x = b.x / c; a.y = b.y / c; a.z = b.z / c; a.w = b.w / c;

using namespace std;

//texture variables
texture<float4, 2, cudaReadModeElementType> refTex;
cudaArray* cuArray;

__global__ void convolution(int nWidth, int nHeight, int nRadius, float4* pfResult){
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x,
			  idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxResult = idxY * nHeight + idxX;

	float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};		//Sum of the neighborhood.
	int nTotal = 0;									//NoPoints in the neighborhood.
	float4 f4Result = {0.0f, 0.0f, 0.0f, 0.0f};		//Output vector to replace the current texture
	float4 f4Temp = {0.0f, 0.0f, 0.0f, 0.0f};

	//Neighborhood summation.
	for (int ii = idxX - nRadius; ii < idxX + nRadius; ii++)
		for (int jj = idxY - nRadius; jj <= idxY + nRadius; jj++)
			if (ii >= 0 && jj >= 0 && ii < nWidth && jj < nHeight) {
				f4Temp = tex2D(refTex, ii, jj);
				VectorAdd(f4Sum,f4Temp);
				nTotal++;
			}
//	f4Result.x = f4Sum.x/(float)nTotal;
//	f4Result.y = f4Sum.y/(float)nTotal;
//	f4Result.z = f4Sum.z/(float)nTotal;
//	f4Result.w = f4Sum.w/(float)nTotal;
	VectorDev(f4Result, f4Sum, (float)nTotal);
	pfResult[idxResult] = f4Result;
}

int main(int argc, char **argv)
{
	CUT_DEVICE_INIT(argc, argv);

	unsigned unWidth = (unsigned)WIDTH;
	unsigned unHeight = (unsigned)HEIGHT;
	unsigned unSizeData = unWidth * unHeight;
	unsigned unRadius = (unsigned)RADIUS;

	//prepare data
	unsigned unData = 0;
	float4* pf4Sampler;
	CUDA_SAFE_CALL(cudaMallocHost((void**)&pf4Sampler, unSizeData * sizeof(float4)));
	for(unsigned i=0; i<unSizeData; i++){
		pf4Sampler[i].x = (float)(unData++);
		pf4Sampler[i].y = (float)(unData++);
		pf4Sampler[i].z = (float)(unData++);
		pf4Sampler[i].w = (float)(unData++);
	}

	//prepare texture
	cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<float4>();
	CUDA_SAFE_CALL(cudaMallocArray(&cuArray, &cuDesc, unWidth, unHeight));
	CUDA_SAFE_CALL(cudaMemcpyToArray(cuArray, 0, 0, pf4Sampler, unSizeData * sizeof(float4), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(refTex, cuArray));

	//allocate global memory to write to
	float4* pfResult;
	CUDA_SAFE_CALL(cudaMalloc((void**)&pfResult, unSizeData * sizeof(float4)));

	//allocate threads and call the global function
    dim3 block(BLOCK_X, BLOCK_Y),
         grid(ceil((float)unWidth/BLOCK_X), ceil((float)unHeight/BLOCK_Y));
    convolution<<<grid, block>>>(unWidth, unHeight, unRadius, pfResult);

    //fetch result
    CUDA_SAFE_CALL(cudaMemcpy(pf4Sampler, pfResult, unSizeData * sizeof(float4), cudaMemcpyDeviceToHost));

    //garbage collection
    CUDA_SAFE_CALL(cudaUnbindTexture(refTex));
    CUDA_SAFE_CALL(cudaFreeHost(pf4Sampler));
    CUDA_SAFE_CALL(cudaFreeArray(cuArray));
    CUDA_SAFE_CALL(cudaFree(pfResult));

	return EXIT_SUCCESS;
}
