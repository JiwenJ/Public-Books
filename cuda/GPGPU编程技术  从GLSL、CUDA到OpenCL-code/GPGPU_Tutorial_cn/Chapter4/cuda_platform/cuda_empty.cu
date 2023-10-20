/*
 * @brief CUDA Initialization and environment check
 * @author Deyuan Qiu
 * @date June 5, 2009
 * @file cuda_platform.cu
 */

#include <iostream>
#include "cutil.h"

using namespace std;

bool InitCUDA()
{
    int count, dev;

    CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    else{
    	printf("\n%d Device(s) Found\n",count);
    	CUDA_SAFE_CALL(cudaGetDevice(&dev));
    	printf("The current Device ID is %d\n",dev);
    }

    int i = 0;
    bool bValid = false;
    cout<<endl<<"The following GPU(s) are detected:"<<endl;;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
        	cout<<"-------Device "<<i<<" -----------"<<endl;
        	cout<<prop.name<<endl;
        	cout<<"Total global memory: "<<prop.totalGlobalMem<<" Byte"<<endl;
        	cout<<"Maximum share memory per block: "<<prop.sharedMemPerBlock<<" Byte"<<endl;
        	cout<<"Maximum registers per block: "<<prop.regsPerBlock<<endl;
        	cout<<"Warp size: "<<prop.warpSize<<endl;
        	cout<<"Maximum threads per block: "<<prop.maxThreadsPerBlock<<endl;
        	cout<<"Maximum block dimensions: ["<<prop.maxThreadsDim[0]<<","<<prop.maxThreadsDim[1]<<","<<prop.maxThreadsDim[2]<<"]"<<endl;
        	cout<<"Maximum grid dimensions: ["<<prop.maxGridSize[0]<<","<<prop.maxGridSize[1]<<","<<prop.maxGridSize[2]<<"]"<<endl;
        	cout<<"Total constant memory: "<<prop.totalConstMem<<endl;
        	cout<<"Supports compute Capability: "<<prop.major<<"."<<prop.minor<<endl;
        	cout<<"Kernel frequency: "<<prop.clockRate<<" kHz"<<endl;
        	if(prop.deviceOverlap)	cout<<"Concurrent memory copy is supported."<<endl;
        	else	cout<<"Concurrent memory copy is not supported."<<endl;
        	cout<<"Number of multi-processors: "<<prop.multiProcessorCount<<endl;
            if(prop.major >= 1) {
                bValid = true;
            }
        }
    }
    cout<<"----------------"<<endl;

    if(!bValid) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    CUDA_SAFE_CALL(cudaSetDevice(1));

    return true;
}

int main()
{
    if(!InitCUDA())	return EXIT_FAILURE;

    printf("CUDA initialized.\n");

    return EXIT_SUCCESS;
}
