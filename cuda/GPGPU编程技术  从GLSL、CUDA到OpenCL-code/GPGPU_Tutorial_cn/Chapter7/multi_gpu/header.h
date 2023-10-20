#include "cutil.h"
#define DATA_SIZE 	1048576	//data of 4 MB
#define DATA_SIZE0	655360
#define DATA_SIZE1	393216	//DATA_SIZE = DATA_SIZE0 + DATA_SIZE1
#define BLOCK_NUM	32
#define THREAD_NUM	256

extern "C" cudaError_t kernel_caller(int nBlocks, int nThreads, int nShared, int* gpudata, int* result, int nSize);
