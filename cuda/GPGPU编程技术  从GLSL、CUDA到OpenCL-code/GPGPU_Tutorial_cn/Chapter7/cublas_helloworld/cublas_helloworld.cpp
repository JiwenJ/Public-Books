#include <iostream>
#include "cublas.h"

#define N  65536

using namespace std;

#define getError(cublasError)								\
	if (cublasError != CUBLAS_STATUS_SUCCESS) {				\
		cout << "CUBLAS error:" << cublasError << endl;		\
        return EXIT_FAILURE;								\
    }

int main(int argc, char** argv){
	unsigned unSize = (unsigned)N;
	float* pfDevice = NULL;
	float* pfHost = NULL;
	
	//Initialize CUBLAS
	cublasStatus status = cublasInit();
	getError(status);
	
	//Allocated memories
	status = cublasAlloc(unSize, sizeof(float), (void**)&pfDevice);
	getError(status);
	pfHost = new float[unSize];
	
	//Set host vector
	for (unsigned i = 0; i < unSize; i++)
		pfHost[i] = rand() / (float)RAND_MAX * 2;
	
	//Set device vector
	status = cublasSetVector(unSize, sizeof(float), pfHost, 1, pfDevice, 1);
	getError(status);
	
	//kernel
	cublasGetError();	//clear last error
	cout << cublasSasum(unSize, pfDevice, 1) << endl;	//a value around N
	getError(cublasGetError());
	
	//shutdown
	delete pfHost;
	status = cublasFree(pfDevice);
	getError(status);
	status = cublasShutdown();
	getError(status);
	
	return EXIT_SUCCESS;
}