//include user header files
#include "CHelloWorld_kernel.cuh"
#include "CHelloWorld.cuh"

void CHelloWorld::init(){
	//initialize CUTIL
	CUT_DEVICE_INIT(_argc, _argv);

	cudaDeviceProp deviceProp;
	deviceProp.major = 1;
	deviceProp.minor = 0;
	int desiredMinorRevision = 0;
	int dev;

	CUDA_SAFE_CALL(cudaChooseDevice(&dev, &deviceProp));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));

	if(deviceProp.major > 1 || deviceProp.minor >= desiredMinorRevision)
	{
		printf("Using Device %d: \"%s\"\n", dev, deviceProp.name);
		CUDA_SAFE_CALL(cudaSetDevice(dev));
	}
	else if(desiredMinorRevision == 3)
	{
		printf("There is no device supporting compute capability %d.%d.\n\n", 1, desiredMinorRevision);
		CUT_EXIT(_argc, _argv);
	}
}

CHelloWorld::~CHelloWorld(){
	//Done with CUTIL
	CUT_EXIT(_argc, _argv);
	printf("\nExit CUTIL.\n");
}
