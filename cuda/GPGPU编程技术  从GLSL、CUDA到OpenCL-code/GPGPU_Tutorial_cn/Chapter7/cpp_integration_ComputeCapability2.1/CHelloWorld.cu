//include user header files
#include "CHelloWorld_kernel.cuh"
#include "CHelloWorld.cuh"

void CHelloWorld::init(){
	//initialize CUTIL
	CUT_DEVICE_INIT(_argc, _argv);

	_nBlockSize = (unsigned)BLOCKSIZE;
}

CHelloWorld::~CHelloWorld(){
	//Done with CUTIL
	CUT_EXIT(_argc, _argv);
}

void CHelloWorld::sayHello(void){
	sayHello_agent(_nBlockSize);
}
