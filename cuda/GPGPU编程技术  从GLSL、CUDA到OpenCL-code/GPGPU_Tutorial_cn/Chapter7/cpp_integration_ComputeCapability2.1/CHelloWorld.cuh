#ifndef CHELLOWORLD_CUH
#define CHELLOWORLD_CUH

//include system libraries
#include <cstdlib>
#include <iostream>

#define	BLOCKSIZE	8

using namespace std;

class CHelloWorld{
public:
	CHelloWorld(int argc, char **argv){
		_argc = argc;
		_argv = argv;
		init();
	}

	~CHelloWorld();

	void sayHello(void);

private:
	void init(void);
	int _argc;
	char** _argv;
	unsigned _nBlockSize;
};

#endif
