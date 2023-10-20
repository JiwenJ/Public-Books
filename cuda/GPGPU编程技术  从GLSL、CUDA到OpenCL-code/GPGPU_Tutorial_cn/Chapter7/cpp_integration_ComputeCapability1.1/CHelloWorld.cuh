#ifndef CHELLOWORLD_CUH
#define CHELLOWORLD_CUH

//include system libraries
#include <cstdlib>
#include <iostream>

using namespace std;

class CHelloWorld{
public:
	CHelloWorld(int argc, char **argv){
		_argc = argc;
		_argv = argv;
		init();
	}

	~CHelloWorld();

	void getSystemInfo(void);

private:
	void init(void);
	int _argc;
	char** _argv;
};

#endif
