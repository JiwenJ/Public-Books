/*
 * cuda_helloworld.cpp
 *
 *  Created on: 2009-3-27
 *      Author: deyuanqiu
 */

#include "CHelloWorld.cuh"

using namespace std;

int main(int argc, char **argv){
	CHelloWorld* hello = new CHelloWorld(argc, argv);

	delete hello;

	return EXIT_SUCCESS;
}
