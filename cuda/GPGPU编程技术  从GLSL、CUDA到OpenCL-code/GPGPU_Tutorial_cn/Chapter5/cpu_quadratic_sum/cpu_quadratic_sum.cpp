/*
 * @brief Quadratic sum on CPU
 * @author Deyuan Qiu
 * @date June 10, 2009
 * @file cpu_quadratic_sum.cpp
 */

#include <iostream>
#include "CTimer.h"

#define DATA_SIZE 1048576

using namespace std;

void GenerateData(float *data, unsigned size)
{
    for(unsigned i = 0; i < size; i++) {
        data[i] = rand() / 100000000.0;
    }
}

int main(int argc, char **argv){
	unsigned unSize = (unsigned)DATA_SIZE;
	float pfData[unSize];
	GenerateData(pfData, unSize);//for(unsigned i=0; i<unSize; i++) cout<<pfData[i]<<endl;

	float fSum = 0.0;
	CTimer timer;
	timer.reset();
	for(unsigned i=0; i<unSize; i++){
		fSum += pfData[i] * pfData[i];
	}
	long lTime = timer.getTime();
	cout<<"Time elapsed: "<<lTime<<" milliseconds."<<endl;cout<<fSum<<endl;

	return EXIT_SUCCESS;
}
