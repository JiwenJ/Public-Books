/*
 * @brief The First Example: Discrete Convolution
 * @author Deyuan Qiu
 * @date May 6, 2009
 * @file convolution.cpp
 */

#include <iostream>
#include "CTimer.h"
#include "CSystem.h"

#define WIDTH 		1024					//Width of the image
#define HEIGHT		1024					//Height of the image
#define CHANNEL		4						//Number of channels
#define RADIUS		2						//Mask radius

using namespace std;

int main(int argc, char **argv)
{
	int nState = EXIT_SUCCESS;
	int unWidth = (int)WIDTH;
	int unHeight = (int)HEIGHT;
	int unChannel = (int)CHANNEL;
	int unRadius = (int)RADIUS;

	//Generate input matrix
	float ***fX;
	int unData = 0;
	CSystem<float>::allocate(unHeight, unWidth, unChannel, fX);
	for(int i=0; i<unHeight; i++)
		for(int j=0; j<unWidth; j++)
			for(int k=0; k<unChannel; k++){
				fX[k][j][i] = (float)unData;unData++;
		}

	//Generate output matrix
	float ***fY;
	CSystem<float>::allocate(unHeight, unWidth, unChannel, fY);
	for(int i=0; i<unHeight; i++)
		for(int j=0; j<unWidth; j++)
			for(int k=0; k<unChannel; k++){
				fY[k][j][i] = 0.0f;
		}


	//Convolution
	float fSum = 0.0f;
	int unTotal = 0;
	CTimer timer;
	timer.reset();

	for(int i=0; i<unHeight; i++)
		for(int j=0; j<unWidth; j++)
			for(int k=0; k<unChannel; k++){
				for(int ii=i-unRadius; ii<=i+unRadius; ii++)
					for(int jj=j-unRadius; jj<=j+unRadius; jj++){
						if(ii>=0 && jj>=0 && ii<unHeight && jj<unWidth){
							fSum += fX[k][jj][ii];
							unTotal++;
						}
					}
				fY[k][j][i] = fSum / (float)unTotal;
				unTotal = 0;
				fSum = 0.0f;
			}

	long lTime = timer.getTime();
	cout<<"Time elapsed: "<<lTime<<" milliseconds."<<endl;

	CSystem<float>::deallocate(fX);
	CSystem<float>::deallocate(fY);
	return nState;
}
