#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cl.h>

using namespace std;

#define DATA_SIZE	65536

//
// Simple compute kernel which computes the square of an input array
//
const char *KernelSource = "\n" \
"__kernel void square( \n" \
" __global int* pnInput, \n" \
" __global int* pnOutput, \n" \
" const unsigned int unCount) \n" \
"{ \n" \
" int i = get_global_id(0); \n" \
" if(i < unCount) \n" \
" pnOutput[i] = pnInput[i] * pnInput[i]; \n" \
"} \n" \
"\n";

int main(int argc, char** argv) {
	int nError; 								// error code returned from api calls
	float pnData[DATA_SIZE]; 					// original data set given to device
	float pnResult[DATA_SIZE]; 				// results returned from device
	unsigned int unCorrect; 					// number of correct results returned
	size_t size_tGlobal; 							// global domain size for our calculation
	size_t size_tLocal; 							// local domain size for our calculation
	cl_device_id device_id; 				// compute device id
	cl_context context; 					// compute context
	cl_command_queue commands; 				// compute command queue
	cl_program program; 					// compute program
	cl_kernel kernel; 						// compute kernel
	cl_mem memInput; 							// device memory used for the input array
	cl_mem memOutput; 							// device memory used for the output array
	cl_platform_id platform_id = NULL;		// compute platform id
	cl_event event;
	cl_ulong start, stop;

	// Fill our data set with random float values
	int i = 0;
	unsigned int unCount = (unsigned)DATA_SIZE;
	for(i = 0; i < unCount; i++)
	pnData[i] = (int)(rand() / RAND_MAX * 10);

	// determine OpenCL platform
	nError = clGetPlatformIDs(1, &platform_id,NULL);

	// Connect to a compute device
	int gpu = 1;
	nError = clGetDeviceIDs(platform_id, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &nError);

	// Create a command commands
//	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &nError);
	commands = clCreateCommandQueue(context, device_id, NULL, &nError);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &nError);

	// Build the program executable
	nError = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "square", &nError);

	// Create the input and output arrays in device memory for our calculation
	memInput = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * unCount, NULL, NULL);
	memOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * unCount, NULL, NULL);

	// Write our data set into the input array in device memory
	nError = clEnqueueWriteBuffer(commands, memInput, CL_TRUE, 0, sizeof(float) * unCount, pnData, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	nError = 0;
	nError = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memInput);
	nError |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memOutput);
	nError |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &unCount);

	// Execute the kernel over the entire range of our 1d input data set
	// using one work item per work group (allows for arbitrary length of data array)
	size_tGlobal = unCount;
	size_tLocal = 256;
//	nError = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &size_tGlobal, &size_tLocal, 0, NULL, &event);
	nError = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &size_tGlobal, &size_tLocal, 0, NULL, NULL);

	// timing
//	clWaitForEvents(1, &event);
//	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
//	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &stop, NULL);

	// Wait for the command commands to get serviced before reading back results
	clFinish(commands);

	// Read back the results from the device to verify the output
	nError = clEnqueueReadBuffer( commands, memOutput, CL_TRUE, 0, sizeof(float) * unCount, pnResult, 0, NULL, NULL );

	// Validate our results
	unCorrect = 0;
	for(i = 0; i < unCount; i++)
	{
		if(pnResult[i] == pnData[i] * pnData[i])
			unCorrect++;
	}

	// Print a brief summary detailing the results
	printf("%d out of %d computed values are correct!\n", unCorrect, unCount);
//	cout<<"Time elapsed: "<< (stop - start) / 1000000.0 <<" ms"<<endl;

	// Shutdown and cleanup
	clReleaseMemObject(memInput);
	clReleaseMemObject(memOutput);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	clReleaseEvent(event);

	return 0;
}
