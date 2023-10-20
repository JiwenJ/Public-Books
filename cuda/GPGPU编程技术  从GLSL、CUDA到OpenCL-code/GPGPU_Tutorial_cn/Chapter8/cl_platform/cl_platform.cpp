/*
 * cl_platform.cpp
 *
 *  Created on: 2010-11-16
 *      Author: deyuanqiu
 */

#include <stdio.h>
#include <cl.h>

int main(int argc, char** argv) {
	char dname[500];
	cl_device_id devices[10];
	cl_uint num_devices, entries;
	cl_ulong long_entries;
	int d;
	cl_int err;
	cl_platform_id platform_id = NULL;
	size_t p_size;

	/* obtain list of platforms available */
	err = clGetPlatformIDs(1, &platform_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failure in clGetPlatformIDs,error code=%d \n", err);
		return 0;
	}

	/* obtain information about platform */
	clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 500, dname, NULL);
	printf("CL_PLATFORM_NAME = %s\n", dname);
	clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 500, dname, NULL);
	printf("CL_PLATFORM_VERSION = %s\n", dname);

	/* obtain list of devices available on platform */
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);
	printf("%d devices found\n", num_devices);

	/* query devices for information */
	for (d = 0; d < num_devices; ++d) {
		clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 500, dname, NULL);
		printf("Device #%d name = %s\n", d, dname);
		clGetDeviceInfo(devices[d], CL_DRIVER_VERSION, 500, dname, NULL);
		printf("\tDriver version = %s\n", dname);
		clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(cl_ulong), &long_entries, NULL);
		printf("\tGlobal Memory (MB):\t%llu\n", long_entries / 1024 / 1024);
		clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
				sizeof(cl_ulong), &long_entries, NULL);
		printf("\tGlobal Memory Cache (MB):\t%llu\n", long_entries / 1024
				/ 1024);
		clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
				&long_entries, NULL);
		printf("\tLocal Memory (KB):\t%llu\n", long_entries / 1024);
		clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY,
				sizeof(cl_ulong), &long_entries, NULL);
		printf("\tMax clock (MHz) :\t%llu\n", long_entries);
		clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(size_t), &p_size, NULL);
		printf("\tMax Work Group Size:\t%d\n", p_size);
		clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(cl_uint), &entries, NULL);
		printf("\tNumber of parallel compute cores:\t%d\n", entries);
	}
	return 0;
}
