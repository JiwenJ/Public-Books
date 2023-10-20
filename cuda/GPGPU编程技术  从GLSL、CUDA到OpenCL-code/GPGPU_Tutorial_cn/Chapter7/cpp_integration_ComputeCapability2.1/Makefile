LIBDIRS =	-L/usr/local/cuda/lib \
		-L/Developer/GPU\ Computing/C/lib
INCDIRS = 	-I/usr/local/cuda/include \
		-I/Developer/GPU\ Computing/C/common/inc

CC = g++
CUDA = nvcc
CFLAGS = 
CUDAFLAGS = -g -arch=sm_21 #--ptxas-options=-v
LIBS = -lcutil_i386 -lcuda -lcudart
PRG = CHelloWorld
PRGKERNEL = CHelloWorld_kernel
APP = cuda_helloworld
${APP}: ${APP}.cpp ${PRG}.cu ${PRG}.cuh ${PRGKERNEL}.cu ${PRGKERNEL}.cuh
	${CUDA} -c ${CUDAFLAGS} ${PRG}.cu ${INCDIRS}
	${CUDA} ${CFLAGS} ${LIBDIRS} ${APP}.cpp ${PRG}.o -o ${APP} ${LIBS} ${INCDIRS}

clean:
	rm $(APP)
	rm *.o
