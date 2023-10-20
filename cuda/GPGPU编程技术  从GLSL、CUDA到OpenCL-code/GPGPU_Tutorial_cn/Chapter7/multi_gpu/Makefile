LIBDIRS =	-L/usr/local/cuda/lib \
			-L/Developer/CUDA/lib \
			-L. \
			-L../lib
INCDIRS = 	-I/usr/local/cuda/include \
			-I/Developer/GPU\ Computing/C/common/inc \
			-I/usr/local/boost_1_39_0

C = g++-4.0
CFLAGS = -g
CUDA = nvcc
CUDAFLAGS = -g #--ptxas-options=-v
LIBS = -lcutil_i386 -lcuda -lcudart -lCTimer -lGPUWorker -lboost_thread-xgcc40-mt-1_39
PRG = multi_gpu

${PRG}: ${PRG}.cpp
	${CUDA} -c ${CUDAFLAGS} kernel.cu ${INCDIRS}
	${C} ${CFLAGS} -o ${PRG} ${PRG}.cpp kernel.o ${INCDIRS} ${LIBDIRS} ${LIBS} 

clean:
	rm $(PRG) *.linkinfo *.o
