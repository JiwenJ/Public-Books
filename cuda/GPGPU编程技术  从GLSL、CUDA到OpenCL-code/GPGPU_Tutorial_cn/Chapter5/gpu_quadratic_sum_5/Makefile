LIBDIRS =	-L/usr/local/cuda/lib \
		-L/Developer/GPU\ Computing/C/lib
INCDIRS = 	-I/usr/local/cuda/include \
		-I/Developer/GPU\ Computing/C/common/inc

CUDA = nvcc
CUDAFLAGS = -O3 #--ptxas-options=-v
LIBS = -lcutil_i386 -lcuda -lcudart
PRG = gpu_quadratic_sum_5

${PRG}: ${PRG}.cu
	${CUDA} ${CUDAFLAGS} -o ${PRG} ${PRG}.cu ${INCDIRS} ${LIBDIRS} ${LIBS} 

clean:
	rm $(PRG)
