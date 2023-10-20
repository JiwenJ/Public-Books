LIBDIRS =	-L/usr/local/cuda/lib \
		-L/Developer/GPU\ Computing/C/lib \
                -L/Developer/GPU\ Computing/C/common/lib
INCDIRS = 	-I/usr/local/cuda/include \
                -I/Developer/GPU\ Computing/C/common/inc

CUDA = nvcc
CUDAFLAGS = -g -G -deviceemu #--ptxas-options=-v
LIBS = -lcutil_i386 -lcuda -lcudart
PRG = cuda_1st_example

${PRG}: ${PRG}.cu
	${CUDA} ${CUDAFLAGS} -o ${PRG} ${PRG}.cu ${INCDIRS} ${LIBDIRS} ${LIBS} 

clean:
	rm $(PRG)
