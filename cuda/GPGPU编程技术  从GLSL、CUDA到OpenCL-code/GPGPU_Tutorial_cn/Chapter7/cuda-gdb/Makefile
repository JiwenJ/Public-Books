LIBDIRS =	-L/usr/local/cuda/lib \
		-L/Developer/GPU\ Computing/C/lib
INCDIRS = 	-I/usr/local/cuda/include \
		-I/Developer/GPU\ Computing/C/common/inc

CUDA = nvcc
CUDAFLAGS = -g -G #--ptxas-options=-v
LIBS = 
PRG = cuda_gdb_1

${PRG}: ${PRG}.cu
	${CUDA} ${CUDAFLAGS} -o ${PRG} ${PRG}.cu ${INCDIRS} ${LIBDIRS} ${LIBS} 

clean:
	rm $(PRG)
