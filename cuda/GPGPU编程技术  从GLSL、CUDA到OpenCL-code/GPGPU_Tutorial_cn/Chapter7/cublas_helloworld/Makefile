LIBDIRS =		-L/usr/local/cuda/lib \
			-L/usr/lib \
			-L/Developer/GPU\ Computing/C/lib/
INCDIRS = 	-I/usr/local/cuda/include \

C = g++-4.0
CFLAGS = -g -Wall
#CUDA = nvcc
#CUDAFLAGS = -O3 --host-compilation=C++ --compiler-bindir=/usr/bin/g++-4.3 #--ptxas-options=-v
LIBS = -lcutil_i386 -lcublas -lcuda -lcudart
PRG = cublas_helloworld

${PRG}: ${PRG}.cpp	
	${C} ${CFLAGS} ${LIBS} ${PRG}.cpp ${INCDIRS} ${LIBDIRS} -o ${PRG}


clean:
	rm $(PRG)
	
redo:
	rm $(PRG)
	make
