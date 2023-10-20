LIBDIRS =
INCDIRS = -I/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers

C = g++-4.0
CFLAGS = -g
LIBS = -framework OpenCL
PRG = cl_square

${PRG}: ${PRG}.cpp
	${C} ${CFLAGS} -o ${PRG} ${PRG}.cpp ${LIBS} ${INCDIRS}

clean:
	rm -r $(PRG)
