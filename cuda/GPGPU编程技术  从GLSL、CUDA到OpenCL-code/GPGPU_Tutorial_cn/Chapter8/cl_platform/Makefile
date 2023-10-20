LIBDIRS =	
INCDIRS = -I/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers

C = g++-4.0
CFLAGS = -g
LIBS = -framework OpenCL
PRG = cl_platform

${PRG}: ${PRG}.cpp
	${C} ${CFLAGS} -o ${PRG} ${PRG}.cpp ${INCDIRS} ${LIBS} 

clean:
	rm -r $(PRG)
