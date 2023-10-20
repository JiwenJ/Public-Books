#########################
# Makefile for the examples of the book "GPGPU Programming - From GLSL, CUDA to OpenCL". Please read README for more information and INSTALL for a brief building guide.
# 03.06.2011
# Deyuan Qiu
# Version 0.2
# deyuan.qiu@gmail.com
#########################

MAKE = make
CLEAN =  make clean
INSTALL = make install

subsystem:
	@echo #
	@echo "+----------------------------+"
	@echo "|====== Compiling libs ======|"
	@echo "+----------------------------+"
	@cd CSystem && $(MAKE)
	@echo [ 0%] Compiling CSystem...
	@cd CTimer && $(MAKE)
	@echo [ 3%] Compiling CTimer...
	@cd CReader && $(MAKE)
	@echo [ 6%] Compiling CReader...
	@cd GPUWorker && $(MAKE)
	@echo [ 8%] Compiling GPUWorker...
	@echo #

	@echo "+-----------------------------+"
	@echo "|====== Installing libs ======|"
	@echo "+-----------------------------+"
	@cd CSystem && $(INSTALL)
	@echo [11%] installing CSystem...
	@cd CTimer && $(INSTALL)
	@echo [14%] installing CTimer...
	@cd CReader && $(INSTALL)
	@echo [17%] installing CReader...
	@cd GPUWorker && $(INSTALL)
	@echo [19%] installing GPUWorker...
	@echo #

	@echo "+--------------------------------+"
	@echo "|====== Compiling examples ======|"
	@echo "+--------------------------------+"
	@echo [25%] Compiling cpu_convolution...
	@cd Chapter1/cpu_convolution && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo [28%] Compiling glsl_uniform...
	@cd Chapter2/glsl_uniform && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [31%] Compiling minimum_opengl...
	@cd Chapter2/minimum_opengl && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [33%] Compiling minimum_shader...
	@cd Chapter2/minimum_shader && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo [36%] Compiling gpu_roundtrip...
	@cd Chapter3/gpu_roundtrip && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [39%] Compiling gpu_convolution...
	@cd Chapter3/gpu_convolution && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo [42%] Compiling cuda_platform...
	@cd Chapter4/cuda_platform && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [44%] Compiling cuda_1st_example...
	@cd Chapter4/cuda_1st_example && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo [47%] Compiling cpu_quadratic_sum...
	@cd Chapter5/cpu_quadratic_sum && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [50%] Compiling gpu_quadratic_sum_1...
	@cd Chapter5/gpu_quadratic_sum_1 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [53%] Compiling gpu_quadratic_sum_1_timer...
	@cd Chapter5/gpu_quadratic_sum_1_timer && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [56%] Compiling gpu_quadratic_sum_2...
	@cd Chapter5/gpu_quadratic_sum_2 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [58%] Compiling gpu_quadratic_sum_3...
	@cd Chapter5/gpu_quadratic_sum_3 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [61%] Compiling gpu_quadratic_sum_4...
	@cd Chapter5/gpu_quadratic_sum_4 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [64%] Compiling gpu_quadratic_sum_5...
	@cd Chapter5/gpu_quadratic_sum_5 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [67%] Compiling gpu_quadratic_sum_5_conflict_free...
	@cd Chapter5/gpu_quadratic_sum_5_conflict_free && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [70%]Compiling gpu_quadratic_sum_6...
	@cd Chapter5/gpu_quadratic_sum_6 && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo [72%] Compiling cuda_texture_roundtrip...
	@cd Chapter6/cuda_texture_roundtrip && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [75%] Compiling cuda_convolution...
	@cd Chapter6/cuda_convolution && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo [78%] Compiling cpp_integration_ComputeCapability1.1...
	@cd Chapter7/cpp_integration_ComputeCapability1.1 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [81%] Compiling cpp_integration_ComputeCapability2.1...
	@cd Chapter7/cpp_integration_ComputeCapability2.1 && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [84%] Compiling cuda-gdb...
	@cd Chapter7/cuda-gdb && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [87%] Compiling cuda_helloworld...
	@cd Chapter7/cublas_helloworld && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [90%] Skip multi_gpu, install boost and compile it on your own.
	@echo "---------------------------------"
	@echo #

	@echo [93%] Compiling cl_platform...
	@cd Chapter8/cl_platform && $(MAKE)
	@echo "---------------------------------"
	@echo #
	@echo [96%] Compiling cl_square...
	@cd Chapter8/cl_square && $(MAKE)
	@echo "---------------------------------"
	@echo #

	@echo Please find executables in each example subfolders.
	@echo [100%] Done! Please review the compiling errors yourself.

clean:
	@echo Tidying up temporary files for libs...
	@cd lib && rm *.a
	@echo Tidying up temporary files for examples...
	@cd Chapter1/cpu_convolution && $(CLEAN)
	@cd Chapter2/glsl_uniform && $(CLEAN)
	@cd Chapter2/minimum_opengl && $(CLEAN)
	@cd Chapter2/minimum_shader && $(CLEAN)
	@cd Chapter3/gpu_roundtrip && $(CLEAN)
	@cd Chapter3/gpu_convolution && $(CLEAN)
	@cd Chapter4/cuda_platform && $(CLEAN)
	@cd Chapter4/cuda_1st_example && $(CLEAN)
	@cd Chapter5/cpu_quadratic_sum && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_1 && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_1_timer && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_2 && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_3 && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_4 && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_5 && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_5_conflict_free && $(CLEAN)
	@cd Chapter5/gpu_quadratic_sum_6 && $(CLEAN)
	@cd Chapter6/cuda_texture_roundtrip && $(CLEAN)
	@cd Chapter6/cuda_convolution && $(CLEAN)
	@cd Chapter7/cpp_integration_ComputeCapability1.1 && $(CLEAN)
	@cd Chapter7/cpp_integration_ComputeCapability2.1 && $(CLEAN)
	@cd Chapter7/cuda-gdb && $(CLEAN)
	@cd Chapter7/cublas_helloworld && $(CLEAN)
	@cd Chapter8/cl_platform && $(CLEAN)
	@cd Chapter8/cl_square && $(CLEAN)
	@echo Done!
