__global__ void sayHello_kernel(void){
	printf("Hello from thread %d\n", threadIdx.x);
}
