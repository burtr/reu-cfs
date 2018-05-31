#include<cuda_runtime.h>
#include<stdio.h>
#include<assert.h>

/*
 * hello world, hello gpu
 * author: bjr
 * date: 29 may 2018
 * last update:
 */



// the GPU kernel. is downloaded onto the GPU and run in parallel

__global__ void hello_world_gpu(void) {
	printf ("Hello World from the GPU, block %d, thread %d\n",
		blockIdx.x, threadIdx.x) ;
}


// the main program. run on the CPU, launches and waits for GPU kernels

int main(int argc, char * argv[]) {
	cudaError_t status ;
	int ntpb ; // number of threads per block
	int nblk ; // number of blocks

	ntpb = 10 ;
	nblk = 1 ;

	printf("launching <<<%d,%d>>> kernel\n", nblk, ntpb ) ;
	hello_world_gpu <<<nblk,ntpb>>> () ;

	status = cudaDeviceSynchronize();
		if (status != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(status));
	}
	printf("kernel done!\n") ;

	return 0 ;
}

