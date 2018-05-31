#include<cuda_runtime.h>
#include<stdio.h>
#include<assert.h>

/*
 * cumulatve sum on a GPU
 * author: bjr
 * date: nov 2017
 * last update: 29 may 2018
 */


// for synchronization, must be one block
#define N_ELEM 32
#define BLOCKS 1

#define PRINT_I 6
#define PRINT_L 2

void initialData(float *ip, int size) {
	time_t t ;
	int i ;
	static int j = 0 ;

	if (!j++) srand ((unsigned)time(&t)) ;
	for (i=0; i<size; i++) {
		ip[i] = (float) ( rand() & 0xFF ) / 10.0f ;
	}
	return ;
}

void printData(const char * s, float *ip, int n, int f, int l) {
	int i ;
	printf("%s\t",s) ;
	for (i=0;i<f;i++) {
		printf("%5.2f ", s, ip[i]) ;
	}
	printf("\t...\t") ;
	for (i=n-l;i<n;i++) {
		printf("%5.2f ", s, ip[i]) ;
	}
	printf("\n") ;
	return ;
}

// cuda kernel

__global__ void cumulate_sum(float * a,int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	assert(blockIdx.x==0) ;
	int level = 2 ;
	while (level<=n) {
		if (i%level==0) {
			a[i] += a[i+(level/2)] ;
		}
		level *= 2 ;
		__syncthreads() ;
	}
	return ;
}

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n = N_ELEM ;
	int nBytes = n * sizeof(float) ;
	float * h_a ;
	float * d_a ;
	int i ;
	float h_cs, d_cs ;
	int nblk ; // number of blocks
	int ntpb ; // number of threads per block
	
	nblk = BLOCKS ;
	ntpb = n/BLOCKS ; // must be divisible

	cudaSetDevice(dev) ;

	h_a = (float *) malloc(nBytes) ;

	initialData(h_a, n ) ;
	printData("a=", h_a, n, PRINT_I, PRINT_L) ;
	h_cs = 0.0 ;
	for (i=0;i<n;i++) {
		h_cs += h_a[i] ;
	}

	// send data to cuda device
	cudaMalloc((float **)&d_a, nBytes) ;
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice) ;
	
	printf("launching <<<%d,%d>>> kernel\n", nblk, ntpb ) ;
	cumulate_sum <<<nblk,ntpb>>> ( d_a, n ) ;
	printf("kernel done!\n") ;
	
	cudaMemcpy(&d_cs, d_a, sizeof(float), cudaMemcpyDeviceToHost) ;
	
	printf("cpu:sum = %6.2f\ngpu:sum = %6.2f\n", h_cs, d_cs ) ;

	cudaFree(d_a) ;
	free(h_a) ;

	return 0 ;
}

