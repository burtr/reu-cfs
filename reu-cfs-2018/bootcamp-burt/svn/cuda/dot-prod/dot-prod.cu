#include<cuda_runtime.h>
#include<stdio.h>
#include<assert.h>

/*
 * dot-product of two vectors
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

void printData(const char * s, float *ip, int n) {
	int i ;
	int f = PRINT_I ;
	int l = PRINT_L ;
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

__device__ void vector_mul(float * a,float *b, int n) {
	
	// write code here

	return ;
}

__device__ void cumulate_sum(float * a,int n) {
	
	// write code here

	return ;
}

__global__ void dot_prod(float * a, float * b, int n) {
	vector_mul(a,b,n) ;
	cumulate_sum(a,n) ;
	*a = 0.0 ;
	return ;
}

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n = N_ELEM ;
	int nBytes = n * sizeof(float) ;
	float * h_a ;
	float * h_b ;
	float * d_a ;
	float * d_b ;
	int i ;
	float h_dp, d_dp ;
	int nblk = BLOCKS ;
	int ntpb = n/BLOCKS ;
	
	cudaSetDevice(dev) ;

	h_a = (float *) malloc(nBytes) ;
	h_b = (float *) malloc(nBytes) ;

	initialData(h_a, n ) ;
	initialData(h_b, n ) ;
	printData("a=", h_a, n) ;
	printData("b=", h_b, n) ;
	
	h_dp = 0.0 ;
	for (i=0;i<n;i++) {
		h_dp += h_a[i]*h_b[i] ;
	}

	// send data to cuda device
	cudaMalloc((float **)&d_a, nBytes) ;
	cudaMalloc((float **)&d_b, nBytes) ;
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice) ;
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice) ;
	
	printf("launching <<<%d,%d>>> kernel\n", nblk, ntpb ) ;
        dot_prod <<<nblk,ntpb>>> ( d_a, d_b, n ) ;
	printf("kernel done!\n") ;
	
	cudaMemcpy(&d_dp, d_a, sizeof(float), cudaMemcpyDeviceToHost) ;
	
	printf("cpu:dotprod = %6.2f\ngpu:dotprod = %6.2f\n", h_dp, d_dp ) ;

	cudaFree(d_a) ;
	cudaFree(d_b) ;
	free(h_a) ;
	free(h_b) ;

	return 0 ;
}

