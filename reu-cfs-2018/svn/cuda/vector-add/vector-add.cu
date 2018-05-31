#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>

/*
 * add vectors in a GPU
 * author: bjr
 * date: nov 2017
 * last update: 29 may 2018
 */


// N_ELEM must be divisible by BLOCKS
#define N_ELEM 32
#define BLOCKS 4

#define PRINT_I 6
#define PRINT_L 2

void initialData(float *ip, int size) {

	// given:
	//   ip a pointer to an array of floats
	//   an integer of the size of the array
	// returns the array filled with random floats

	time_t t ;
	int i ;
	static int j = 0 ;

	if (!j++) srand ((unsigned)time(&t)) ; // first time initialize
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

float distance(float * a, float * b, float *c, int n) {
	float f, dist = 0.0 ;
	int i ;
	for (i=0;i<n;i++) {
		f = b[i] + c[i] - a[i] ;
		dist += f*f ;
	}
	return sqrt(dist) ;
}






// cuda kernel

__global__ void sum_array(float * a, float *b, float * c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	a[i] = b[i] + c[i] ;
	return ;
}

// host program

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n = N_ELEM ;
	int nBytes = n * sizeof(float) ;
	float * h_a, * h_b, * h_c ;
	float * d_a, * d_b, * d_c ;
	int nblk ; // number of blocks
	int ntpb ; // number of threads per block
	nblk = BLOCKS ;
	ntpb = n/BLOCKS ; // must be disvisible


	cudaSetDevice(dev) ;

	h_a = (float *) malloc(nBytes) ;
	h_b = (float *) malloc(nBytes) ;
	h_c = (float *) malloc(nBytes) ;

	initialData(h_b, n ) ;
	initialData(h_c, n ) ;
	printData("b =", h_b,n) ;
	printData("c =", h_c,n) ;

	// send data to cuda device
	cudaMalloc((float **)&d_a, nBytes) ;
	cudaMalloc((float **)&d_b, nBytes) ;
	cudaMalloc((float **)&d_c, nBytes) ;
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice) ;
	cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice) ;

	printf("launching <<<%d,%d>>> kernel\n", nblk, ntpb ) ;
	sum_array <<<nblk,ntpb>>> ( d_a, d_b, d_c ) ;
	printf("kernel done!\n") ;
	cudaMemcpy(h_a, d_a, nBytes, cudaMemcpyDeviceToHost) ;
	
	printData("sum =",h_a,n) ;
	printf("error = %f\n", distance(h_a,h_b,h_c,n) ) ;

	cudaFree(d_a) ;
	cudaFree(d_b) ;
	cudaFree(d_c) ;
	free(h_a) ;
	free(h_b) ;
	free(h_c) ;

	return 0 ;
}

