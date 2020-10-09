#include "cuda.h"

__global__ void kernel_saxpy( int n, float a, float * x, float * y, float * z ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { 
		z[i] = a * x[i] + y [i];
	}
}

void saxpy( int nblocks, int nthreads, int n, float a, float * x, float * y, float * z ) {
	kernel_saxpy<<<nblocks, nthreads>>>( n, a, x, y, z );
}
