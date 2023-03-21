#include <cuda_runtime.h>

#include "multi_add.h"

__global__
void MultiAddCuda(double *a, double *b, double *c, double *d, unsigned int n) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		for (int loop = 0; loop < 100'000; ++loop)
			d[index] = a[index] * b[index] + c[index];
	}
}


void MultiAdd(double *a, double *b, double *c, double *d, unsigned int n) {
	double *cua, *cub, *cuc, *cud;

	cudaMalloc(&cua, n*sizeof(double));
	cudaMalloc(&cub, n*sizeof(double));
	cudaMalloc(&cuc, n*sizeof(double));
	cudaMalloc(&cud, n*sizeof(double));

	cudaMemcpy(cua, a, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cub, b, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuc, c, n*sizeof(double), cudaMemcpyHostToDevice);

	// 把循环放在这里是为了避免多次重复申请、释放显存空间
	MultiAddCuda<<<(n+255)/256, 256>>>(cua, cub, cuc, cud, n);

	cudaMemcpy(d, cud, n*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(cua);
	cudaFree(cub);
	cudaFree(cuc);
	cudaFree(cud);
}
