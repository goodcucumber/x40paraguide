#include <cuda_runtime.h>
#include "./sharedlib.h"

__global__ void cuadd(double* a, double* b, double* c, unsigned int N){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        c[id] = a[id] + b[id];
    }
}

void vecadd(double* a, double* b, double* c, unsigned int N){
    double* cua;
    double* cub;
    double* cuc;
    cudaMalloc(&cua, N*sizeof(double));
    cudaMalloc(&cub, N*sizeof(double));
    cudaMalloc(&cuc, N*sizeof(double));
    cudaMemcpy(cua, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cub, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cuadd<<<(N+127)/128,128>>>(cua,cub,cuc,N);
    cudaMemcpy(c, cuc, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cua);
    cudaFree(cub);
    cudaFree(cuc);
}


