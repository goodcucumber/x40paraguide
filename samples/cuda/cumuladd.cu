#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void muladd(double* a, double* b, double* c, double* d,
                       unsigned int N){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j;
    if(id < N){
        for(j = 0; j < 1000000; j++){
            d[id] = a[id] * b[id] + c[id];
        }
    }
}

int main(){
    double* a; 
    double* b; 
    double* c; 
    double* d;

    double* cua;
    double* cub;
    double* cuc;
    double* cud;

    a = (double*)(malloc(8192*sizeof(double)));
    b = (double*)(malloc(8192*sizeof(double)));
    c = (double*)(malloc(8192*sizeof(double)));
    d = (double*)(malloc(8192*sizeof(double)));

    cudaMalloc(&cua, 8192*sizeof(double));
    cudaMalloc(&cub, 8192*sizeof(double));
    cudaMalloc(&cuc, 8192*sizeof(double));
    cudaMalloc(&cud, 8192*sizeof(double));

    //Prepare data
    unsigned long long i;
    for(i = 0; i < 8192; i++){
        a[i] = (double)(rand()%2000) / 200.0;
        b[i] = (double)(rand()%2000) / 200.0;
        c[i] = ((double)i)/10000.0;
    }
    
    clock_t start, stop;
    double elapsed;
    start = clock();
    cudaMemcpy(cua, a, 8192*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cub, b, 8192*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuc, c, 8192*sizeof(double), cudaMemcpyHostToDevice);
    
    muladd<<<32, 256>>>(cua, cub, cuc, cud, 8192);
    
    cudaMemcpy(d, cud, 8192*sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    elapsed = (double)(stop-start) / CLOCKS_PER_SEC;
    printf("Elapsed time = %8.6f s\n", elapsed);
    for(i = 0; i < 8192; i++){
        if(i % 1001 == 0){
            printf("%5llu: %16.8f * %16.8f + %16.8f = %16.8f (%d)\n",
                   i, a[i], b[i], c[i], d[i], d[i]==a[i]*b[i]+c[i]);
        }
    }

    free(a);
    free(b);
    free(c);
    free(d);
    cudaFree(cua);
    cudaFree(cub);
    cudaFree(cuc);
    cudaFree(cud);
}
