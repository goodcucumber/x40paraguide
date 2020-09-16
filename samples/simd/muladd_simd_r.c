#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

__attribute__ ((noinline))
void muladd(double* a, double* b, double* c, double* d, 
            unsigned long long N){
    unsigned long long i, j;
    register unsigned long long M = N>>2;
    for(i = 0; i < M; i++){
        register __m256d ymma = _mm256_load_pd(a+i*4);
        register __m256d ymmb = _mm256_load_pd(b+i*4);
        register __m256d ymmc = _mm256_load_pd(c+i*4);
        register __m256d ymmd = _mm256_mul_pd(ymma, ymmb);
        ymmd = _mm256_add_pd(ymmd, ymmc);
        _mm256_store_pd(d+i*4,ymmd);
    }
    for(i = N-N%4; i < N; i++){
        d[i] = a[i] * b[i] + c[i];
    }
}

int main(){
    double* a; 
    double* b; 
    double* c; 
    double* d;

    a = (double*)(aligned_alloc(32,8192*sizeof(double)));
    b = (double*)(aligned_alloc(32,8192*sizeof(double)));
    c = (double*)(aligned_alloc(32,8192*sizeof(double)));
    d = (double*)(aligned_alloc(32,8192*sizeof(double)));

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

    for(i = 0; i < 1000000; i++){
        muladd(a, b, c, d, 8192);
    }

    stop = clock();
    elapsed = (double)(stop-start) / CLOCKS_PER_SEC;
    printf("Elapsed time = %8.6f s\n", elapsed);
    for(i = 0; i < 8192; i++){
        if(i % 1001 == 0){
            printf("%5d: %16.8f * %16.8f + %16.8f = %16.8f (%d)\n",
                    i, a[i], b[i], c[i], d[i], d[i]==a[i]*b[i]+c[i]);
        }
    }

    free(a);
    free(b);
    free(c);
    free(d);
}

