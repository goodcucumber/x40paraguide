#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

__attribute__ ((noinline))
void muladd(double* a, double* b, double* c, double* d,
            unsigned long long N){
    unsigned long long i;
    __asm__ __volatile__(
            "movq %0, %%rax \n\t"
            "movq %1, %%rbx \n\t"
            "movq %2, %%rcx \n\t"
            "movq %3, %%rdx \n\t"
            "movq %4, %%r8  \n\t"
            "shr  $2, %%r8  \n\t"
            "movq $0, %%r9  \n\t"
            "jmp  .check_%= \n\t"
            ".loop_%=:         \n\t"
            "shl $2, %%r9   \n\t"
            "vmovupd (%%rax, %%r9, 8), %%ymm0\n\t"
            "vmovupd (%%rbx, %%r9, 8), %%ymm1\n\t"
            "vmovupd (%%rcx, %%r9, 8), %%ymm2\n\t"
            "vmulpd %%ymm0, %%ymm1, %%ymm3 \n\t"
            "vaddpd %%ymm2, %%ymm3, %%ymm3 \n\t"
            "vmovupd %%ymm3, (%%rdx, %%r9, 8)\n\t"
            "shr $2, %%r9                  \n\t"
            "add $1, %%r9                  \n\t"
            ".check_%=:                    \n\t"
            "cmpq %%r8, %%r9               \n\t"
            "jl .loop_%=                   \n\t"
            :
            :"m"(a), "m"(b), "m"(c), "m"(d), "m"(N)
            :"%rax", "%rbx", "%rcx", "%rdx", "%r8", "%r9",
             "%ymm0", "%ymm1", "%ymm2", "%ymm3", "memory"
            );
    if(N%4!=0){
        for(i = N-N%4; i<N; i++){
            d[i] = a[i]*b[i]+c[i];
        }
    }
}

int main(){
    double* a; 
    double* b; 
    double* c; 
    double* d;

    a = (double*)(malloc(8192*sizeof(double)));
    b = (double*)(malloc(8192*sizeof(double)));
    c = (double*)(malloc(8192*sizeof(double)));
    d = (double*)(malloc(8192*sizeof(double)));

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

