#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <threads.h>

struct Input{
    double* a;
    double* b;
    double* c;
    double* d;
    unsigned long long N;
    unsigned long long R;
};

__attribute__ ((noinline))
int muladd_th(void* input){
    struct Input* in;
    unsigned long long i, j;
    double* va;
    double* vb;
    double* vc;
    double* vd;
    in = (struct Input*)input;
    va = in->a;
    vb = in->b;
    vc = in->c;
    vd = in->d;
    for(i = 0; i < in->R; i++){
        for(j = 0; j < in->N; j++){
            vd[j] = va[j] * vb[j] + vc[j];
        }
    }
    return 0;
}

__attribute__ ((noinline))
void muladd(double* a, double* b, double* c, double* d,
            unsigned long long N, unsigned long long R){
    thrd_t threads[4];
    struct Input inputs[4];
    unsigned long long i;
    for(i = 0; i < 4; i++){
        inputs[i].a = a + N/4*i;
        inputs[i].b = b + N/4*i;
        inputs[i].c = c + N/4*i;
        inputs[i].d = d + N/4*i;
        inputs[i].N = N/4;
        inputs[i].R = R;
        if(i == 3){
            inputs[i].N = N-N/4*3;
        }
        thrd_create(&(threads[i]), muladd_th, &(inputs[i]));
    }
    for(i = 0; i < 4; i++){
        thrd_join((threads[i]), NULL);
    }
    
}

int main(){
    double* a = (double*)(malloc(8192*sizeof(double)));
    double* b = (double*)(malloc(8192*sizeof(double)));
    double* c = (double*)(malloc(8192*sizeof(double)));
    double* d = (double*)(malloc(8192*sizeof(double)));
    
    //Prepare data
    unsigned long long i;
    for(i = 0; i < 8192; i++){
        a[i] = (double)(rand()%2000) / 200.0;
        b[i] = (double)(rand()%2000) / 200.0;
        c[i] = ((double)i)/10000.0;
    }
    
    struct timespec start, stop;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    muladd(a, b, c, d, 8192, 1000000);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed = (double)(stop.tv_sec-start.tv_sec);
    elapsed += (double)(stop.tv_nsec-start.tv_nsec)/ 1000000000.0;
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

