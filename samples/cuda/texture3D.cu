#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void FetchFrom1DTexture(cudaTextureObject_t tex,
                                   float3* position,
                                   float2* result,
                                   unsigned int N){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        //拾取
        //tex3D的后三个参数分别为第一，第二和第三个维度上的坐标
        result[id] = tex3D<float2>(tex, position[id].x,
                                        position[id].y,
                                        position[id].z);
    }
}

int main(){
    //我们要使用的纹理的原始值
    float2* resource = (float2*)(malloc(8*sizeof(float2)));
    for(int i = 0; i < 8; i++){
        resource[i].x = 1.0 + (float)i;
        resource[i].y = -2.0 - 2.0 * (float)i;
        //顺便一提：
        //float2的两个分量是x和y
        //float3的三个分量是x，y和z
        //float4的四个分量是x，y，z和w
    }
    
    //创建一个CUDA Array
    //cudaChannelFormatDesc: 
    //  描述Array里每个元素的样子。这里每个元素是float2。
    cudaChannelFormatDesc floatChannelDesc
                       = cudaCreateChannelDesc<float2>();
    
    //cudaExtent是用来描述array的形状的
    //三个参数分别是宽(x)，高(y)，和深(z)
    cudaExtent ext = make_cudaExtent(2,2,2);
    //声明CUDA Array
    cudaArray_t cuArray;
    //为cuArray分配空间。
    cudaMalloc3DArray(&cuArray, &floatChannelDesc, ext);

    //由于三维拷贝参数比较复杂
    //所以CUDA设计了一个类型用于表达参数
    cudaMemcpy3DParms cpy3d={0};
    //纹理来源于resource数组
    cpy3d.srcPtr.ptr = resource;
    //来源数组中每两个float2算作一行
    cpy3d.srcPtr.pitch = 2*sizeof(float2);
    //来源数组每行取两个元素
    cpy3d.srcPtr.xsize = 2;
    //来源数组每两行组成一层
    cpy3d.srcPtr.ysize = 2;
    //拷贝的目的地是cuArray数组
    cpy3d.dstArray = cuArray;
    //目标里面在三个维度上各有几个元素，也是用cudaExtent
    //这里要复制到整个array，所以重用了前面定义的ext
    cpy3d.extent = ext;
    //从主机内存拷贝到显存
    cpy3d.kind = cudaMemcpyHostToDevice;
    //三维拷贝也可以像二维拷贝一样设置起点，这里就不详述了
    //可以参考cuda Toolkit 11.0.3的文档里
    //cuda runtime API中5.9节关于cudaMemcpy3D的说明
    cudaMemcpy3D(&cpy3d);
    

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    //这回我们使用“边界”模式
    //因为是三维纹理，所以三个都要设置
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.addressMode[2]   = cudaAddressModeBorder;

    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    //这次不做归一化
    texDesc.normalizedCoords = 0;
    
    //声明并创建纹理
    cudaTextureObject_t tex3DObj;
    cudaCreateTextureObject(&tex3DObj, &resDesc, &texDesc, NULL);
    
    //我们想要拾取的纹理的坐标。
    float3* position = (float3*)(malloc(64*sizeof(float3)));
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
                int id = i + j*4 + k*16;
                position[id].x = (float)i - 0.5;
                position[id].y = (float)j - 0.5;
                position[id].z = (float)k - 0.5;
            }
        }
    }
    
    float2* result = (float2*)(malloc(64*sizeof(float2)));
    float3* cuposition;
    float2* curesult;
    cudaMalloc(&cuposition, 64*sizeof(float3));
    cudaMalloc(&curesult, 64*sizeof(float2));
    cudaMemcpy(cuposition, position, 64*sizeof(float3),
               cudaMemcpyHostToDevice);
    
    FetchFrom1DTexture<<<1, 64>>>(tex3DObj, cuposition, curesult, 64);
    
    cudaMemcpy(result, curesult, 64*sizeof(float2),
               cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
                int id = i + j*4 + k*16;
                printf("Texel at position (%- 5.3f, %- 5.3f, %- 5.3f)"
                       " is (%- 5.3f, %- 5.3f) \n",
                        position[id].x, position[id].y, position[id].z,
                        result[id].x, result[id].y);
            }
        }
    }
    cudaFree(curesult);
    cudaFree(cuposition);
    free(result);
    free(position);
    free(resource);
    cudaDestroyTextureObject(tex3DObj);
    cudaFreeArray(cuArray);
    
}
