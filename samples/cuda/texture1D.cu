#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void FetchFrom1DTexture(cudaTextureObject_t tex,
                                   float* position,
                                   float* result,
                                   unsigned int N){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){
        //拾取
        //<float>是c++模板
        //这里不详细解释模板了，总之在尖括号里填写纹理的类型就对了
        //第一个参数是一个texture object (cudaTextureObject_t)
        //第二个参数是要拾取的位置。
        result[id] = tex1D<float>(tex, position[id]);
    }
}

int main(){
    //我们要使用的纹理的原始值
    float* resource = (float*)(malloc(4*sizeof(float)));
    resource[0] = 1.0;
    resource[1] = 2.0;
    resource[2] = 3.0;
    resource[3] = 4.0;
    
    //创建一个CUDA Array
    
    //cudaChannelFormatDesc: 
    //  描述Array里每个元素的样子。这里每个元素是float。
    cudaChannelFormatDesc floatChannelDesc
                      = cudaCreateChannelDesc<float>();
    //声明CUDA Array
    cudaArray_t cuArray;
    
    //为cuarray分配空间。第三个参数是array在第一个维度上的长度。
    //这里长度单位是“个”，
    //因为在floatChannelDesc里已经描述了每个元素的大小了。
    //cudaMallocArray可以分配一维或二维的array
    //第四个参数用于指定第二个维度的长度（单位是多少行(háng)），默认为0
    cudaMallocArray(&cuArray, &floatChannelDesc, 4);
    
    //向CUDA Array中拷贝数据。
    //cudaMemcpy2DToArray用于向一维和二维Array中拷贝数据。
    //一维Array就是第二个维度长度为0的二维Array
    //第一个参数是目标CUDA Array
    //第二个参数是目标的x位置
    //第三个参数是目标的y位置
    //第二个参数和第三个参数的意义在于可以只更新Array的一部分
    //第四个参数是被拷贝的数据的指针
    //第五个参数是描述被拷贝的内容按照每一行多少byte来排列
    //第六个参数是目标位置每一行拷贝多少byte
    //第七个参数是一共拷贝多少行
    //第八个参数是拷贝方向，这里是从主机内存到显卡内存。原因前面讲过。
    //可以通过后文的图片详细了解这个函数的工作方式。
    cudaMemcpy2DToArray(cuArray, 0, 0,  resource, 4*sizeof(float),
                        4*sizeof(float), 1, cudaMemcpyHostToDevice);
    //其实拷贝完以后主机内存里的resource已经没用了，可以现在就free掉。
    
    
    //描述要成为纹理的东西是个啥
    //resType设置为cudaResourceTypeArray，表明要从一个Array来构建纹理
    //res.array.array设置成我们刚刚准备好的Array
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    //描述我们要的纹理是个啥样子
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    //addressMode可取的值有：
    //  cudaAddressModeWrap
    //  cudaAddressModeClamp
    //  cudaAddressModeMirror
    //  cudaAddressModeBorder
    //详情见上文描述
    //addressMode是长度为3的数组，分别对应三个维度。
    //这里是一维所以只用第一个。
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    //filterMode可取的值有：
    //  cudaFilterModeLinear  线性插值模式
    //  cudaFilterModePoint   取最近点模式
    texDesc.filterMode       = cudaFilterModeLinear;
    //readMode可取的值有：
    //  cudaReadModeElementType  Array是什么类型拾取出来就是什么类型
    //  cudaReadModeNormalizedFloat  进行归一化
    //注意，只有8位或16位整数支持归一化，其他类型比如int/float则不支持。
    texDesc.readMode         = cudaReadModeElementType;
    //是否对坐标归一化到[0,1). 1表示进行归一化。
    texDesc.normalizedCoords = 1;
    
    //声明并创建纹理
    //这里相当于将纹理绑定到之前创建的cuArray
    //所以不能释放掉cuArray
    cudaTextureObject_t tex1DObj;
    cudaCreateTextureObject(&tex1DObj, &resDesc, &texDesc, NULL);
    
    //我们想要拾取的纹理的坐标。
    float* position = (float*)(malloc(60*sizeof(float)));
    for(int i = 0; i < 60; i++){
        position[i] = -1.0 + (float)(i)/20.0;
    }
    
    float* result = (float*)(malloc(60*sizeof(float)));
    
    float* cuposition;
    float* curesult;
    cudaMalloc(&cuposition, 60*sizeof(float));
    cudaMalloc(&curesult, 60*sizeof(float));
    cudaMemcpy(cuposition, position, 60*sizeof(float),
               cudaMemcpyHostToDevice);
    
    FetchFrom1DTexture<<<1, 60>>>(tex1DObj, cuposition, curesult, 60);
    
    cudaMemcpy(result, curesult, 60*sizeof(float),
               cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < 60; i++){
        printf("Texel at position (%- 5.3f) is %5.3f \n",
               position[i], result[i]);
    }
    
    cudaFree(curesult);
    cudaFree(cuposition);
    
    free(result);
    free(position);
    free(resource);
    
    //不再使用的纹理要销毁掉
    cudaDestroyTextureObject(tex1DObj);
    //Array用完了也要释放
    cudaFreeArray(cuArray);
    
}
