#include <cuda.h>

__device__ __inline__ int get1DGlobalIdx()
{
    return blockIdx.x*blockDim.x+threadIdx.x;
}

template<typename T>
inline T inc_div(T a, T b)
{
    return (a+b-1) / b;
}
