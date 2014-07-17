#ifndef CUDALIB_H
#define CUDALIB_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

inline void cudaHandleError(cudaError_t err, const char* filename, int line, const char* funcname)
{
    if (err == cudaSuccess)
        return;
    std::cerr << "CUDA error: " << err << std::endl;
    std::cerr << "At: " << filename << " : " << line << " : " << funcname << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    abort();
}

#define cutilSafeCall(x) cudaHandleError((x), __FILE__, __LINE__, __FUNCTION__)

#endif // CUDALIB_H
