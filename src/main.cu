#include "main.h"
#include <cuda.h>

__global__ void kernel()
{
}

void foo()
{
    kernel<<< dim3(16,1,1), dim3(1,1,1) >>> ();
}
