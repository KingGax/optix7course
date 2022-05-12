#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
__global__ void calculatePhysics()
{

    printf("hii");
}

extern "C" __host__ void run_Calculate_Physics()
{
    calculatePhysics<<<1,1>>>();
}