#pragma once

#include <windows.h>

#include "CudaEx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct
{
    int scrwidth;
    int scrheight;
    int aa;
    int iterations;
} CudaArgs;


void cudaMandel(CudaArgs *args, double xmin, double xmax, double ymin, double ymax, 
    unsigned char* lpBitmapBits);