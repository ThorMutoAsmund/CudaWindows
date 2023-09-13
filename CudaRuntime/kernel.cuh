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
    int slow;
    double width;
    double height;
    double xmin;
    double ymin;
    int last;
} CudaArgs;


void cudaMandel(CudaArgs *args, unsigned char* lpBitmapBits);