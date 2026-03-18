#include "kernel.cuh"

#define CUDA_MAGNITUDE_CUTOFF 4

// Good explanation
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#include <chrono>
using namespace std::chrono;

#define THR 24

__global__ void asyncMandel(CudaArgs* args, int border, unsigned long* result, unsigned long* fastBuf)
{
    unsigned long n;
    double p, q, r, i, prev_r, prev_i;

    unsigned long cols[4];

    const bool fastPass = !args->slow && !args->last;
    const int renderWidth = fastPass ? (args->scrwidth + 1) / 2 : args->scrwidth;
    const int renderHeight = fastPass ? (args->scrheight + 1) / 2 : args->scrheight;
    const unsigned int fastIdx = blockIdx.y * gridDim.x + blockIdx.x;

    int ni = 0;
    int nj = 0;
    
    if (border == 0)
    {        
        switch (threadIdx.y)
        {
        case 0:
            ni = blockIdx.x * THR + threadIdx.x;
            nj = blockIdx.y * THR;
            break;
        case 1:
            ni = blockIdx.x * THR;
            nj = blockIdx.y * THR + THR - 1 - threadIdx.x;
            break;
        }
    }
    else if (border == 1)
    {
        unsigned long col = fastBuf[fastIdx];
        if ((col & 0xFF000000) == 0)
        {            
            return;
        }
        ni = blockIdx.x * THR + threadIdx.x + 1;
        nj = blockIdx.y * THR + threadIdx.y + 1;

        if (ni >= renderWidth || nj >= renderHeight)
        {
            return;
        }

        if ((col & 0xFF000000) == 0x01000000)
        {
            unsigned long fillCol = col & 0x00FFFFFF;
            if (fastPass)
            {
                int dstX = ni * 2;
                int dstY = nj * 2;
                for (int y = 0; y < 2; ++y)
                {
                    int outY = dstY + y;
                    if (outY >= args->scrheight)
                    {
                        continue;
                    }
                    for (int x = 0; x < 2; ++x)
                    {
                        int outX = dstX + x;
                        if (outX < args->scrwidth)
                        {
                            result[args->scrwidth * outY + outX] = fillCol;
                        }
                    }
                }
            }
            else
            {
                result[args->scrwidth * nj + ni] = fillCol;
            }
            return;
        }
    }
    else if (border == 2)
    {
        ni = blockIdx.x * THR + threadIdx.x;
        nj = blockIdx.y * THR + threadIdx.y;
    }

    if (ni >= renderWidth || nj >= renderHeight)
    {
        return;
    }

    int colno = 0;
    int aa = (args->last && args->aa) ? 1 : 0;
    for (int x = 0; x <= aa % 2; ++x)
    {
        p = (((double)ni + x / 2.0f) * args->width / renderWidth) + args->xmin;

        for (int y = 0; y <= aa; ++y)
        {
            q = (((double)nj + y / 2.0f) * args->height / renderHeight) + args->ymin;

            prev_i = 0.0f;
            prev_r = 0.0f;

            for (n = 0; n < args->iterations; n++)
            {
                r = (prev_r * prev_r) - (prev_i * prev_i) + p;
                i = 2 * (prev_r * prev_i) + q;

                if (r * r + i * i >= CUDA_MAGNITUDE_CUTOFF)
                {
                    break;
                }

                prev_r = r;
                prev_i = i;
            }

            cols[colno] = 0;

            if (n < args->iterations)
            {
                n = n % 128;

                if (args->useCustomPalette)
                {
                    cols[colno] = args->palette[n];
                }
                else if (n < 32)
                {
                    cols[colno] = 0x000008 * n;
                }
                else if (n < 64)
                {
                    cols[colno] = (0x080800 * (n - 32)) | 0xff;
                }
                else if (n < 96)
                {
                    cols[colno] = (0x08 * (95 - n)) | 0xffff00;
                }
                else
                {
                    cols[colno] = (0x080800 * (127 - n));
                }
            }
            colno++;
        }
    }

    if (aa)
    {
        cols[0] =
            ((((cols[0] & 0xff0000) + (cols[1] & 0xff0000) + (cols[2] & 0xff0000) + (cols[3] & 0xff0000)) / 4) & 0xff0000) |
            ((((cols[0] & 0xff00) + (cols[1] & 0xff00) + (cols[2] & 0xff00) + (cols[3] & 0xff00)) / 4) & 0xff00) |
            ((((cols[0] & 0xff) + (cols[1] & 0xff) + (cols[2] & 0xff) + (cols[3] & 0xff)) / 4) & 0xff);
    }

    if (fastPass)
    {
        int dstX = ni * 2;
        int dstY = nj * 2;
        for (int y = 0; y < 2; ++y)
        {
            int outY = dstY + y;
            if (outY >= args->scrheight)
            {
                continue;
            }
            for (int x = 0; x < 2; ++x)
            {
                int outX = dstX + x;
                if (outX < args->scrwidth)
                {
                    result[args->scrwidth * outY + outX] = cols[0];
                }
            }
        }
    }
    else
    {
        result[args->scrwidth * nj + ni] = cols[0];
    }

    if (border == 0)
    {       
        {
            unsigned int idx = fastIdx;
            if ((fastBuf[idx] & 0xFF000000) == 0)
            {
                fastBuf[idx] = 0x01000000 | cols[0];
            }
            else if ((fastBuf[idx] & 0xFF000000) == 0x01000000 && (fastBuf[idx] & 0x00FFFFFF) != cols[0])
            {
                fastBuf[idx] = 0x02000000;
            }
        }
        if (threadIdx.y == 0 && blockIdx.y > 0)
        {
            unsigned int idx = (blockIdx.y - 1) * gridDim.x + blockIdx.x;
            if ((fastBuf[idx] & 0xFF000000) == 0)
            {
                fastBuf[idx] = 0x01000000 | cols[0];
            }
            else if ((fastBuf[idx] & 0xFF000000) == 0x01000000 && (fastBuf[idx] & 0x00FFFFFF) != cols[0])
            {
                fastBuf[idx] = 0x02000000;
            }
        }
        if (threadIdx.y == 1 && blockIdx.x > 0)
        {
            unsigned int idx = blockIdx.y * gridDim.x + blockIdx.x - 1;
            if ((fastBuf[idx] & 0xFF000000) == 0)
            {
                fastBuf[idx] = 0x01000000 | cols[0];
            }
            else if ((fastBuf[idx] & 0xFF000000) == 0x01000000 && (fastBuf[idx] & 0x00FFFFFF) != cols[0])
            {
                fastBuf[idx] = 0x02000000;
            }
        }
    }
}

long GetRandomColor()
{
    long color;
    float red = ((float)rand() / RAND_MAX) * 0.99f + 0.01f;
    float green = ((float)rand() / RAND_MAX) * 0.99f + 0.01f;
    float blue = ((float)rand() / RAND_MAX) * 0.99f + 0.01f;
    
    float max = red > green ? red : green;
    max = max > blue ? max : blue;

    red /= max;
    green /= max;
    blue /= max;

    // reduce those aggressive pinks and lime greens 
    green = red * 0.35f * green * 0.3f + blue * 0.35f;
    
    return 
        ((red >= 1.0 ? 255 : (int)floor(red * 256.0))<<16) |
        ((green >= 1.0 ? 255 : (int)floor(green * 256.0)) << 8) |
        (blue >= 1.0 ? 255 : (int)floor(blue * 256.0));

    //srand((unsigned)time(NULL));
}
 

void cudaMandel(CudaArgs *args, unsigned char* lpBitmapBits)
{
    auto t1 = high_resolution_clock::now();

    // Image buffer
    unsigned long outputSize = sizeof(unsigned long) * args->scrwidth * args->scrheight;
    unsigned long* cudaResult = 0;
    cudaMalloc(&cudaResult, outputSize);

    //cudaMemset(&cudaResult, 0, outputSize);

    CudaArgs* gpuArgs;
    cudaMalloc((void**)&gpuArgs, sizeof(CudaArgs));
    cudaMemcpy(gpuArgs, args, sizeof(CudaArgs), cudaMemcpyHostToDevice);

    const bool fastPass = !args->slow && !args->last;
    int renderWidth = fastPass ? (args->scrwidth + 1) / 2 : args->scrwidth;
    int renderHeight = fastPass ? (args->scrheight + 1) / 2 : args->scrheight;
    dim3 numBlocks((renderWidth + THR - 1) / THR, (renderHeight + THR - 1) / THR);

    // Use direct full-tile kernel for both passes; fast pass differs by reduced resolution only.
    dim3 threadsPerBlock2(THR, THR);
    asyncMandel <<< numBlocks, threadsPerBlock2 >>> (gpuArgs, 2, cudaResult, nullptr);


    cudaMemcpy(lpBitmapBits, cudaResult, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(cudaResult);
    cudaFree(gpuArgs);

    auto t2 = high_resolution_clock::now();

    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    char debug[32];
    sprintf(debug, "Time: %0.2f ms\n", fp_ms.count());
    OutputDebugString(debug);
}


/*
__global__ void asyncMandel(int _y, CudaArgs* args, int maxIter, double width, double height, double xmin, double ymin, unsigned long* result)
{
    unsigned long n;
    double p, q, r, i, prev_r, prev_i;

    unsigned long cols[4];

    int colno = 0;
    for (int x = 0; x <= args->aa; ++x)
    {
        p = (((double)threadIdx.x + x / 2.0f) * width / args->scrwidth) + xmin;

        for (int y = 0; y <= args->aa; ++y)
        {
            q = (((double)(_y + blockIdx.x) + y / 2.0f) * height / args->scrheight) + ymin;

            prev_i = 0.0f;
            prev_r = 0.0f;

            for (n = 0; n < maxIter; n++)
            {
                r = (prev_r * prev_r) - (prev_i * prev_i) + p;
                i = 2 * (prev_r * prev_i) + q;

                if (r * r + i * i >= CUDA_MAGNITUDE_CUTOFF)
                {
                    break;
                }

                prev_r = r;
                prev_i = i;
            }

            cols[colno] = 0;

            if (n < maxIter)
            {
                n = n % 128;

                if (n < 32)
                {
                    cols[colno] = 0x000008 * n;
                }
                else if (n < 64)
                {
                    cols[colno] = (0x080800 * (n - 32)) | 0xff;
                }
                else if (n < 96)
                {
                    cols[colno] = (0x08 * (95 - n)) | 0xffff00;
                }
                else
                {
                    cols[colno] = (0x080800 * (127 - n));
                }
            }
            colno++;
        }
    }

    if (args->aa)
    {
        result[threadIdx.x + blockIdx.x * args->scrwidth] =
            ((((cols[0] & 0xff0000) + (cols[1] & 0xff0000) + (cols[2] & 0xff0000) + (cols[3] & 0xff0000)) / 4) & 0xff0000) |
            ((((cols[0] & 0xff00) + (cols[1] & 0xff00) + (cols[2] & 0xff00) + (cols[3] & 0xff00)) / 4) & 0xff00) |
            ((((cols[0] & 0xff) + (cols[1] & 0xff) + (cols[2] & 0xff) + (cols[3] & 0xff)) / 4) & 0xff);
    }
    else
    {
        result[threadIdx.x + blockIdx.x * args->scrwidth] = cols[0];
    }
}

void cudaMandel(int WIDTH, int HEIGHT, int iterations, double xmin, double xmax, double ymin, double ymax, int antialias,
    unsigned char* lpBitmapBits)
{
    auto t1 = high_resolution_clock::now();

    int block = 240;
    unsigned long rowSize = sizeof(unsigned long) * WIDTH * block;

    unsigned long* cudaResult = 0;

    cudaMalloc(&cudaResult, rowSize);

    int numBytes = sizeof(CudaArgs);
    CudaArgs* cpuCudaArgs = (CudaArgs*)malloc(numBytes);
    CudaArgs* gpuPointArray;
    cudaMalloc((void**)&gpuPointArray, numBytes);

    cpuCudaArgs->scrheight = HEIGHT;
    cpuCudaArgs->scrwidth = WIDTH;
    cpuCudaArgs->aa = antialias;

    cudaMemcpy(gpuPointArray, cpuCudaArgs, numBytes, cudaMemcpyHostToDevice);

    for (int y = 0; y < HEIGHT/block; y++)
    {
        // Execute kernel
        asyncMandel <<< block, WIDTH >>> (y*block, gpuPointArray, iterations, (xmax - xmin), (ymax - ymin), xmin, ymin, cudaResult);
        cudaMemcpy(lpBitmapBits + y* rowSize, cudaResult, rowSize, cudaMemcpyDeviceToHost);
    }

    auto t2 = high_resolution_clock::now();

    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    char debug[32];
    sprintf(debug, "Time: %0.2f ms\n", fp_ms.count());
    OutputDebugString(debug);
}

*/ 