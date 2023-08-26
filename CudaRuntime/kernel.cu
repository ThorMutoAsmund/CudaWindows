#include "kernel.cuh"

#define CUDA_MAGNITUDE_CUTOFF 4

// Good explanation
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#include <chrono>
using namespace std::chrono;

__global__ void asyncMandel(CudaArgs* args, double width, double height, double xmin, double ymin, unsigned long* result)
{
    unsigned long n;
    double p, q, r, i, prev_r, prev_i;

    unsigned long cols[4];

    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    int nj = blockIdx.y * blockDim.y + threadIdx.y;

    int colno = 0;
    for (int x = 0; x <= args->aa; ++x)
    {
        p = (((double)ni + x / 2.0f) * width / args->scrwidth) + xmin;

        for (int y = 0; y <= args->aa; ++y)
        {

            q = (((double)nj + y / 2.0f) * height / args->scrheight) + ymin;

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
        result[args->scrwidth * nj + ni] =
            ((((cols[0] & 0xff0000) + (cols[1] & 0xff0000) + (cols[2] & 0xff0000) + (cols[3] & 0xff0000)) / 4) & 0xff0000) |
            ((((cols[0] & 0xff00) + (cols[1] & 0xff00) + (cols[2] & 0xff00) + (cols[3] & 0xff00)) / 4) & 0xff00) |
            ((((cols[0] & 0xff) + (cols[1] & 0xff) + (cols[2] & 0xff) + (cols[3] & 0xff)) / 4) & 0xff);
    }
    else
    {
        result[args->scrwidth * nj + ni] = cols[0];
    }
}

void cudaMandel(CudaArgs *args, double xmin, double xmax, double ymin, double ymax,
    unsigned char* lpBitmapBits)
{
    auto t1 = high_resolution_clock::now();

    unsigned long outputSize = sizeof(unsigned long) * args->scrwidth * args->scrheight;
    unsigned long* cudaResult = 0;

    cudaMalloc(&cudaResult, outputSize);

    int numBytes = sizeof(CudaArgs);

    CudaArgs* gpuArgs;
    cudaMalloc((void**)&gpuArgs, numBytes);
    cudaMemcpy(gpuArgs, args, numBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(args->scrwidth / threadsPerBlock.x, args->scrheight / threadsPerBlock.y);

    // Execute kernel
    asyncMandel <<< numBlocks, threadsPerBlock >>> (gpuArgs, (xmax - xmin), (ymax - ymin), xmin, ymin, cudaResult);
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