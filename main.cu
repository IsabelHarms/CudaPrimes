#include <cstdio>
#include <cstdint>
#include <cassert>
#include <chrono>

#include <cuda_runtime.h>


const int SectionSize = 10000;
const int SectionCount = 100000; // 250 or 256 ?
const int  length = SectionCount * SectionSize;
// 10 000 000: count = 664579

__global__ static void primeCountParallelSections(bool *primes) // count primes in section
{
    unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * SectionSize;
    unsigned stop = start + SectionSize;
    for (unsigned test = start; test < stop; test++) // is "test" a prime number?
    {
        primes[test] = true;
        for (int factor = 2; factor * factor <= test; factor++) // stop at square root
        {
            if (test % factor == 0) {
                primes[test] = false;
                break;
            }
        }
    }
}

__global__ static void primeCountParallel(bool *primes) // count primes in section
{
    unsigned number = (blockIdx.x * blockDim.x + threadIdx.x);
    for (int factor = 2; factor * factor <= number; factor++) {
        if (number % factor == 0) {
            primes[number] = false;
            break;
        }
    }
}

static void primeCount(bool *primes)
{
    for (int number = 2; number < length; number++) {
        for (int factor = 2; factor * factor <= number; factor++) {
            if (number % factor == 0) {
                primes[number] = false;
                break;
            }
        }
    }
}

static void primeCountOptimized(bool *primes)
{
    for (int number = 2; number < length; number++) {
            for (int factor = 2; factor * factor <= number; factor++) {
                if (primes[factor] && number % factor == 0) {
                    primes[number] = false;
                    break;
                }
            }
    }
}

__global__ static void primeSieveParallelSections(bool *primes)
{
    unsigned start = (blockIdx.x * blockDim.x + threadIdx.x) * SectionSize;
    unsigned stop = start + SectionSize;

    for (int factor = 2; 2*factor<=stop; factor++)
    {
        unsigned multiple = start / factor;
        if (multiple < 2) multiple = 2;
        unsigned i = multiple * factor;
        if (i < start) i += factor; // eg. start at 10: 10/3 = 3; 3*3 = 9; --> start at 12

        while (i < stop)
        {
            primes[i] = false;
            i += factor;
        }
    }
}

__global__ static void primeSieveParallel(bool *primes)
{
    unsigned factor = threadIdx.x + blockIdx.x * blockDim.x;
    if (factor >= 2) {
        for (unsigned i = factor+factor; i<length; i+= factor)
        {
            primes[i] = false;
        }
    }
}

static void primeSieve(bool *primes) // count primes in section
{
    for (int factor = 2; factor < length/2; factor++) {
        for (int i = factor+factor; i<length; i+= factor)
        {
            primes[i] = false;
        }
    }
}

static void primeSieveOptimized(bool *primes) // count primes in section
{
    for (int factor = 2; factor < length/2; factor++) {
        if (primes[factor])
        {
            int multiple = 2 * factor;
            while (multiple < length) { primes[multiple] = false; multiple += factor; }
        }
    }
}

int main()
{
    auto begin = std::chrono::high_resolution_clock::now();
    int count = 0;
    bool *h_primes;
    h_primes = (bool*)malloc(length);
    for (int i = 0; i < length; i++) h_primes[i] = true;

    int deviceId;
    int numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    size_t threadsPerBlock;
    size_t numberOfBlocks;
    threadsPerBlock = 100;
    numberOfBlocks = length/100;//32 * numberOfSMs; //length/threadsPerBlock;//

    bool *d_primes;
    cudaMalloc( (void **) &d_primes, length);
    cudaMemcpy(d_primes, h_primes, length, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //primeSieveOptimized(h_primes);
    primeCountParallel<<<numberOfBlocks,threadsPerBlock,0,stream>>>(d_primes);

    cudaDeviceSynchronize();
    cudaMemcpy( h_primes, d_primes, length, cudaMemcpyDeviceToHost);
    cudaStreamDestroy(stream);
    cudaFree(d_primes);

    h_primes[0] = h_primes[1] = false;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    for (int i = 0; i < SectionSize * SectionCount; i++) {
       if (h_primes[i]) count++;
    }
    printf("%d, %.3f",count, elapsed.count() * 1e-9);

    free(h_primes);
}

/*
 *
    //primeCount(h_primes);
 *   //primeSieve(h_primes);
    //
    //primeSieveOptimized(h_primes);
    //primeSieveParallel<<<length/2, 1, 0, stream>>>(d_primes);




 * */
