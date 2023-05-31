#include <cstdio>
#include <cstdint>
#include <cassert>

#include <cuda_runtime.h>


const int SectionSize = 100000;
const int SectionCount = 10; // 250 or 256 ?
// 10 000 000: count = 664579

__global__ static void primeCount(int sectionNr, float *primes) // count primes in section
{
    unsigned int start_time = 0, stop_time = 0;
    start_time = clock();
    int count = 0;
    int start = sectionNr * SectionSize;
    int stop = start + SectionSize;
    if (start < 2) start = 2;

    for (int test = start; test < stop; test++) // ist "test" a prime number?
    {
        bool prime = true;
        primes[test] = 1;
        for (int factor = 2; factor * factor <= test; factor++) // stop at square root
        {
            if (test % factor != 0) continue;
            prime = false; // devider found!
            primes[test] = 0;
            break;
        }
        if (prime) count++;
    }
    stop_time = clock();
    printf("section %d: %d, time: %d ms\n", sectionNr, count, stop_time-start_time);
}

__global__ static void primeSieve(int sectionNr)
{
    unsigned int start_time = 0, stop_time = 0;
    start_time = clock();
    int start = sectionNr * SectionSize;
    int stop = start + SectionSize;

    bool prime[SectionSize];
    // stack data in c and c++ must be initialized!
    for (int i = 0; i < SectionSize; i++) prime[i] = true;
    if (sectionNr == 0) prime[0] = prime[1] = false;

    for (int factor = 2; 2*factor<stop; factor++)
    {
        int multiple = start / factor;
        if (multiple < 2) multiple = 2;
        int i = multiple * factor;
        if (i < start) i += factor; // eg. start at 10: 10/3 = 3; 3*3 = 9; --> start at 12

        while (i < stop)
        {
            prime[i - start] = false;
            i += factor;
        }
    }
    int count = 0;
    for (int i = 0; i < SectionSize; i++) {
        if (prime[i]) count++;
    }
    stop_time = clock();
    printf("section %d: %d, time: %d ms\n", sectionNr, count, stop_time-start_time);
}

int main()
{
    int size = SectionSize * SectionCount * sizeof(float);

    float *h_primes;
    float *d_primes;

    h_primes = (float*)malloc(size);
    cudaMalloc( (void **) &d_primes, size);
    cudaMemcpy(d_primes, h_primes, size, cudaMemcpyHostToDevice);

    int count = 0;

    for (int section = 0; section < SectionCount; section++)
    {
        //count += PrimeSieve(section);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        primeSieve<<<1, 1, 0, stream>>>(section);
        //primeCount<<<1, 1, 0, stream>>>(section, d_primes);
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize();
    cudaMemcpy( h_primes, d_primes, size, cudaMemcpyDeviceToHost);
    cudaFree(d_primes);
    free(h_primes);
}
