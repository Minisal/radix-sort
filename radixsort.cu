#include "cuda_runtime.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>

#define WARP_SIZE 32
#define WARP_MASK 0xffffffff
#define NUM_RADICES 256

//#define DEBUG_HOST_IMPL
//#define DEBUG_DEVICE_IMPL
//#define VALIDATE_FULL
#define MEASURE_TIME_HOST
#define MEASURE_TIME_DEVICE

//  error handler for cuda's APIs
#define cudaErrHnd(result)                                                                              \
    if (result != cudaSuccess) {                                                                        \
        std::fprintf(stderr, "ERROR! exit with CUDA error code %d at line %d.", result, __LINE__);      \
        std::abort();                                                                                   \
    }                                                                                                   \

//  perform inclusive scan across warp lanes
__device__ void warpIncScan(unsigned int laneIdx, unsigned int& val)
{
    for (unsigned int delta = 1; delta <= WARP_SIZE / 2; delta <<= 1)
    {
        unsigned int neighbor = __shfl_up_sync(WARP_MASK, val, delta);
        if (laneIdx >= delta) val += neighbor;
    }
}

//  PHASE I : counting the radix (tabulation)
__global__ void countKernel(unsigned int* in_numbers, unsigned int* out_counters, 
    unsigned int numElements, unsigned int numBlocks, unsigned int pass
)
{
    //  identify thread's indices
    unsigned int inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdx = threadIdx.x >> 5;
    unsigned int laneIdx = threadIdx.x & 0x1f;

    //  initialize shared memory for counters across one thread block
    __shared__ unsigned int shm_radixCounters[WARP_SIZE][NUM_RADICES];
    for (unsigned int r = laneIdx; r < NUM_RADICES; r += WARP_SIZE)
        shm_radixCounters[warpIdx][r] = 0;
    __syncwarp();

    //  read inputs and increment the exist of found radix
    if (inputIdx < numElements)
    {
        unsigned int radixMask = 0x000000ff << 8 * pass;
        unsigned int number = in_numbers[inputIdx];
        unsigned int radix = (number & radixMask) >> 8 * pass;

        //  ToDo : use warp primitives to reduce shared memory size
        atomicAdd(&shm_radixCounters[warpIdx][radix], 1); 
    }
    __syncwarp();

    //  write radix count to the global counters table
    for (unsigned int r = laneIdx; r < NUM_RADICES; r += WARP_SIZE)
    {
        unsigned int count = shm_radixCounters[warpIdx][r];
        out_counters[r * numBlocks * WARP_SIZE + blockIdx.x * WARP_SIZE + warpIdx] = count;
    }
}

//  PHASE II.1 : radix summation (prefix-sum)
__global__ void radixSumScanKernel(unsigned int* inout_counters, unsigned int* out_totalSums, unsigned int numBlocks)
{
    //  identify warp and lane index
    unsigned int warpIdx = threadIdx.x >> 5;
    unsigned int laneIdx = threadIdx.x & 0x1f;

    //  idebtify assigned worked radix and pointer to the array of counters corresponding to the radix
    unsigned int radix = blockIdx.x;
    unsigned int* counters = inout_counters + radix * numBlocks * WARP_SIZE;

    //  initialize shared memory for one thread block
    __shared__ unsigned int shm_prefixSums[WARP_SIZE];
    if (warpIdx == 0) shm_prefixSums[laneIdx] = 0;
    __syncthreads();

    //  loop through a row of radix counters
    //  Note : the numbers of inputs always be a multiple of WARP_SIZE
    unsigned int numElements = numBlocks * WARP_SIZE,
        numRounds = (numElements + blockDim.x - 1) / blockDim.x;
    unsigned int inputIdx = threadIdx.x;
    for (unsigned int round = 0; round < numRounds; round++)
    {
        //  read corresponding input and add it with last offset calculated from last portion
        unsigned int count = (inputIdx < numElements) ? counters[inputIdx] : 0;
        unsigned int offset = shm_prefixSums[WARP_SIZE - 1];

        //  inclusive scan across lanes
        ::warpIncScan(laneIdx, count);

        __syncthreads();

        //  last lane of each warp record the sum of its warp to shared memory
        if (laneIdx == WARP_SIZE - 1)
            shm_prefixSums[warpIdx] = count;

        __syncthreads();

        //  only the first warp, scan all the sums in shared memory
        if (warpIdx == 0)
        {
            unsigned int highLevelCount = shm_prefixSums[laneIdx];

            ::warpIncScan(laneIdx, highLevelCount);

            shm_prefixSums[laneIdx] = highLevelCount;
        }

        __syncthreads();

        //  offseting
        if (warpIdx > 0)
            offset += shm_prefixSums[warpIdx - 1];
        count += offset;

        __syncthreads();

        //  offsetting back to shared prefixes
        if (warpIdx == 0)
            shm_prefixSums[laneIdx] += offset;

        if (inputIdx < numElements)
            counters[inputIdx] = count;

        inputIdx += blockDim.x;

        __syncthreads();
    }

    //  record total sum of this radix
    if (threadIdx.x == 0) out_totalSums[radix] = shm_prefixSums[WARP_SIZE - 1];
}

//  PHASE II.2 : prefix-sum on total sum array -> only NUM_RADICES threads required!
__global__ void totalSumScanKernel(unsigned int* inout_totalSums)
{
    unsigned int warpIdx = threadIdx.x >> 5;
    unsigned int laneIdx = threadIdx.x & 0x1f;

    //  initialize shared memory for one thread block
    __shared__ unsigned int shm_prefixSums[NUM_RADICES / WARP_SIZE];
    if (warpIdx == 0) shm_prefixSums[laneIdx] = 0;
    __syncthreads();

    //  load total sum of each radix
    unsigned int totalSum = inout_totalSums[threadIdx.x];

    //  inclusive scan across lanes
    ::warpIncScan(laneIdx, totalSum);

    //  last lane of each warp record the sum of its warp to shared memory
    if (laneIdx == WARP_SIZE - 1)
        shm_prefixSums[warpIdx] = totalSum;

    __syncthreads();
    
    //  only the first warp, scan all the sums in shared memory
    if (warpIdx == 0)
    {
        int predicate = laneIdx < NUM_RADICES / WARP_SIZE;
        unsigned int highLevelTotalSum = predicate ? shm_prefixSums[laneIdx] : 0;

        ::warpIncScan(laneIdx, highLevelTotalSum);

        if (predicate) shm_prefixSums[laneIdx] = highLevelTotalSum;
    }

    __syncthreads();

    //  offseting
    if (warpIdx > 0)
    {
        unsigned int offset = shm_prefixSums[warpIdx - 1];
        totalSum += offset;
    }

    //  save final scanned absolute sum to global memory
    inout_totalSums[threadIdx.x] = totalSum;
}

//  PHASE III : reordering
__global__ void reorderKernel(unsigned int* in_numbers, int* in_indices, 
    unsigned int* in_radixSums, unsigned int* in_totalSums,
    unsigned int* out_numbers, int* out_indices,
    unsigned int numElements, unsigned int numBlocks, unsigned int pass
)
{
    //  identify thread's indices
    unsigned int inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdx = threadIdx.x >> 5;
    unsigned int laneIdx = threadIdx.x & 0x1f;

    //  initialize shared memory for counters across one thread block
    __shared__ unsigned int shm_radixPrefix[WARP_SIZE][NUM_RADICES];
    for (unsigned int r = laneIdx; r < NUM_RADICES; r += WARP_SIZE)
    {
        //  load scanned radix sums and corresponding absolute sum from global memory
        unsigned int radixSum = in_radixSums[r * numBlocks * WARP_SIZE + blockIdx.x * WARP_SIZE + warpIdx];
        unsigned int offset = (r == 0) ? 0 : in_totalSums[r - 1];

        //  calculate local sum for each radix for this warp
        shm_radixPrefix[warpIdx][r] = radixSum + offset;
    }

    //  read inputs and increment the exist of found radix
    int predicate = inputIdx < numElements;
    unsigned int activeMask = __ballot_sync(WARP_MASK, predicate);
    if (predicate)
    {
        unsigned int radixMask = 0x000000ff << 8 * pass;
        unsigned int number = in_numbers[inputIdx];
        unsigned int index = in_indices[inputIdx];
        unsigned int radix = (number & radixMask) >> 8 * pass;

        //  check which lanes of all 32 lanes haave the same radix as this lane
        unsigned int matchMask = 0;
        for (unsigned int srcLane = 0; (0x1 << srcLane) & activeMask; srcLane++)
        {
            //  get mask of threads with the same input value as of srcLane
            unsigned int neighbor = __shfl_sync(activeMask, radix, srcLane);
            unsigned int vote = __ballot_sync(activeMask, neighbor == radix);
            
            //  if this is this thread(lane)'s check, save vote(match) result
            if ((0x1 << laneIdx) & vote) matchMask = vote;
        }

        //  calculate yields from local sum to place the input with stability
        unsigned int yield = 0;
        for (int i = WARP_SIZE - 1; i >= (int)laneIdx; i--)
        {
            if (matchMask & 0x80000000) yield++;
            matchMask <<= 1;
        }

        //  write the input to the right index of output
        unsigned int prefix = shm_radixPrefix[warpIdx][radix];
        out_numbers[prefix - yield] = number;
        out_indices[prefix - yield] = index;
    }
}

//  radixsort implementation in CPU
int radixsort_simple(unsigned int* numbers, int* indices, std::size_t numElements)
{
    //	for each round, count 4-bit each
    static unsigned int radixCounts[NUM_RADICES];

    //	allocate auxiliary buffer serving for sorting process
    unsigned int* auxNumbers = (unsigned int*)std::malloc(sizeof(unsigned int) * numElements);
    int* auxIndices = (int*)std::malloc(sizeof(int) * numElements);

#ifdef DEBUG_HOST_IMPL 
    std::cout << std::endl << "(H) :" << std::endl;
#endif

#ifdef MEASURE_TIME_HOST
    auto time_start = std::chrono::high_resolution_clock::now();
#endif

    unsigned int digitMask = 0x000000ffu;
    for (unsigned int d = 0; d < 4; d++)
    {
#ifdef DEBUG_HOST_IMPL 
        std::cout << "Pass [" << d << "]" << std::endl;
#endif

        //	reset radix counter
        std::memset(radixCounts, 0, sizeof(unsigned int) * NUM_RADICES);

        //	pick up in and out buffer for this round
        unsigned int* inNumbers, * outNumbers;
        int* inIndices, * outIndices;
        if (d % 2)
        {
            inNumbers = auxNumbers; inIndices = auxIndices;
            outNumbers = numbers; outIndices = indices;
        }
        else
        {
            inNumbers = numbers; inIndices = indices;
            outNumbers = auxNumbers; outIndices = auxIndices;
        }

        //	count
        for (int i = 0; i < numElements; i++)
        {
            int radix = (inNumbers[i] & digitMask) >> 8 * d;
            radixCounts[radix] += 1;
        }

        //	prefix-sum
        for (int i = 1; i < NUM_RADICES; i++)
            radixCounts[i] += radixCounts[i - 1];

        //	reorder
        //	indices array at first will act as output buffer 
        for (int i = numElements - 1; i >= 0; i--)
        {
            int radix = (inNumbers[i] & digitMask) >> 8 * d;
            unsigned int count = --radixCounts[radix]; // because index start with 0
            outNumbers[count] = inNumbers[i];
            outIndices[count] = inIndices[i];
        }
        digitMask <<= 8;

#ifdef DEBUG_HOST_IMPL 
        if (d == 3) break;
        std::cout << "Out for next pass [401:410]: ";
        for (int i = 401; i < 411; i++)
            std::cout << std::hex << ((outNumbers[i] & (0x000000ff << 8 * (d + 1))) >> 8 * (d + 1)) << " ";
        std::cout << std::dec << std::endl;
#endif
    }
    //	finally, the final output will be stored in input array themself

#ifdef MEASURE_TIME_HOST
    auto time_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
    std::cout << "Radixsort (H) exec time : " << duration.count() << "us." << std::endl;
#endif

    //	free auxiliary arrays
    std::free(auxNumbers);
    std::free(auxIndices);

    return 0;
}

#ifdef DEBUG_DEVICE_IMPL 
void radixInspect_count(unsigned int radix, unsigned int d, 
    unsigned int* numbers, unsigned int* radixCounters, 
    unsigned int numElements, unsigned int numBlocks)
{
    unsigned int* warpCounter = radixCounters;
    for (std::size_t i = 0; i < numElements; i++)
    {
        unsigned int input_radix = ((numbers[i] & (0x000000ff << 8 * d)) >> 8 * d);
        if (input_radix == radix)
            (*warpCounter)++;

        if ((i + 1) % WARP_SIZE == 0)
            warpCounter++;
    }
}

unsigned int radixInspect_prefixsum(unsigned int* radixCounters, unsigned int numBlocks)
{
    //   do prefix-sum
    for (std::size_t i = 1; i < numBlocks * WARP_SIZE; i++)
        radixCounters[i] += radixCounters[i - 1];

    //  return total sum
    return radixCounters[numBlocks * WARP_SIZE - 1];
}
#endif

int radixsort_cuda(unsigned int* numbers, int* indices, std::size_t numElements)
{
    unsigned int* d_numbers = nullptr,
        *aux_numbers = nullptr;
    int* d_indices = nullptr,
        *aux_indices = nullptr;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaErrHnd(cudaMalloc((void**)&d_numbers, sizeof(unsigned int) * numElements));
    cudaErrHnd(cudaMalloc((void**)&aux_numbers, sizeof(unsigned int) * numElements));
    cudaErrHnd(cudaMalloc((void**)&d_indices, sizeof(int) * numElements));
    cudaErrHnd(cudaMalloc((void**)&aux_indices, sizeof(int) * numElements));

    unsigned int numBlocks = (numElements + 1024 - 1) / 1024;
    unsigned int* aux_counters = nullptr,
        *aux_totalSum = nullptr;

    //  allocate GPU buffer for radix counters table and total sum of each radix
    cudaErrHnd(cudaMalloc((void**)&aux_counters, sizeof(unsigned int) * NUM_RADICES * numBlocks * WARP_SIZE));
    cudaErrHnd(cudaMalloc((void**)&aux_totalSum, sizeof(unsigned int) * NUM_RADICES));

#ifdef MEASURE_TIME_DEVICE
    //  warm up the device first
    countKernel<<< numBlocks, 1024 >>>(d_numbers, aux_counters, numElements, numBlocks, 0);
    cudaErrHnd(cudaGetLastError());
    radixSumScanKernel<<< NUM_RADICES, 1024 >>>(aux_counters, aux_totalSum, numBlocks);
    cudaErrHnd(cudaGetLastError());
    totalSumScanKernel<<< 1, NUM_RADICES >>>(aux_totalSum);
    cudaErrHnd(cudaGetLastError());
    reorderKernel<<< numBlocks, 1024 >>>(d_numbers, d_indices, aux_counters, aux_totalSum,
        aux_numbers, aux_indices, numElements, numBlocks, 0);
    cudaErrHnd(cudaGetLastError());

    //  flush every kernel running in GPU
    cudaErrHnd(cudaDeviceSynchronize());

    //  measure time per kernel
    cudaEvent_t ev_start[4], ev_stop[4];
    for (int i = 0; i < 4; i++)
    {
        cudaEventCreate(&ev_start[i]);
        cudaEventCreate(&ev_stop[i]);
    }


    auto time_start = std::chrono::high_resolution_clock::now();
#endif

    //  transfer input data to GPU
    cudaErrHnd(cudaMemcpy(d_numbers, numbers, sizeof(unsigned int) * numElements, cudaMemcpyHostToDevice));
    cudaErrHnd(cudaMemcpy(d_indices, indices, sizeof(int) * numElements, cudaMemcpyHostToDevice));

#ifdef DEBUG_DEVICE_IMPL 
    std::cout << std::endl << "(D) with Num Blocks = " << numBlocks << " :" << std::endl;
    std::vector<unsigned int> h_counters(NUM_RADICES * numBlocks * WARP_SIZE);
    std::vector<unsigned int> h_tempCounters(numBlocks * WARP_SIZE);
    std::vector<unsigned int> h_totalSums(NUM_RADICES);
    std::vector<unsigned int> h_tempTotalSums(NUM_RADICES);
#endif

    //  we need 4 passes to sort each radix
    for (unsigned int d = 0; d < 4; d++)
    {
        //	pick up in and out buffer for this round
        unsigned int* inNumbers, * outNumbers;
        int* inIndices, * outIndices;
        if (d % 2)
        {
            inNumbers = aux_numbers; inIndices = aux_indices;
            outNumbers = d_numbers; outIndices = d_indices;
        }
        else
        {
            inNumbers = d_numbers; inIndices = d_indices;
            outNumbers = aux_numbers; outIndices = aux_indices;
        }

#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_start[0]);
#endif
        //  launch PHASE I
        countKernel<<< numBlocks, 1024 >>>(inNumbers, aux_counters, numElements, numBlocks, d);
        cudaErrHnd(cudaGetLastError());
#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_stop[0]);
#endif

#ifdef DEBUG_DEVICE_IMPL 
        std::cout << "Pass [" << d << "]" << std::endl;
        
        //  check counters table
        std::cout << "On counting, Found inequality at radix : ";
        for (int ir = 0; ir < NUM_RADICES; ir++)
        {
            unsigned int* targetCounters = h_counters.data() + ir * numBlocks * WARP_SIZE;
            std::memset(targetCounters, 0, sizeof(unsigned int) * numBlocks * WARP_SIZE);
            ::radixInspect_count(ir, d, numbers, targetCounters, numElements, numBlocks);

            cudaErrHnd(cudaMemcpy(h_tempCounters.data(), aux_counters + ir * numBlocks * WARP_SIZE, 
                sizeof(unsigned int) * numBlocks * WARP_SIZE, cudaMemcpyDeviceToHost));

            for (int j = 0; j < numBlocks * WARP_SIZE; j++)
            {
                if (targetCounters[j] != h_tempCounters[j])
                {
                    std::cout << ir << " start at idx " << j << ", ";
                    break;
                }
            }
        }
        std::cout << "." << std::endl;
#endif        

#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_start[1]);
#endif
        //  launch PHASE II.1
        radixSumScanKernel<<< NUM_RADICES, 1024 >>> (aux_counters, aux_totalSum, numBlocks);
        cudaErrHnd(cudaGetLastError());
#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_stop[1]);
#endif

#ifdef DEBUG_DEVICE_IMPL 
        //  check prefix-sum table
        std::cout << "On prefix-sum, Found inequality at radix : ";
        for (int ir = 0; ir < NUM_RADICES; ir++)
        {
            unsigned int* targetCounters = h_counters.data() + ir * numBlocks * WARP_SIZE;
            h_totalSums[ir] = ::radixInspect_prefixsum(targetCounters, numBlocks);

            cudaErrHnd(cudaMemcpy(h_tempCounters.data(), aux_counters + ir * numBlocks * WARP_SIZE,
                sizeof(unsigned int) * numBlocks * WARP_SIZE, cudaMemcpyDeviceToHost));

            for (int j = 0; j < numBlocks * WARP_SIZE; j++)
            {
                if (targetCounters[j] != h_tempCounters[j])
                {
                    std::cout << ir << " start at idx " << j << ", ";
                    break;
                }
            }
        }
        std::cout << "." << std::endl;

        //  check total-sum list
        cudaErrHnd(cudaMemcpy(h_tempTotalSums.data(), aux_totalSum, sizeof(unsigned int) * NUM_RADICES, cudaMemcpyDeviceToHost));
        std::cout << "On original total sum, Found inequality at radix : ";
        for (int ir = 0; ir < NUM_RADICES; ir++)
            if(h_totalSums[ir] != h_tempTotalSums[ir])
                std::cout << ir << "(" << h_totalSums[ir] << " vs " << h_tempTotalSums[ir] << "), ";
        std::cout << "." << std::endl;
#endif

#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_start[2]);
#endif
        //  launch PHASE II.2
        totalSumScanKernel<<< 1, NUM_RADICES >>>(aux_totalSum);
        cudaErrHnd(cudaGetLastError());
#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_stop[2]);
#endif

#ifdef DEBUG_DEVICE_IMPL 
        //  check prefix-sum of total-sum list
        cudaErrHnd(cudaMemcpy(h_tempTotalSums.data(), aux_totalSum, sizeof(unsigned int) * NUM_RADICES, cudaMemcpyDeviceToHost));
        std::cout << "On scanned total sum, Found inequality at radix : ";
        for (int ir = 0; ir < NUM_RADICES; ir++)
        {
            if (ir > 0) h_totalSums[ir] += h_totalSums[ir - 1];
            if (h_totalSums[ir] != h_tempTotalSums[ir])
                std::cout << ir << "(" << h_totalSums[ir] << " vs " << h_tempTotalSums[ir] << "), ";
        }
        std::cout << "." << std::endl;
#endif

#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_start[3]);
#endif
        //  launch PHASE III
        reorderKernel<<< numBlocks, 1024 >>>(inNumbers, inIndices, aux_counters, aux_totalSum,
            outNumbers, outIndices, numElements, numBlocks, d);
        cudaErrHnd(cudaGetLastError());
#ifdef MEASURE_TIME_DEVICE
        if (d == 0)
            cudaEventRecord(ev_stop[3]);
#endif

#ifdef DEBUG_DEVICE_IMPL 
        if (d == 3) break;
        cudaErrHnd(cudaMemcpy(numbers, outNumbers, sizeof(unsigned int) * numElements, cudaMemcpyDeviceToHost));
        std::cout << "Out: ";
        for (int i = 401; i < 411; i++)
            std::cout << std::hex << ((numbers[i] & (0x000000ff << 8 * (d + 1))) >> 8 * (d + 1)) << " ";
        std::cout << std::dec << std::endl;
#endif
    }

    //  transfer result back to host
    cudaErrHnd(cudaMemcpy(numbers, d_numbers, sizeof(unsigned int) * numElements, cudaMemcpyDeviceToHost));
    cudaErrHnd(cudaMemcpy(indices, d_indices, sizeof(int) * numElements, cudaMemcpyDeviceToHost));

#ifdef MEASURE_TIME_DEVICE
    auto time_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);
    std::cout << "Radixsort (D) exec time : " << duration.count() << "us." << std::endl;

    std::cout << "Per kernel :" << std::endl;
    float kernelExecTimes[4];
    for (int i = 0; i < 4; i++)
    {
        cudaEventElapsedTime(&kernelExecTimes[i], ev_start[i], ev_stop[i]);
        cudaEventDestroy(ev_start[i]);
        cudaEventDestroy(ev_stop[i]);
    }
    std::cout << "\tPhase I : " << kernelExecTimes[0] << "ms." << std::endl;
    std::cout << "\tPhase II.1 : " << kernelExecTimes[1] << "ms." << std::endl;
    std::cout << "\tPhase II.2 : " << kernelExecTimes[2] << "ms." << std::endl;
    std::cout << "\tPhase III : " << kernelExecTimes[3] << "ms." << std::endl;


#endif

    //  free device memory
    cudaErrHnd(cudaFree(aux_totalSum));
    cudaErrHnd(cudaFree(aux_counters));

    cudaErrHnd(cudaFree(aux_indices));
    cudaErrHnd(cudaFree(d_indices));
    cudaErrHnd(cudaFree(aux_numbers));
    cudaErrHnd(cudaFree(d_numbers));

    return 0;
}

int main()
{
    //  setup the sample input array
    std::size_t numElements = 0x1 << 20;
    std::vector<unsigned int> numbers(numElements);
    std::vector<int> indices(numElements);
    std::iota(indices.begin(), indices.end(), 0);
    int result;

    //  generate randon numbers for sample input input
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution_input(0, 0xffffffff);
    for (auto itr = numbers.begin(); itr != numbers.end(); itr++)
        *itr = distribution_input(generator);

    //  prepare data for test
    std::vector<unsigned int> numbers_copy(numbers);
    std::vector<int> indices_copy(indices);

    //  sorting by CPU
    std::cout << "Running CPU Implementation... " << std::endl;
    result = ::radixsort_simple(numbers.data(), indices.data(), numElements);
    if (result != 0) {
        fprintf(stderr, "radixsort (CPU) failed!");
        return 1;
    }
    std::cout << "Completed!" << std::endl;

    //  sorting by GPU
    std::cout << "Running GPU (CUDA) Implementation... " << std::endl;
    result = ::radixsort_cuda(numbers_copy.data(), indices_copy.data(), numElements);
    if (result != 0) {
        fprintf(stderr, "radixsort (CUDA) failed!");
        return 1;
    }
    std::cout << "Completed!" << std::endl;

    int valid = 0;

#ifdef VALIDATE_FULL
    //  full validation
    std::cout << "Full validation :\nMismatch index (H vs D) at " << std::endl;
    for (unsigned int i = 0; i < numElements; i++)
    {
        if (numbers[i] != numbers_copy[i])
        {
            std::cout << i << " ( " << numbers[i] << " at " << indices[i] << " vs "
                << numbers_copy[i] << " at " << indices_copy[i] << " ), \n";
            valid = 1;
        }
    }
    std::cout << std::endl;
#else
    //  validate the result stocastically
    std::cout << "Stocastic validation :\nMismatch index(H,D) at " << std::endl;
    std::uniform_int_distribution<unsigned int> distribution_validate(1, numElements - 1);
    for (std::size_t i = 0; i < 100; i++)
    {
        std::size_t sampleIndex = distribution_validate(generator);
        if (numbers[sampleIndex] != numbers_copy[sampleIndex])
        {
            std::cout << i << " ( " << numbers[i] << " at " << indices[i] << " vs "
                << numbers_copy[i] << " at " << indices_copy[i] << " ), \n";
            valid = 1;
        }
    }
#endif

    if (valid)
        printf("There is a bug somewhere.\n");
    else
        printf("Congrats! All goes well.\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
