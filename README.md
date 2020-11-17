# Radix Sort
It is a project to demonstrate how to implement radix sort in both serialized version and parallel version using CUDA. The execution time of each version is also measured to compare the performance of the algorithm.

## System Requirements
- Any NVIDIA GPU with compute capability > 3.0 ( > 6.0 is recommended because of a hardware-based atomic operation on shared memory)

You can check the compute capability corresponding to your GPU [here](https://developer.nvidia.com/cuda-gpus).

## How to build
```
nvcc radixsort.cu -o radixsort.o
./radixsort.o
```
## Radixsort exec time comparison
|magnitude|CPU time(us)|GPU time(us)|
|---------|--------|--------|
|2^10(10^3)|165|197|
|2^12|100|205|
|2^14(10^4)|396|242|
|2^16|1706|473|
|2^18(10^5)|7606|1729|
|2^20(10^6)|29520|6426|