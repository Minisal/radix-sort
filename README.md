# Radix Sort
This is my project task demonstrating how to implement radix sort in both serialized version and parallel version using CUDA. The execution time of each version is also measured to compare the performance of the algorithm.

## System Requirements
- Any NVIDIA GPU with compute capability > 3.0 ( > 6.0 is recommended because of a hardware-based atomic operation on shared memory)

You can check the compute capability corresponding to your GPU [here](https://developer.nvidia.com/cuda-gpus).

## How to build
Just compile the single-file code with `nvcc`. The compiler is already bundled with [CUDA SDK](https://developer.nvidia.com/cuda-downloads).

### References
Le Grand, Scott. Chapter 32. Broad-Phase Collision Detection with CUDA. [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda). NVIDIA Corporation.
