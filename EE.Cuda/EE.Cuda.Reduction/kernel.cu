#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

int cpu_reduce(int *data, unsigned int n) {
	int res = 0;
	for (int i = 0; i < n; ++i)
		res += data[i];
	return res;
}

__global__ void reduce(int *data, int *result) {
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	sdata[tid] = data[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) result[blockIdx.x] = sdata[0];
}

void print_device_info() {
	int dev = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, dev);

	printf("DEVICE: %s\r\n", prop.name);
	cudaSetDevice(dev);
}

void cpu_expirement(int* data, int size) {
	clock_t startc, end;
	double cpu_time_used;

	startc = clock();

	int cpu_sum = cpu_reduce(data, size);

	end = clock();
	cpu_time_used = ((double)(end - startc)) / CLOCKS_PER_SEC;
	cpu_time_used *= 1000;

	printf("CPU: TIME=%fms; RESULT=%d\n", cpu_time_used, cpu_sum);
}

void gpu_experiment(int* data, int size) {
	float timerValueGPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	size_t bytes = size * sizeof(int);
	int blockSize = 512;
	const dim3 block(blockSize, 1);
	const dim3 grid((size + block.x - 1) / block.x, 1);

	int *result = (int *)malloc(grid.x * sizeof(int));

	int *device_input;
	int *device_output;
	cudaMalloc((void **)&device_input, bytes);
	cudaMalloc((void **)&device_output, bytes);

	cudaMemcpy(device_input, data, bytes, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0);
	reduce <<<grid, block, blockSize * sizeof(int)>>>(device_input, device_output);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);

	cudaMemcpy(result, device_output, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int gpu_sum = 0;
	for (int i = 0; i < grid.x; ++i) gpu_sum += result[i];
	printf("GPU: TIME=%fms; RESULT=%d", timerValueGPU, gpu_sum);

	cudaDeviceReset();
	cudaFree(device_output);
	cudaFree(device_input);
	free(result);
}

int main(int argc, char **argv) {
	print_device_info();

	int size = 6660000;
	printf("SIZE:%d\n", size);
	size_t bytes = size * sizeof(int);
	int *data = (int *)malloc(bytes);
	for (int i = 0; i < size; ++i)
		data[i] = (int)(rand() & 0xFF);

	cpu_expirement(data,size);
	gpu_experiment(data, size);

	free(data);
	return 0;
}
