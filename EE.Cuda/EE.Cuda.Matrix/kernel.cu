#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include <time.h>
#define index(i,j,ld) (((j)*(ld))+(i))

void print_device_info() {
	int dev = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, dev);

	printf("DEVICE: %s\r\n", prop.name);
	cudaSetDevice(dev);
}

int random_int(int min, int max) {
	return min + (rand() * (int)(max - min) / RAND_MAX);
}

void setup_matrix(float* matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[index(i, j, size)] = (float)random_int(0, 1000);
		}
	}
}

void cpu_matrix_mult(float *a, float *b, float *result, int size) {
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			int tmp = 0.0;
			for (int h = 0; h < size; ++h)
			{
				tmp += b[i * size + h] * a[h * size + j];
			}
			result[i * size + j] = tmp;
		}
	}
}

bool assert_matrix(float *a, float *b, int size) {
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			if (a[i * size + j] != b[i * size + j]) {
				return false;
			}
		}
	}
	return true;
}

void cpu_experiment(float* a, float* b, float* result, int size) {
	clock_t startc, end;
	double cpu_time_used;

	startc = clock();

	cpu_matrix_mult(a, b, result, size);

	end = clock();
	cpu_time_used = ((double)(end - startc)) / CLOCKS_PER_SEC;
	cpu_time_used *= 1000;
	printf("===CPU===:\nTIME=%fms.\n", cpu_time_used);
}

void gpu_experiment(float* a, float* b, float* result, int size) {
	float timerValueGPU, setupTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* gpu_a; float* gpu_b; float* gpu_result;

	cublasAlloc(size*size, sizeof(float), (void**)&gpu_a);
	cublasAlloc(size*size, sizeof(float), (void**)&gpu_b);
	cublasAlloc(size*size, sizeof(float), (void**)&gpu_result);

	cudaEventRecord(start, 0);

	cublasSetMatrix(size, size, sizeof(float), a, size, gpu_a, size);
	cublasSetMatrix(size, size, sizeof(float), b, size, gpu_b, size);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setupTime, start, stop);

	cudaEventRecord(start, 0);

	cublasSgemm('n', 'n', size, size, size, 1, gpu_a, size, gpu_b, size, 0, gpu_result, size);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);

	cublasGetMatrix(size, size, sizeof(float), gpu_result, size, result, size);

	printf("===GPU===:\nSETUP TIME=%fms;\nCALCULATE TIME=%f.\n", setupTime ,timerValueGPU);
	cublasFree(gpu_a);
	cublasFree(gpu_b);
	cublasFree(gpu_result);
}

int  main(int argc, char** argv) {
	print_device_info();

	int size = 100;
	printf("SIZE:%d\n", size);

	cublasStatus status;
	float *a = (float*)malloc(size * size * sizeof(float));
	float *b = (float*)malloc(size*size * sizeof(float));
	float *cpu_result = (float*)malloc(size*size * sizeof(float));
	float *gpu_result = (float*)malloc(size*size * sizeof(float));

	setup_matrix(a, size);
	setup_matrix(b, size);

	cpu_experiment(a, b, cpu_result, size);
	gpu_experiment(a, b, gpu_result, size);
	
	printf("===RESULT===:\n");
	if (assert_matrix(cpu_result, gpu_result, size)) {
		printf("SUCCESS:)\n");
	}
	else {
		printf("ERROR:(\n");
	}

	free(a);
	free(b);
	free(cpu_result);
	free(gpu_result);

	status = cublasShutdown();
	return 0;
}
