/**
*	This is my first program in learning parallel programming using CUDA.
*	Equivalent to a hello World program :-)
*	This program basically performs two tasks:
*	1. It selects suitable CUDA enabled device(GPU) and prints the device properties
*	2. It demonstrate basic parallel addition of two arrays on the device(GPU) using add kernel.
*	Author: Shubham Singh
**/

#include "cuda_runtime.h"
#include <iostream>

#define N 10						/*N is size of arrays*/

using namespace std;

/************************************************************************************************************
*	Function:	Kernel to perform addition of two arrays in parallel on device(GPU)
*	Input:		Takes 3 pointer to int variables pointing to some memory locations on the device(GPU)
*	Output:		None
************************************************************************************************************/
__global__ void add(int *a, int *b, int *c)
{
	int i = blockIdx.x;				/*blockIDx.x holds ID of block and acts as index*/
	if (i < N)
		c[i] = a[i] + b[i];
}

int main()
{
	cudaDeviceProp prop;			/*Structure variable to hold properties of a CUDA enabled device(GPU)*/
	int count, dev;	
	cudaGetDevice(&dev);			/*Function to get current device ID and store device ID in dev*/
	cout << "ID of current cuda Device: " << dev<< endl;
	
	/*
	*	If system has multiple GPUs then
	*	Find CUDA device(GPU) having major computing capability greater than 1
	*	or having major computing capability 1 and minor computing capability greater than 3
	*/
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	cudaChooseDevice(&dev, &prop);	/*Get device ID for revision greater than 1.3*/
	cout << "ID of CUDA device closest to revision 1.3: " << dev << endl;
	cudaSetDevice(dev);				/*Set current device(GPU) to device ID dev which is having revision greater than 1.3*/
	
	/*
	*	Get properties of device dev from CUDA runtime and hold properties into prop
	*	and print few details of device dev
	*/
	cudaGetDeviceProperties(&prop, dev);
	cout << "\n----Properties for device ID " << dev << "----" << endl << endl;
	cout << "Device name: " << prop.name << endl;
	cout << "Device clock rate(in kilohertz): " << prop.clockRate << endl;
	cout << "Device global memory(in bytes): " << prop.totalGlobalMem << endl;
	cout << "Device constant memory(in bytes): " << prop.totalConstMem << endl;
	if (prop.deviceOverlap)
		cout << "Device Overlap: Enabled" << endl;
	else
		cout << "Device Overlap: Disabled" << endl;
	if (prop.concurrentKernels)
		cout << "Concurrent kernels: Yes" << endl;
	else
		cout << "Concurrent kernels: Yes" << endl;
	cout << "Multiprocessor count: " << prop.multiProcessorCount << endl;
	cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
	cout << "Max thread dimension: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << " " << endl;
	cout << "Max grid dimension: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << " " << endl;
	cout << "Size of L2 cache(in bytes): " << prop.l2CacheSize << endl;
	cout << "Device's revision/compute capability: " << prop.major << "." << prop.minor << endl;
	
	/*
	*	So, we are done with printing device(GPU) details
	*	It's time to perform some parallel computation on device(GPU)
	*	having device ID dev.
	*	Let's perform simple array addition on device(GPU)
	*/
	cout << "\nLet's perform some addition on arrays" << endl;
	int *a, *b, *c;					/*Variables to hold arrays on host(CPU)*/
	int *dev_a, *dev_b, *dev_c;		/*Variables to hold arrays on device(GPU)*/

	/*Allocate memory for arrays on host(CPU)*/
	a = new int[N];
	b = new int[N];
	c = new int[N];

	/*Fill values in arrays on host(CPU)*/
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = N - i - 1;
	}

	/*Print arrays*/
	cout << "Array a: ";
	for (int i = 0; i < N; i++)
		cout << a[i] << " ";
	cout << "\nArray b: ";
	for (int i = 0; i < N; i++)
		cout << b[i] << " ";

	/*Allocate memory for arrays on device(GPU)*/
	cudaMalloc((void **)&dev_a, N * sizeof(int));
	cudaMalloc((void **)&dev_b, N * sizeof(int));
	cudaMalloc((void **)&dev_c, N * sizeof(int));

	/*Copy arrays a and b from host(CPU) to device(GPU)*/
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	/*Call add kernel to perform addition on device(GPU)*/
	add <<< N, 1 >>> (dev_a, dev_b, dev_c);

	/*Copy c array back from device(GPU) to host(CPU)*/
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	/*Print sum*/
	cout << "\nArray c: ";
	for (int i = 0; i < N; i++)
		cout << c[i] << " ";
	cout << endl;

	/*Free memory allocated on host(CPU)*/
	delete[] a;
	delete[] b;
	delete[] c;

	/*Free memory allocated on device(GPU)*/
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}