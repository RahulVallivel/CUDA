#!/usr/bin/env python
import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
 
# -- initialize the device
import pycuda.autoinit

class Transpose:
	def __init__(self, a_cpu):
		self.a_cpu=a_cpu 
		self.b_cpu=np.zeros([self.a_cpu.shape[1],self.a_cpu.shape[0]])
		self.a_gpu = gpuarray.to_gpu(self.a_cpu)  #copy a to device
		self.b_gpu = gpuarray.empty(([self.a_cpu.shape[1],self.a_cpu.shape[0]]), self.a_gpu.dtype)  #create an empty array in device for the result
        # Kernel code
		self.kernel_code = """
		
		
		__global__ void transpose(float *idata, float *odata, float w, float h)
{
 		
 		unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int yIdx = blockDim.y * blockIdx.y + threadIdx.y;
        if ( xIdx <w && yIdx <h){
        unsigned int idx_in = xIdx + w * yIdx;
        unsigned int idx_out = yIdx + h * xIdx;
        odata[idx_out] = idata[idx_in];
        }
        
}

		"""


	def serial_transpose(self):
		start = time.time()
		for i in range(self.a_cpu.shape[1]):
			for j in range(self.a_cpu.shape[0]):
				self.b_cpu[i,j]=self.a_cpu[j,i]
		self.times_cpu=time.time()-start
		
		return self.b_cpu,self.times_cpu


	def parallel_transpose(self):
		# return: the transpose of input matrix
        # TODO:
        # Memory copy to device
        # Function call and measuring time here
		

		mod = compiler.SourceModule(self.kernel_code)  #compiling the kernel
		transpose = mod.get_function("transpose") #getting the function

		start = time.time() #start timer
		transpose(
        # inputs
        self.a_gpu,
        # output
        self.b_gpu,

        np.float32(self.a_cpu.shape[1]),

        np.float32(self.a_cpu.shape[0]),

        # block size
        block = (16, 16, 1)
        
        

        )      #kernel call
		self.times_gpu=time.time()-start #get time
        # Memory copy to host
		self.b_cpu=self.b_gpu.get()  #copy result to host variable
        # Return output and measured time

		return self.b_cpu, self.times_gpu

class MatrixMultiply:
	def __init__(self, a_cpu, b_cpu):
		self.a_cpu=a_cpu 
		self.b_cpu=b_cpu
		self.a_gpu = gpuarray.to_gpu(self.a_cpu)
		self.b_gpu = gpuarray.to_gpu(self.b_cpu)
		self.c_gpu = gpuarray.empty(([self.a_cpu.shape[0],self.b_cpu.shape[1]]), self.a_cpu.dtype)
		self.TILE_WIDTH=self.a_cpu.shape[0]
		self.kernel_code_template = """
__global__ void MatrixMulKernel_naive(float *A, float *B, float *C, int m, int n, int k)
{
    int Row= blockDim.y * blockIdx.y + threadIdx.y;
    int Col= blockDim.x * blockIdx.x + threadIdx.x;

    if ((Row<m) && (Col<k)){
    float Cvalue=0.0;
    for(int i =0; i<n; ++i)
    Cvalue+= A[Row*n+i] * B[Col+i*k];
    C[Row*k+Col] = Cvalue;
    }
}	
	

__global__ void MatrixMulKernel_optimized1(float *A, float *B, float *C, int m, int n, int k)
{	const int TILE_WIDTH= %(TILE_WIDTH)s;
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx= blockIdx.x; int by =blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * blockDim.y +ty;
	int Col = bx * blockDim.x +tx;
	float Cvalue=0.0;

	for (int t = 0; t <(n-1)/TILE_WIDTH + 1; ++t) {
	if(Row < m && t*TILE_WIDTH+tx < n) {
	ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
	}
	else{
	ds_A[ty][tx] = 0.0;
	}
	if (t*TILE_WIDTH+ty < n && Col < k) {
	ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + Col];
	}
	else {
 	ds_B[ty][tx] = 0.0;
	}
	__syncthreads();

	for (int i = 0; i < TILE_WIDTH; ++i)
	Cvalue += ds_A[ty][i] * ds_B[i][tx];
 	__syncthreads();

 	}


	C[Row*k+Col] = Cvalue;

}

__global__ void MatrixMulKernel_optimized2(float *A, float *B, float *C, int m, int n, int k)
{	const int TILE_WIDTH= %(TILE_WIDTH)s;
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH+1];

	int bx= blockIdx.x; int by =blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * blockDim.y +ty;
	int Col = bx * blockDim.x +tx;
	float Cvalue=0.0;

	for (int t = 0; t <(n-1)/TILE_WIDTH + 1; ++t) {
	if(Row < m && t*TILE_WIDTH+tx < n) {
	ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
	}
	else{
	ds_A[ty][tx] = 0.0;
	}
	if (t*TILE_WIDTH+ty < n && Col < k) {
	ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*k + Col];
	}
	else {
 	ds_B[ty][tx] = 0.0;
	}
	__syncthreads();

	for (int i = 0; i < TILE_WIDTH; ++i)
	Cvalue += ds_A[ty][i] * ds_B[i][tx];
 	__syncthreads();

 	}
 	

	C[Row*k+Col] = Cvalue;
}




"""

	def matrix_mul_naive(self):
		#MATRIX_SIZE = 10
		#self.kernel_code = self.kernel_code_template % {
    #'MATRIX_SIZE': MATRIX_SIZE
    #}	
	
		TILE_WIDTH = self.TILE_WIDTH
		self.kernel_code = self.kernel_code_template % {
		'TILE_WIDTH': TILE_WIDTH}
		n=int(np.ceil(self.b_cpu.shape[1]/TILE_WIDTH))
		m=int(np.ceil(self.a_cpu.shape[0]/TILE_WIDTH))
		mod = compiler.SourceModule(self.kernel_code)
		matmul = mod.get_function("MatrixMulKernel_naive")
		start = time.time()
		matmul(
				self.a_gpu,
				self.b_gpu,
				self.c_gpu,
				np.int32(self.a_cpu.shape[0]),
				np.int32(self.a_cpu.shape[1]),
				np.int32(self.b_cpu.shape[1]),
				block=(TILE_WIDTH,TILE_WIDTH, 1),
				grid=(n,m,1)

			)
		times_gpu_=time.time()-start 
		

		self.c_cpu=self.c_gpu.get()

		return self.c_cpu,times_gpu_

	def matrix_mul_optimized1(self):
		TILE_WIDTH = self.TILE_WIDTH
		self.kernel_code = self.kernel_code_template % {
		'TILE_WIDTH': TILE_WIDTH}

		n=int(np.ceil(self.b_cpu.shape[1]/TILE_WIDTH))
		m=int(np.ceil(self.a_cpu.shape[0]/TILE_WIDTH))

		mod = compiler.SourceModule(self.kernel_code)
		matmul_1 = mod.get_function("MatrixMulKernel_optimized1")
		start = time.time()
		matmul_1(
				self.a_gpu,
				self.b_gpu,
				self.c_gpu,
				np.int32(self.a_cpu.shape[0]),
				np.int32(self.a_cpu.shape[1]),
				np.int32(self.b_cpu.shape[1]),
				block=(TILE_WIDTH, TILE_WIDTH, 1),
				grid=(n,m,1)
			)
		times_gpu_=time.time()-start 
		self.c_cpu=self.c_gpu.get()

		return self.c_cpu,times_gpu_


	def matrix_mul_optimized2(self):
		TILE_WIDTH = self.TILE_WIDTH
		self.kernel_code = self.kernel_code_template % {
		'TILE_WIDTH': TILE_WIDTH}

		n=int(np.ceil(self.b_cpu.shape[1]/TILE_WIDTH))
		m=int(np.ceil(self.a_cpu.shape[0]/TILE_WIDTH))

		mod = compiler.SourceModule(self.kernel_code)
		matmul_1 = mod.get_function("MatrixMulKernel_optimized2")
		start = time.time()
		matmul_1(
					self.a_gpu,
					self.b_gpu,
					self.c_gpu,
					np.int32(self.a_cpu.shape[0]),
					np.int32(self.a_cpu.shape[1]),
					np.int32(self.b_cpu.shape[1]),
					block=(TILE_WIDTH, TILE_WIDTH, 1),
					grid=(n,m,1)
				)
		times_gpu_1=time.time()-start 
		self.c_cpu=self.c_gpu.get()

		return self.c_cpu,times_gpu_1


def main():

	a_cpu=np.random.randn(7,2).astype(np.float32)
	b_cpu=a_cpu.T.copy()
	#b_cpu=np.random.randn(2,2).astype(np.float32)
		
	mul=MatrixMultiply(a_cpu,b_cpu)

	c_cpu1,times_gpu_1=mul.matrix_mul_optimized1()

	print(c_cpu1)
	#print(c_cpu2)
	#print(c_cpu3)
	print(np.dot(a_cpu,b_cpu))
	print(np.sum(c_cpu1-np.dot(a_cpu,b_cpu)))
	#print(np.sum(c_cpu2-np.dot(a_cpu,b_cpu)))
	#print(np.sum(c_cpu3-np.dot(a_cpu,b_cpu)))

if __name__ == '__main__':
    main()

