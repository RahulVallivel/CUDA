
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np 
import time
import cv2 as cv 
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def compute_histo(img_cpu):
	img_cpu=np.int32(img_cpu)
	img_gpu=gpuarray.to_gpu(img_cpu)
	hist_gpu=gpuarray.empty(256,img_cpu.dtype)

	kernel="""
	#include <stdio.h>
	__global__ void histo_kernel( int *buffer, int size,
                                int *histo )
{
     

     int x = threadIdx.x + blockIdx.x * blockDim.x;
     int y = threadIdx.y + blockIdx.y * blockDim.y;
     int offsetx = blockDim.x * gridDim.x;
     int offsety = blockDim.y * gridDim.y;
     
     
     int i= (x*offsetx)+y ;     
     
    
     __syncthreads();



     while (i < size)
     {
     

              atomicAdd( &histo[buffer[i]], 1);

              i=size+1;
     }
     __syncthreads();

    
}


	"""
	m=((9-1)//8) + 1
	n=((9-1)//8) + 1
	mod = compiler.SourceModule(kernel)
	compute_histo_gpu = mod.get_function("histo_kernel")
	size=np.int32(9*9)
	compute_histo_gpu(img_gpu,size, hist_gpu, block = (8,8,1), grid =(m,n,1))
	hist_cpu = hist_gpu.get()
	print(hist_cpu)
	print(hist_cpu.shape)
	c_X = np.histogram(np.uint8(img_cpu.flatten()),bins=np.arange(257))[0]
	print(c_X)
	print(hist_cpu==c_X.T)
	print('gpu:',np.sum(hist_cpu))
	print('cpu:',np.sum(c_X))
	return hist_cpu

def compute_histo2d(img_cpu1,img_cpu2):
	img_cpu1=np.int32(img_cpu1)
	img_cpu2=np.int32(img_cpu2)
	img_gpu1=gpuarray.to_gpu(img_cpu1)
	img_gpu2=gpuarray.to_gpu(img_cpu2)
	hist_gpu=gpuarray.empty([256,256],img_cpu1.dtype)
	kernel="""
	#include <stdio.h>
	__global__ void histo_kernel2(int *dev_x_coord, int *dev_y_coord, int size, int *histo) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
   
    int i= (x*offsetx)+y ;
   
    while (i<size)
    {
        atomicAdd(&histo[dev_y_coord[i] * 256 + dev_x_coord[i]], 1);
        i=size+1;
     
    }
}

	"""
	#include <stdio.h>
	m=((9-1)//8) + 1
	n=((9-1)//8) + 1
	mod = compiler.SourceModule(kernel)
	compute_histo_gpu2 = mod.get_function("histo_kernel2")
	size=np.int32(9*9)
	compute_histo_gpu2(img_gpu1,img_gpu2,size, hist_gpu, block = (8,8,1), grid =(m,n,1))
	hist_cpu = hist_gpu.get()
	print(hist_cpu)
	print(hist_cpu.shape)
	x=np.arange(257)
	y=np.arange(257)
	c_X = np.histogram2d(np.uint8(img_cpu1.flatten()),np.uint8(img_cpu2.flatten()),bins=(x,y) )[0]
	c_X=np.int32(c_X)
	print(c_X)
	print(hist_cpu==c_X.T)
	print('gpu:',np.sum(hist_cpu))
	print('cpu:',np.sum(c_X))
	return hist_cpu,c_X.T

def main():

	L = cv.imread('/home/rs3871/project/data/im0.ppm')
	R = cv.imread('/home/rs3871/project/data/im6.ppm')
	L_gray = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
	R_gray = cv.cvtColor(R, cv.COLOR_BGR2GRAY)
	img1 = L_gray 
	img2= R_gray
	img1=img1[0:9,0:9]
	img2=img2[0:9,0:9]
	print(img1.shape)
	histo=compute_histo2d(img1,img2)
	#plt.imshow(histo)
	#plt.savefig('histo')
	#plt.imshow(histo1)
	#plt.savefig('histo1')
if __name__ == "__main__":
	main()

