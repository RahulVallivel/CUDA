#!/usr/bin/env python
import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
 
# -- initialize the device
import pycuda.autoinit
import random
from array import array


class MergeSort:
    def merge(self,a, left, mid, right):
    
        copy_list = []
        i, j = left, mid + 1
        ind = left
        
        while ind < right+1:
            
            
            if i > mid:
                copy_list.append(a[j])
                j +=1
            
            elif j > right:
                copy_list.append(a[i])
                i +=1
            
            elif a[j] < a[i]:
                copy_list.append(a[j])
                j +=1
            else:
                copy_list.append(a[i])
                i +=1
            ind +=1
            
        ind=0
        for x in (xrange(left,right+1)):
            a[x] = copy_list[ind]
            ind += 1
    

    def merge_sort_serial(self,list_):
        
        factor = 2
        temp_mid = 0
        
        while 1:
            index = 0
            left = 0
            right = len(list_) - (len(list_) % factor) - 1
            mid = (factor / 2) - 1
            
            
            while index < right:
                temp_left = index
                temp_right = temp_left + factor -1
                mid2 = (temp_right +temp_left) / 2
                self.merge (list_, temp_left, mid2, temp_right)
                index = (index + factor)
            
            
            if len(list_) % factor and temp_mid !=0:
                
                self.merge(list_, right +1, temp_mid, len(list_)-1)
                
                mid = right
            
            factor = factor * 2
            temp_mid = right
           
            
            if factor > len(list_) :
                mid = right
                right = len(list_)-1
                self.merge(list_, 0, mid, right)
                break
        return list_


    def merge_sort_naive(self, a):
        a_cpu = np.array(a)
    
        if(a_cpu.shape[0]==1):
            return a_cpu.to_list()
        
        a_gpu = gpuarray.to_gpu(a_cpu.astype(np.int32))
    
        b_gpu = gpuarray.empty(a_cpu.shape, a_cpu.dtype)

        kernel_code = """
        #include <stdio.h>
        __device__ void merge(const int *arr1,  int arr1size, const int *arr2, int arr2size, int *out)
        {
        
        int x = 0;
        int y = 0;
        int n = 0;
        while(x < arr1size || y < arr2size) 
        {
        if(x < arr1size && y < arr2size)
       {
        if(arr1[x] < arr2[y]) 
        {
          out[n] = arr1[x];
          n++;
          x++;
        }
        
         else 
        {
          out[n] = arr2[y];
          n++;
          y++;
        }
      } 
    
      else if(x < arr1size) 
      {
        out[n] = arr1[x];
        n++;
        x++;
      } 
     
      else if(y < arr2size) 
      {
        out[n] = arr2[y];
        n++;
        y++;
      }
      
    
    }
}



__device__ void cpy(int *out,const int *in, int size) 
{
    int x = 0;

    for(x = 0; x < size; x++) 
    {
      out[x] = in[x];
      __syncthreads();
      printf("");

    }

    
  }

  


  __global__ void mergesort_naive(int *a,int *b, int len)
  {     

        
  int size = 1;
    int indx = (blockDim.x * blockIdx.x + threadIdx.x)*2;
    int i = 0;

    while(size <=len) 
    {
      __syncthreads();

      if((indx + size) <len) 
      {
        i = (indx + size) + (((indx + (size * 2)) > len) ? len - (indx + size) : size);
        
        
        merge(&a[indx], size, &a[indx + size], ((indx + (size * 2)) > len) ? len - (indx + size) : size, &b[indx]);
       
        
      } 
      else 
      {
        
        return;

      }
      size *= 2;
      __syncthreads();
      
      if(size>len)
      {
        break;
      }
      cpy(&a[indx], &b[indx], size);
      
      __syncthreads();
      if(indx%(2*size) != 0) 
      {
        
        
        return;
        
      }
    }
    len = i;
    __syncthreads();
  }
        """


        mod = compiler.SourceModule(kernel_code)  #compiling the kernel
        sort_naive = mod.get_function("mergesort_naive")
        if(a_cpu.shape[0]<=2048):
            block_threads=int(a_cpu.shape[0]/2)
            grid_blocks=1
        else:
            for i in range(1,33):
                if (a_cpu.shape[0]/(2*i)<1024):
                    block_threads=int(np.floor(a_cpu.shape[0]/(2*i))+1)
                    grid_blocks=int(i)
                    break
                
        start = time.time()
        sort_naive( a_gpu, b_gpu, np.int32(a_cpu.shape[0]), block=(block_threads,1,1),grid=(grid_blocks,1,1))
        times_cpu=time.time()-start
        b_cpu=b_gpu.get()

        return b_cpu.tolist(),times_cpu


    def merge_sort_optimized1(self, a):
        a_cpu = np.array(a)
    
        if(a_cpu.shape[0]==1):
            return a_cpu.to_list()
        
        a_gpu = gpuarray.to_gpu(a_cpu.astype(np.int32))
    
        b_gpu = gpuarray.empty(a_cpu.shape, a_cpu.dtype)

        kernel_code_opti = """
        #include <stdio.h>
        __device__ void merge(const int *arr1,  int arr1size, const int *arr2, int arr2size, int *out)
        {
        
        int x = 0;
        int y = 0;
        int n = 0;
        while(x < arr1size || y < arr2size) 
        {
        if(x < arr1size && y < arr2size)
       {
        if(arr1[x] < arr2[y]) 
        {
          out[n] = arr1[x];
          n++;
          x++;
        }
        
         else 
        {
          out[n] = arr2[y];
          n++;
          y++;
        }
      } 
    
      else if(x < arr1size) 
      {
        out[n] = arr1[x];
        n++;
        x++;
      } 
     
      else if(y < arr2size) 
      {
        out[n] = arr2[y];
        n++;
        y++;
      }
      
    
    }
}



__device__ void cpy(int *out,const int *in, int size) 
{
    int x = 0;

    for(x = 0; x < size; x++) 
    {
      out[x] = in[x];
      __syncthreads();
      printf("");

    }

    
  }

  

__global__ void mergesort_optimized(int *a,int *b, int len)
  {     

    __shared__ int ds_A[1024];  

    for(int e=0;e<len;e++)
    {
    ds_A[e]=a[e];
    }

    int size = 1;
    int indx = (blockDim.x * blockIdx.x + threadIdx.x)*2;
    int i = 0;

    while(size <=len) 
    {
      __syncthreads();

      if((indx + size) <len) 
      {
        i = (indx + size) + (((indx + (size * 2)) > len) ? len - (indx + size) : size);
        
        
        merge(&ds_A[indx], size, &ds_A[indx + size], ((indx + (size * 2)) > len) ? len - (indx + size) : size, &b[indx]);
       
        
      } 
      else 
      {
        
        return;

      }
      size *= 2;
      __syncthreads();
     
      if(size>len)
      {
        break;
      }
      cpy(&ds_A[indx], &b[indx], size);
     
      __syncthreads();
      if(indx%(2*size) != 0) 
      {
       
        
        return;
        
      }
    }
    len = i;
    __syncthreads();
  }
  
        """


        mod = compiler.SourceModule(kernel_code_opti)  #compiling the kernel
        sort_optimized = mod.get_function("mergesort_optimized")

        if(a_cpu.shape[0]<=2048):
            block_threads=int(a_cpu.shape[0]/2)
            grid_blocks=1
        else:
            for i in range(1,33):
                if (a_cpu.shape[0]/(2*i)<1024):
                    block_threads=int(np.floor(a_cpu.shape[0]/(2*i))+1)
                    grid_blocks=int(i)
                    break
        start = time.time()
        sort_optimized( a_gpu, b_gpu, np.int32(a_cpu.shape[0]), block=(block_threads,1,1),grid=(grid_blocks,1,1))
        times_cpu=time.time()-start
        b_cpu=b_gpu.get()

        return b_cpu.tolist(),times_cpu



def main():
    ti_c=[]
    ti_gn=[]
    ti_go=[]
    
    for i in range(1,10):
        sort= MergeSort()
        a_cpu = np.random.randint(1,9000,size=10*i).astype(np.int32)
        start = time.time()
        sorted_a_serial = sort.merge_sort_serial(a_cpu.copy())
        ti_serial =time.time()-start
        sorted_a_naive,ti_naive = sort.merge_sort_naive(a_cpu.copy())
        sorted_a_optimized,ti_opti = sort.merge_sort_optimized1(a_cpu.copy())
        del sort
        ti_c.append(ti_serial)
        ti_gn.append(ti_naive)
        ti_go.append(ti_opti)
    sorted_a_cpu=sorted_a_serial
    sorted_a_gpu_naive=sorted_a_naive
    sorted_a_gpu_optimized=sorted_a_optimized
    plt.plot(ti_c)
    plt.plot(ti_gn)
    plt.plot(ti_go)

    plt.xlabel('size of input')
    plt.ylabel('time')
    plt.title('python vs pycuda')
    plt.legend(['cpu','gpu_naive','gpu_optimized'])
    plt.savefig('pycuda_only_kernel.png')
    plt.close()

    print("CPU:",sorted_a_cpu)
    print("GPU_naive:",sorted_a_gpu_naive)
    print("GPU_optimized:",sorted_a_gpu_optimized)

    print("is Naive code correct:",sorted_a_cpu==sorted_a_gpu_naive)
    print("is optiomized code correct:",sorted_a_cpu==sorted_a_gpu_optimized)       

if __name__=='__main__':
    main()
