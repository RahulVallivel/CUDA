import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
 
# -- initialize the device
import pycuda.autoinit



class cudaModule:
    def __init__(self, idata):
        # idata: an array of lower characters.
        # TODO:
        # Declare host variables
        self.a_cpu = idata   #copying the input data to variable
        self.times_gpu = 0  #initialising variable to measure gpu time
        self.times_cpu = 0  #initialising variable to measure cpu time
        # Device memory allocation
        self.a_gpu = gpuarray.to_gpu(self.a_cpu)  #copy a to device
        self.b_gpu = gpuarray.empty((self.a_cpu.shape), self.a_cpu.dtype)  #create an empty array in device for the result
        # Kernel code
        self.kernel_code = """
        __global__ void capitalize(char *a, char *b)
        {
        int tx = threadIdx.x;
        b[tx] = a[tx]-32;
        }
        """
   




    def runAdd_parallel(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # Memory copy to device
        # Function call and measuring time here
        mod = compiler.SourceModule(self.kernel_code)  #compiling the kernel
        Capitalize = mod.get_function("capitalize") #getting the function

        start = time.time() #start timer
        Capitalize(
        # inputs
        self.a_gpu,
        # output
        self.b_gpu,
        # block size
        block = (1024, 1, 1),
        
        

        )      #kernel call
        self.times_gpu=time.time()-start #get time
        # Memory copy to host
        b_cpu=self.b_gpu.get()  #copy result to host variable
        # Return output and measured time
        return b_cpu,self.times_gpu #return result and time
    def runAdd_serial(self):
        #return: an array containing capitalized characters from idata and running time.
        output=[] #initialise variable for result
        
        for i in self.a_cpu:
            k=ord(i)  #get ascii value of each character
            start = time.time() #start timer
            k=k-32 #subtract 32 from ascii value to capitalise
            self.times_cpu=self.times_cpu + time.time()-start #get time
            i=chr(k) #convert modified ascii value back to character
            output.append(i) #append the results
        
        return output,self.times_cpu #return the output and time

def main():
    times_cpu=[] #initialise variable to copy time
    times_gpu=[]
    test=0 #test variable to check if cpu time is greater than gpu time

    for itr in range(1, 40):
        idata = list("abcdefghijklmnopqrstuvwxyz"*itr) #extend the array(input data)
        idata=np.array(idata) #convert list to numpy array
    ##############################################################################################
    #   capitalize idata using your serial and parallel functions, record the running time here  #
        p=cudaModule(idata) #create class object
        gpu_output,times_gpu_=p.runAdd_parallel() #call function for gpu execution
        cpu_output,times_cpu_=p.runAdd_serial() #call function for cpu execution
        times_gpu.append(times_gpu_) #append time
        times_cpu.append(times_cpu_)

    ##############################################################################################
        print 'py_output=\n', cpu_output # py_output is the output of your serial function
        print 'parallel_output=\n', gpu_output # parallel_output is the output of your parallel function
        print 'Code equality:\t', (cpu_output==gpu_output) #check if cpu_output and gpu_output are same
        print 'string_len=', len(idata), '\tpy_time: ', times_cpu[itr-1], '\tparallel_time: ', times_gpu[itr-1] # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.
    
    for i in range(len(times_cpu)):
        if(times_cpu[i]>=times_gpu[i]):
            print('the L_CUDA value is:{}'.format(i+1)) #calculating the iteration at which the cpu executon time becomes greater than gpu execution time
            test=test+1
            break
    if(test==0):
        print('CPU execution time did not cross GPU execution time, Increase the number of iterations to observe the GPU perform better than the CPU')
    
    plt.plot(times_gpu)
    plt.plot(times_cpu)
    plt.xlabel('iterations')
    plt.ylabel('time(seconds)')
    plt.legend(['pycuda_gpu','python_cpu'])
    plt.savefig('time_pycuda.png')
    plt.close()

if __name__ == '__main__':
    main()


        
