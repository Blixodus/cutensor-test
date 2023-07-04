#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};

template <typename DataType, typename SizeType = size_t>
class GPUTensor {
	cutensorHandle_t handle_;
	DataType *dataPtr_;
	std::vector<SizeType> dimSize_;
	std::vector<SizeType> ld_;
	std::vector<SizeType> dimName_;
	
public:
	GPUTensor(cutensorHandle_t handle, DataType *dataPtr, std::span<SizeType> dimSize, std::span<SizeType> ld, std::span<SizeType> dimName)
		: handle_(handle)
	{
		
	}
	
};

int main() {
	cutensorHandle_t handle;
	

	
	return 0;
}