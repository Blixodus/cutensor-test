#include <cassert>

#include <span>
#include <vector>
#include <numeric>
#include <iostream>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef DEBUG
constexpr bool DEBUG_MODE = true;
#else
constexpr bool DEBUG_MODE = false;
#endif

#define HANDLE_CUTENSOR_ERROR(x)                                      \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS ) [[unlikely]]                   \
  { printf("Error: %s\n", cutensorGetErrorString(err)); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess ) [[unlikely]]                           \
  { printf("Error: %s\n", cudaGetErrorString(err)); } \
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

template <typename T>
static inline std::vector<int64_t> computeLD(std::span<T const> dimSize) {
  std::vector<int64_t> ld(dimSize.size());
	ld[0] = 1;
	std::partial_sum(dimSize.begin(), dimSize.end()-1, ld.begin()+1, std::multiplies<T>());
	return ld;
}

template <typename T>
cudaDataType_t convertToCudaDataType() {
	if constexpr(std::is_same_v<T, double>) { return CUDA_R_64F; }
	if constexpr(std::is_same_v<T, float>) { return CUDA_R_32F; }
	if constexpr(std::is_same_v<T, __half>) { return CUDA_R_16F; }
	if constexpr(std::is_same_v<T, __nv_bfloat16>) { return CUDA_R_16BF; }
}

template <typename DataType>
class GPUTensor {
	template <typename D>
	friend class GPUTensor;
  
	cutensorHandle_t *handlePtr_;
	cutensorTensorDescriptor_t desc_;
	cudaDataType_t cudaDataType_;
	cutensorOperator_t unaryOp_;
	std::vector<int64_t> dimSize_;
	std::vector<int64_t> ld_;
	std::vector<int32_t> dimName_;
	uint32_t alignment_;
	size_t nelem_;
	DataType *hostDataPtr_;
	DataType *deviceDataPtr_;
	
public:
	GPUTensor(cutensorHandle_t *handlePtr,
						std::span<size_t const> dimSize,
						std::span<size_t const> dimName)
		:	handlePtr_(handlePtr),
		cudaDataType_(convertToCudaDataType<DataType>()),
		unaryOp_(CUTENSOR_OP_IDENTITY),
		dimSize_(dimSize.begin(), dimSize.end()),
		ld_(computeLD(dimSize)),
		dimName_(dimName.begin(), dimName.end()),
		nelem_(std::reduce(dimSize.begin(), dimSize.end(), static_cast<size_t>(1), std::multiplies<size_t>()))
	{
		HANDLE_CUDA_ERROR(cudaMallocHost(&hostDataPtr_, nelem_ * sizeof(DataType)));
		HANDLE_CUDA_ERROR(cudaMalloc(&deviceDataPtr_, nelem_ * sizeof(DataType)));
	  HANDLE_CUTENSOR_ERROR(cutensorInitTensorDescriptor(handlePtr_, &desc_, dimSize_.size(), dimSize_.data(), NULL, cudaDataType_, unaryOp_));
		HANDLE_CUTENSOR_ERROR(cutensorGetAlignmentRequirement(handlePtr_, deviceDataPtr_, &desc_, &alignment_));
	}

	~GPUTensor() {
		cudaFreeHost(hostDataPtr_);
		cudaFree(deviceDataPtr_);
	}

	template <typename OtherDataType>
	bool descEq(GPUTensor<OtherDataType> const& other) const {
		return ((dimSize_ == other.dimSize_) && (ld_ == other.ld_) && (cudaDataType_ == other.cudaDataType_) && (unaryOp_ == other.unaryOp_));
	}

	template <typename OtherDataType>
	bool dimNameEq(GPUTensor<OtherDataType> const& other) const {
		return (dimName_ == other.dimName_);
	}

	GPUTensor const& copyToDevice() const {
		HANDLE_CUDA_ERROR(cudaMemcpy(deviceDataPtr_, hostDataPtr_, nelem_ * sizeof(DataType), cudaMemcpyHostToDevice));
		return *this;
	}

	GPUTensor const& copyToHost() const {
		HANDLE_CUDA_ERROR(cudaMemcpy(hostDataPtr_, deviceDataPtr_, nelem_ * sizeof(DataType), cudaMemcpyDeviceToHost));
		return *this;
	}

	GPUTensor const& print(bool copy = false) const {
		if(copy) { copyToHost(); }
		std::cout << "Tensor : ";
		for(size_t i = 0; i < nelem_; i++) { std::cout << hostDataPtr_[i] << " "; }
		std::cout << std::endl;
		return *this;
	}

	GPUTensor& init(std::function<DataType(size_t)> func) {
		for(size_t i = 0; i < nelem_; i++) {
			hostDataPtr_[i] = func(i);
		}
		copyToDevice();
		if constexpr(DEBUG_MODE) {
			std::cout << "Initialized tensor : ";
			print();
			std::cout << std::endl;
		}
		return *this;
	}

	template <typename InDataType, typename OutDataType, typename ScalarDataType>
	static auto perm(GPUTensor<InDataType> const& in, ScalarDataType const scalar, GPUTensor<OutDataType>& out) {
		assert(in.handlePtr_ == out.handlePtr_);
		if constexpr(DEBUG_MODE) {
			std::cout << "Permuting tensor : ";
			in.print(true);
		}
		GPUTimer timer;
		HANDLE_CUTENSOR_ERROR(cutensorPermutation(in.handlePtr_,
																							&scalar, in.deviceDataPtr_, &in.desc_, in.dimName_.data(),
																							out.deviceDataPtr_, &out.desc_, out.dimName_.data(),
																							convertToCudaDataType<ScalarDataType>(), 0));
		auto time = timer.seconds();
		if constexpr(DEBUG_MODE) {
			std::cout << "Output tensor : ";
			out.print(true);
			std::cout << std::endl;
		}
		return time;
	}

	template <typename InDataType1, typename InDataType2, typename OutDataType, typename ScalarDataType>
	static auto binaryOp(GPUTensor<InDataType1> const& in1, ScalarDataType const alpha, GPUTensor<InDataType2> const& in2, ScalarDataType const gamma, GPUTensor<OutDataType>& out, cutensorOperator_t const op) {
		assert(in1.handlePtr_ == in2.handlePtr_);
		assert(in1.handlePtr_ == out.handlePtr_);
		assert(in2.descEq(out));    // Required by cuTensor
		assert(in2.dimNameEq(out)); // Required by cuTensor
		if constexpr(DEBUG_MODE) {
			std::cout << "Operation on tensors : ";
			in1.print(true);
			in2.print(true);
		}
		GPUTimer timer;
		HANDLE_CUTENSOR_ERROR(cutensorElementwiseBinary(in1.handlePtr_,
																										&alpha, in1.deviceDataPtr_, &in1.desc_, in1.dimName_.data(),
																										&gamma, in2.deviceDataPtr_, &in2.desc_, in2.dimName_.data(),
																										out.deviceDataPtr_, &in2.desc_, in2.dimName_.data(),
																										op, convertToCudaDataType<ScalarDataType>(), 0));
		auto time = timer.seconds();
		if constexpr(DEBUG_MODE) {
			std::cout << "Output tensor : ";
			out.print(true);
			std::cout << std::endl;
		}
		return time;
	}

	template <typename InDataType1, typename InDataType2, typename InDataType3, typename OutDataType, typename ScalarDataType>
	static auto trinaryOp(GPUTensor<InDataType1> const& in1, ScalarDataType const alpha, GPUTensor<InDataType2> const& in2, ScalarDataType const beta, GPUTensor<InDataType3> const& in3, ScalarDataType const gamma, GPUTensor<OutDataType>& out, cutensorOperator_t const op12, cutensorOperator_t const op123) {
		assert(in1.handlePtr_ == in2.handlePtr_);
		assert(in1.handlePtr_ == in3.handlePtr_);
		assert(in1.handlePtr_ == out.handlePtr_);
		assert(in3.descEq(out));    // Required by cuTensor
		assert(in3.dimNameEq(out)); // Required by cuTensor
		if constexpr(DEBUG_MODE) {
			std::cout << "Operation on tensors : ";
			in1.print(true);
			in2.print(true);
			in3.print(true);
		}
		GPUTimer timer;
		HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinary(in1.handlePtr_,
																										 &alpha, in1.deviceDataPtr_, &in1.desc_, in1.dimName_.data(),
																										 &beta, in2.deviceDataPtr_, &in2.desc_, in2.dimName_.data(),
																										 &gamma, in3.deviceDataPtr_, &in3.desc_, in3.dimName_.data(),
																										 out.deviceDataPtr_, &in3.desc_, in3.dimName_.data(),
																										 op12, op123, convertToCudaDataType<ScalarDataType>(), 0));
		auto time = timer.seconds();
		if constexpr(DEBUG_MODE) {
			std::cout << "Output tensor : ";
			out.print(true);
			std::cout << std::endl;
		}
		return time;
	}
};

template <typename DataType>
void testPerm(std::vector<size_t>& dimNameIn, std::vector<size_t>& dimNameOut, std::unordered_map<size_t, size_t>& nameToSize) {
	cutensorHandle_t *handle;
	HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));
	
	std::vector<size_t> dimSizeIn, dimSizeOut;
	for(auto name : dimNameIn) { dimSizeIn.push_back(nameToSize[name]); }
	for(auto name : dimNameOut) { dimSizeOut.push_back(nameToSize[name]); }
	auto nelem = std::reduce(dimSizeIn.begin(), dimSizeIn.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  
	std::cout << "Creating tensors of size " << double(nelem)/(1024*1024*1024)*sizeof(DataType) << "GB" << std::endl;
	GPUTensor<DataType> In(handle, dimSizeIn, dimNameIn);
	GPUTensor<DataType> Out(handle, dimSizeOut, dimNameOut);
	In.init([](size_t i) -> DataType { return i; });
	Out.init([](size_t i) -> DataType { (void)i; return 0; });
	
	std::cout << "Starting measurement" << std::endl;
	auto time = GPUTensor<DataType>::perm(In, 1.0, Out);
	std::cout << "Permutation done in " << time << "s" << std::endl;
	std::cout << "Permutation done at " << double(nelem)/(time*1024*1024*1024)*sizeof(DataType) << "GB/s" << std::endl;

	cutensorDestroy(handle);
}

int main() {
	cutensorHandle_t *handle;
	HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));

	std::vector<size_t> dimNameIn{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'};
	std::vector<size_t>	dimNameOut{'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'};
	std::unordered_map<size_t, size_t> nameToSize;
	nameToSize['a'] = 4;
	nameToSize['b'] = 4;
	nameToSize['c'] = 4;
	nameToSize['d'] = 4;
	nameToSize['e'] = 4;
	nameToSize['f'] = 4;
	nameToSize['g'] = 4;
	nameToSize['h'] = 4;
	nameToSize['i'] = 4;
	nameToSize['j'] = 4;
	nameToSize['k'] = 4;
	nameToSize['l'] = 4;
	nameToSize['m'] = 4;
	nameToSize['n'] = 4;
	nameToSize['o'] = 4;

	testPerm<double>(dimNameIn, dimNameOut, nameToSize);
	
	// std::vector<size_t> dimNameIn{'a', 'b', 'c', 'd', 'e', 'f'}, dimNameOut{'f', 'e', 'd', 'c', 'b', 'a'};
	// std::unordered_map<size_t, size_t> nameToSize;
	// nameToSize['a'] = 64;
	// nameToSize['b'] = 64;
	// nameToSize['c'] = 64;
	// nameToSize['d'] = 64;
	// nameToSize['e'] = 10;
	// nameToSize['f'] = 8;

	// testPerm<double>(dimNameIn, dimNameOut, nameToSize);
	
	// std::unordered_map<size_t, size_t> nameToSize;
	// nameToSize['m'] = 4;
	// nameToSize['n'] = 8;
	// nameToSize['k'] = 6;
	
	// std::vector<size_t> dimSizeA, dimSizeB;
	// std::vector<size_t> dimNameA{'m', 'n', 'k'}, dimNameB{'n', 'm', 'k'};

	// for(size_t i = 0; i < dimNameA.size(); i++) {
	// 	dimSizeA.push_back(nameToSize[dimNameA[i]]);
	// 	dimSizeB.push_back(nameToSize[dimNameB[i]]);
	// }
	
	// GPUTensor<float> A(handle, dimSizeA, dimNameA);
	// GPUTensor<float> B(handle, dimSizeB, dimNameB);
	// GPUTensor<float> C(handle, dimSizeB, dimNameB);

	// A.init([](size_t i) -> float { return i; });
	// B.init([](size_t i) -> float { (void)i; return 0; });
	// C.init([](size_t i) -> float { return i+1; });

	// GPUTensor<float>::perm(A, 1.0f, B);
	// GPUTensor<float>::binaryOp(A, 1.0f, B, 1.0f, C, CUTENSOR_OP_ADD);
  
	// std::vector<size_t> dimSizeD;
	// std::vector<size_t> dimNameD{'k', 'm', 'n'};

	// for(auto n : dimNameD) {
	// 	dimSizeD.push_back(nameToSize[n]);
	// }

	// GPUTensor<float> D(handle, dimSizeD, dimNameD);

	// D.init([](size_t i) -> float { return i+2; });

	// GPUTensor<float>::trinaryOp(A, 1.0f, D, 1.0f, B, 1.0f, C, CUTENSOR_OP_ADD, CUTENSOR_OP_ADD);

	cutensorDestroy(handle);
	return 0;
}