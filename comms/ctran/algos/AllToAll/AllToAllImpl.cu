// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllToAll/AllToAll.cuh"

// Explicit template instantiations for AllToAll kernels
DECL_CTRAN_ALLTOALL_KERN(int8_t);
DECL_CTRAN_ALLTOALL_KERN(uint8_t);
DECL_CTRAN_ALLTOALL_KERN(int32_t);
DECL_CTRAN_ALLTOALL_KERN(uint32_t);
DECL_CTRAN_ALLTOALL_KERN(int64_t);
DECL_CTRAN_ALLTOALL_KERN(uint64_t);
DECL_CTRAN_ALLTOALL_KERN(half);
DECL_CTRAN_ALLTOALL_KERN(float);
DECL_CTRAN_ALLTOALL_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__) || defined(__HIP_PLATFORM_AMD__)
DECL_CTRAN_ALLTOALL_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_CTRAN_ALLTOALL_KERN(__nv_fp8_e4m3);
DECL_CTRAN_ALLTOALL_KERN(__nv_fp8_e5m2);
#endif