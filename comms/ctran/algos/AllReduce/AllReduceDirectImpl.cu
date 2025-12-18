// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllReduce/AllReduceDirect.cuh"

// Explicit template instantiations for AllReduceDirect kernels
// For each data type, instantiate for all reduction operations: Sum, Prod, Avg, Max, Min

#define INSTANTIATE_ALLREDUCE_DIRECT(T) \
  DECL_CTRAN_ALLREDUCEDIRECT_KERN(T, commSum); \
  DECL_CTRAN_ALLREDUCEDIRECT_KERN(T, commProd); \
  DECL_CTRAN_ALLREDUCEDIRECT_KERN(T, commAvg); \
  DECL_CTRAN_ALLREDUCEDIRECT_KERN(T, commMax); \
  DECL_CTRAN_ALLREDUCEDIRECT_KERN(T, commMin);

INSTANTIATE_ALLREDUCE_DIRECT(int8_t);
INSTANTIATE_ALLREDUCE_DIRECT(uint8_t);
INSTANTIATE_ALLREDUCE_DIRECT(int32_t);
INSTANTIATE_ALLREDUCE_DIRECT(uint32_t);
INSTANTIATE_ALLREDUCE_DIRECT(int64_t);
INSTANTIATE_ALLREDUCE_DIRECT(uint64_t);
INSTANTIATE_ALLREDUCE_DIRECT(half);
INSTANTIATE_ALLREDUCE_DIRECT(float);
INSTANTIATE_ALLREDUCE_DIRECT(double);
#if defined(__CUDA_BF16_TYPES_EXIST__) || defined(__HIP_PLATFORM_AMD__)
INSTANTIATE_ALLREDUCE_DIRECT(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
INSTANTIATE_ALLREDUCE_DIRECT(__nv_fp8_e4m3);
INSTANTIATE_ALLREDUCE_DIRECT(__nv_fp8_e5m2);
#endif