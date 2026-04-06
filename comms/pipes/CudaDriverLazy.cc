// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/CudaDriverLazy.h"

#include <cstdio>
#include <mutex>

namespace comms::pipes {

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

// ROCm: assign directly to HIP driver functions (no lazy loading needed,
// there is no libcuda.so.1 dependency to avoid on ROCm).
decltype(&hipDeviceGet) pfn_cuDeviceGet = &hipDeviceGet;
decltype(&hipDeviceGetAttribute) pfn_cuDeviceGetAttribute = &hipDeviceGetAttribute;
decltype(&hipCtxGetCurrent) pfn_cuCtxGetCurrent = &hipCtxGetCurrent;
decltype(&hipDrvGetErrorString) pfn_cuGetErrorString = &hipDrvGetErrorString;
decltype(&hipMemCreate) pfn_cuMemCreate = &hipMemCreate;
decltype(&hipMemRelease) pfn_cuMemRelease = &hipMemRelease;
decltype(&hipMemAddressReserve) pfn_cuMemAddressReserve = &hipMemAddressReserve;
decltype(&hipMemAddressFree) pfn_cuMemAddressFree = &hipMemAddressFree;
decltype(&hipMemMap) pfn_cuMemMap = &hipMemMap;
decltype(&hipMemUnmap) pfn_cuMemUnmap = &hipMemUnmap;
decltype(&hipMemSetAccess) pfn_cuMemSetAccess = &hipMemSetAccess;
decltype(&hipMemGetAllocationGranularity) pfn_cuMemGetAllocationGranularity = &hipMemGetAllocationGranularity;
decltype(&hipMemExportToShareableHandle) pfn_cuMemExportToShareableHandle = &hipMemExportToShareableHandle;
decltype(&hipMemImportFromShareableHandle) pfn_cuMemImportFromShareableHandle = &hipMemImportFromShareableHandle;
decltype(&hipMemGetAllocationPropertiesFromHandle) pfn_cuMemGetAllocationPropertiesFromHandle = &hipMemGetAllocationPropertiesFromHandle;
decltype(&hipMemRetainAllocationHandle) pfn_cuMemRetainAllocationHandle = &hipMemRetainAllocationHandle;
decltype(&hipMemGetAddressRange) pfn_cuMemGetAddressRange = &hipMemGetAddressRange;

int cuda_driver_lazy_init() {
  return 0;
}

#else

// CUDA: lazy-load driver symbols via cudaGetDriverEntryPoint
PFN_cuDeviceGet_v2000 pfn_cuDeviceGet = nullptr;
PFN_cuDeviceGetAttribute_v2000 pfn_cuDeviceGetAttribute = nullptr;
PFN_cuCtxGetCurrent_v4000 pfn_cuCtxGetCurrent = nullptr;
PFN_cuGetErrorString_v6000 pfn_cuGetErrorString = nullptr;
PFN_cuMemCreate_v10020 pfn_cuMemCreate = nullptr;
PFN_cuMemRelease_v10020 pfn_cuMemRelease = nullptr;
PFN_cuMemAddressReserve_v10020 pfn_cuMemAddressReserve = nullptr;
PFN_cuMemAddressFree_v10020 pfn_cuMemAddressFree = nullptr;
PFN_cuMemMap_v10020 pfn_cuMemMap = nullptr;
PFN_cuMemUnmap_v10020 pfn_cuMemUnmap = nullptr;
PFN_cuMemSetAccess_v10020 pfn_cuMemSetAccess = nullptr;
PFN_cuMemGetAllocationGranularity_v10020 pfn_cuMemGetAllocationGranularity =
    nullptr;
PFN_cuMemExportToShareableHandle_v10020 pfn_cuMemExportToShareableHandle =
    nullptr;
PFN_cuMemImportFromShareableHandle_v10020 pfn_cuMemImportFromShareableHandle =
    nullptr;
PFN_cuMemGetAllocationPropertiesFromHandle_v10020
    pfn_cuMemGetAllocationPropertiesFromHandle = nullptr;
PFN_cuMemRetainAllocationHandle_v11000 pfn_cuMemRetainAllocationHandle =
    nullptr;
PFN_cuMemGetAddressRange_v3020 pfn_cuMemGetAddressRange = nullptr;

namespace {

std::once_flag init_flag;
int init_result = -1;

int load_sym(const char* name, void** ptr) {
  cudaDriverEntryPointQueryResult status;
  auto res = cudaGetDriverEntryPoint(name, ptr, cudaEnableDefault, &status);
  if (res != cudaSuccess || status != cudaDriverEntryPointSuccess) {
    fprintf(
        stderr,
        "pipes: failed to resolve CUDA driver symbol %s "
        "(cudaError=%d, status=%d)\n",
        name,
        static_cast<int>(res),
        static_cast<int>(status));
    return -1;
  }
  return 0;
}

void do_init() {
#define LOAD(symbol)                                                     \
  if (load_sym(#symbol, reinterpret_cast<void**>(&pfn_##symbol)) != 0) { \
    init_result = -1;                                                    \
    return;                                                              \
  }

  LOAD(cuDeviceGet);
  LOAD(cuDeviceGetAttribute);
  LOAD(cuCtxGetCurrent);
  LOAD(cuGetErrorString);
  LOAD(cuMemCreate);
  LOAD(cuMemRelease);
  LOAD(cuMemAddressReserve);
  LOAD(cuMemAddressFree);
  LOAD(cuMemMap);
  LOAD(cuMemUnmap);
  LOAD(cuMemSetAccess);
  LOAD(cuMemGetAllocationGranularity);
  LOAD(cuMemExportToShareableHandle);
  LOAD(cuMemImportFromShareableHandle);
  LOAD(cuMemGetAllocationPropertiesFromHandle);
  LOAD(cuMemRetainAllocationHandle);
  LOAD(cuMemGetAddressRange);

#undef LOAD

  init_result = 0;
}

} // namespace

int cuda_driver_lazy_init() {
  std::call_once(init_flag, do_init);
  return init_result;
}

#endif

} // namespace comms::pipes
