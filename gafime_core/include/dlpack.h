#pragma once
// Minimal, spec-correct subset of the official DLPack ABI (https://github.com/dmlc/dlpack)
// vendored for GAFIME's zero-copy framework export (__dlpack__). Field order and sizes
// match the upstream legacy DLManagedTensor that numpy>=1.23 / torch / JAX / CuPy consume
// via from_dlpack(). Only what we need is declared.

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
} DLDeviceType;

typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;  // NULL => compact row-major (C-contiguous)
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

#ifdef __cplusplus
}  // extern "C"
#endif
