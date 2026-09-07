#include "precision_kernels.cuh"
#include "cuda_internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/covariance_policy.hpp"
#include "../common/gpu_abi_impl.hpp"
#include "../common/gafime_gpu_internal_abi.hpp"
#include "../common/semantic_primitives_abi_impl.hpp"

#ifndef GAFIME_GPU_MI_ACCUMULATION_FP64
#define GAFIME_GPU_MI_ACCUMULATION_FP64 0
#endif

#ifndef GAFIME_CUDA_LOCAL_DEVICE_FLAGS
#define GAFIME_CUDA_LOCAL_DEVICE_FLAGS 0u
#endif

namespace {

constexpr uint64_t kPrecisionCudaMatrixMagic = 0x4741465052454332ull;  // "GAFPREC2"
constexpr uint64_t kSemanticCudaBankMagic = 0x47414653454D4231ull;  // "GAFSEMB1"

struct PrecisionGraphChunkShape {
    uint32_t arity;
    uint32_t family;
    uint32_t scaled_covariance;
    uint64_t descriptor_offset;
    uint64_t combo_count;
};

struct PrecisionCudaMatrix;

struct SemanticCudaBank {
    uint64_t magic;
    GafimePrecisionProfile profile;
    const gafime_cuda_v1::CudaSemanticKernelSet* kernels;
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t architecture_class;
    gafime_cuda_v1::CudaKernelLaunchPolicy launch_policy;
    uint64_t rows;
    uint32_t source_slots;
    uint32_t slot_capacity;
    GafimeNumericRoute route;
    void* columns;
    bool sources_uploaded;
    std::vector<uint8_t> initialized_slots;
};

struct PrecisionHostOps {
    int (*upload_f32)(PrecisionCudaMatrix*, const float*, const float*, uint64_t, uint32_t);
    int (*upload_f64)(PrecisionCudaMatrix*, const double*, const double*, uint64_t, uint32_t);
    int (*update_f32)(PrecisionCudaMatrix*, const float*, uint64_t);
    int (*update_f64)(PrecisionCudaMatrix*, const double*, uint64_t);
    int (*execute_f32)(
        PrecisionCudaMatrix*, const GafimePrecisionLaunchProtocol*, GafimeResultTable*
    );
    int (*execute_f64)(
        PrecisionCudaMatrix*, const GafimePrecisionLaunchProtocol*, GafimeResultTableF64*
    );
    int (*permutation_f32)(
        PrecisionCudaMatrix*, const GafimePrecisionLaunchProtocol*,
        GafimePermutationSignificanceTable*
    );
    int (*permutation_f64)(
        PrecisionCudaMatrix*, const GafimePrecisionLaunchProtocol*,
        GafimePermutationSignificanceTableF64*
    );
};

struct PrecisionCudaMatrix {
    uint64_t magic;
    GafimePrecisionProfile profile;
    const gafime_cuda_v1::CudaPrecisionKernelSet* kernels;
    const PrecisionHostOps* host_ops;
    uint32_t device_id;
    uint32_t device_flags;
    uint64_t architecture_class;
    gafime_cuda_v1::CudaKernelLaunchPolicy launch_policy;
    uint64_t rows;
    uint32_t cols;
    // Set only for a matrix allocated through the stable ABI 1.0 adapter.
    // This keeps compatibility policy at the host boundary without creating
    // another public precision profile or resident engine.
    bool legacy_abi10;

    bool content_valid;
    bool features_are_finite;
    bool target_is_finite;
    uint64_t feature_generation;
    uint64_t target_generation;
    uint32_t feature_stats_profile;
    uint32_t target_stats_profile;

    void* features;
    void* target;
    void* column_means;
    void* target_stats;
    void* feature_stats;
    uint64_t* spearman_target_ranks_twice;
    bool spearman_target_ranks_ready;

    std::vector<float> target_host_f32;
    std::vector<double> target_host_f64;
    std::vector<int> feature_abs_exponents;
    int target_abs_exponent;

    uint32_t* combo_indices;
    uint64_t combo_capacity;
    uint32_t* metric_ids;
    uint64_t metric_id_capacity;
    uint64_t descriptor_generation;
    uint64_t descriptor_combo_len;
    uint64_t descriptor_metric_id_len;
    uint32_t descriptor_profile;
    std::vector<uint8_t> covariance_modes;

    void* metric_values;
    uint64_t metric_value_capacity;
    uint32_t* selected_indices;
    uint64_t selected_index_capacity;
    void* selected_metric_values;
    uint64_t selected_metric_value_capacity;
    void* topk_partial_scores;
    uint64_t topk_partial_score_capacity;
    uint32_t* topk_partial_indices;
    uint64_t topk_partial_index_capacity;
    void* significance_observed_values;
    uint64_t significance_observed_value_capacity;
    void* significance_metric_max;
    uint64_t significance_metric_max_capacity;
    uint32_t* significance_exceedance_counts;
    uint64_t significance_exceedance_count_capacity;
    void* permutation_target_host;
    uint64_t permutation_target_capacity;

    cudaStream_t graph_stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_valid;
    bool graph_has_target_copy;
    uint32_t graph_profile;
    uint32_t graph_chunk_count;
    uint64_t graph_combo_len;
    uint64_t graph_metric_id_len;
    uint64_t graph_metric_value_count;
    uintptr_t graph_target_copy_ptr;
    uintptr_t graph_combo_ptr;
    uintptr_t graph_metric_ids_ptr;
    uintptr_t graph_metric_values_ptr;
    uint64_t graph_metric_signature;
    std::vector<PrecisionGraphChunkShape> graph_chunk_shapes;
};

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return GAFIME_STATUS_OK;
    if (status == cudaErrorMemoryAllocation) return GAFIME_STATUS_OUT_OF_MEMORY;
    return GAFIME_STATUS_DEVICE_ERROR;
}

// CUDA's current-device selection belongs to the caller thread. Payload entry
// points borrow it only for this scope; restoration on every return path keeps
// GAFIME from leaking its matrix/device choice into the embedding application.
class ScopedCudaDevice {
public:
    explicit ScopedCudaDevice(uint32_t device_id) {
        if (device_id > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            status_ = cudaErrorInvalidDevice;
            return;
        }
        status_ = cudaGetDevice(&previous_device_);
        if (status_ != cudaSuccess) return;
        restore_previous_ = true;
        status_ = cudaSetDevice(static_cast<int>(device_id));
    }

    ScopedCudaDevice(const ScopedCudaDevice&) = delete;
    ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

    ~ScopedCudaDevice() {
        if (restore_previous_) {
            static_cast<void>(cudaSetDevice(previous_device_));
        }
    }

    cudaError_t status() const {
        return status_;
    }

private:
    int previous_device_ = 0;
    bool restore_previous_ = false;
    cudaError_t status_ = cudaSuccess;
};

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t* out) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t* out) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

bool fits_size_t(uint64_t count, size_t element_size) {
    return count <= std::numeric_limits<size_t>::max() / element_size;
}

uint64_t next_precision_generation() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

uint64_t cuda_architecture_class(const cudaDeviceProp& props) {
    if (props.major >= 10) return GAFIME_GPU_ARCH_NVIDIA_BLACKWELL;
    if (props.major == 9) return GAFIME_GPU_ARCH_NVIDIA_HOPPER;
    if (props.major == 8 && props.minor >= 9) return GAFIME_GPU_ARCH_NVIDIA_ADA;
    if (props.major == 8) return GAFIME_GPU_ARCH_NVIDIA_AMPERE;
    if (props.major == 7 && props.minor >= 5) return GAFIME_GPU_ARCH_NVIDIA_TURING;
    return static_cast<uint64_t>(props.major * 10 + props.minor);
}

int cuda_device_attr(uint32_t device_id, cudaDeviceAttr attribute) {
    int value = 0;
    return cudaDeviceGetAttribute(&value, attribute, static_cast<int>(device_id)) == cudaSuccess
        ? value
        : 0;
}

uint32_t precision_cuda_device_flags(const cudaDeviceProp& props, uint32_t device_id) {
    uint32_t flags = GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL |
        GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION;
#if GAFIME_GPU_MI_ACCUMULATION_FP64
    flags |= GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64;
#endif
    const int integrated = cuda_device_attr(device_id, cudaDevAttrIntegrated);
    const int managed = cuda_device_attr(device_id, cudaDevAttrManagedMemory);
    const int concurrent_managed = cuda_device_attr(device_id, cudaDevAttrConcurrentManagedAccess);
    const int unified = cuda_device_attr(device_id, cudaDevAttrUnifiedAddressing);
    const int memory_bus_width = cuda_device_attr(device_id, cudaDevAttrGlobalMemoryBusWidth);
    const int l2_cache_size = cuda_device_attr(device_id, cudaDevAttrL2CacheSize);
    if (props.integrated != 0 || integrated != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_INTEGRATED | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    } else {
        flags |= GAFIME_GPU_DEVICE_FLAG_DISCRETE;
    }
    if (props.managedMemory != 0 || props.concurrentManagedAccess != 0 ||
        managed != 0 || concurrent_managed != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY;
    }
    if (props.unifiedAddressing != 0 || unified != 0) flags |= GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    if (memory_bus_width >= 384 || l2_cache_size >= (40 * 1024 * 1024)) {
        flags |= GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH;
    }
    flags |= static_cast<uint32_t>(GAFIME_CUDA_LOCAL_DEVICE_FLAGS);
    return flags;
}

bool profile_is_valid(uint32_t profile) {
    return profile == GAFIME_PRECISION_FP32 || profile == GAFIME_PRECISION_MIXED ||
        profile == GAFIME_PRECISION_FP64;
}

bool profile_dtype_matches(GafimePrecisionProfile profile, uint32_t dtype) {
    switch (profile) {
    case GAFIME_PRECISION_FP32:
    case GAFIME_PRECISION_MIXED:
        return dtype == GAFIME_DTYPE_F32;
    case GAFIME_PRECISION_FP64:
        return dtype == GAFIME_DTYPE_F64;
    default:
        return false;
    }
}

size_t dtype_size(uint32_t dtype) {
    return dtype == GAFIME_DTYPE_F32 ? sizeof(float) :
        dtype == GAFIME_DTYPE_F64 ? sizeof(double) : 0;
}

void destroy_precision_graph(PrecisionCudaMatrix* matrix) {
    if (matrix == nullptr) return;
    if (matrix->graph_exec != nullptr) {
        static_cast<void>(cudaGraphExecDestroy(matrix->graph_exec));
        matrix->graph_exec = nullptr;
    }
    if (matrix->graph != nullptr) {
        static_cast<void>(cudaGraphDestroy(matrix->graph));
        matrix->graph = nullptr;
    }
    matrix->graph_valid = false;
    matrix->graph_has_target_copy = false;
    matrix->graph_profile = 0;
    matrix->graph_chunk_count = 0;
    matrix->graph_combo_len = 0;
    matrix->graph_metric_id_len = 0;
    matrix->graph_metric_value_count = 0;
    matrix->graph_target_copy_ptr = 0;
    matrix->graph_combo_ptr = 0;
    matrix->graph_metric_ids_ptr = 0;
    matrix->graph_metric_values_ptr = 0;
    matrix->graph_metric_signature = 0;
    matrix->graph_chunk_shapes.clear();
}

void invalidate_precision_descriptors(PrecisionCudaMatrix* matrix) {
    matrix->descriptor_generation = 0;
    matrix->descriptor_combo_len = 0;
    matrix->descriptor_metric_id_len = 0;
    matrix->descriptor_profile = 0;
    matrix->covariance_modes.clear();
}

int require_precision_matrix(const PrecisionCudaMatrix* matrix) {
    return matrix != nullptr && matrix->magic == kPrecisionCudaMatrixMagic && matrix->content_valid &&
        matrix->feature_stats_profile == static_cast<uint32_t>(matrix->profile) &&
        matrix->target_stats_profile == static_cast<uint32_t>(matrix->profile)
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_INVALID_ARGUMENT;
}

int ensure_device_buffer(void** pointer, uint64_t* capacity, uint64_t required, size_t element_size) {
    if (required <= *capacity) return GAFIME_STATUS_OK;
    if (!fits_size_t(required, element_size)) return GAFIME_STATUS_OUT_OF_MEMORY;
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / element_size;
    const uint64_t doubled = *capacity > max_capacity / 2 ? max_capacity : *capacity * 2;
    const uint64_t next_capacity = std::max(required, *capacity == 0 ? required : doubled);
    void* next = nullptr;
    const int status = cuda_status(cudaMalloc(&next, static_cast<size_t>(next_capacity) * element_size));
    if (status != GAFIME_STATUS_OK) return status;
    static_cast<void>(cudaFree(*pointer));
    *pointer = next;
    *capacity = next_capacity;
    return GAFIME_STATUS_OK;
}

template <typename T>
class PrecisionCudaTransientBuffer {
public:
    PrecisionCudaTransientBuffer() = default;
    PrecisionCudaTransientBuffer(const PrecisionCudaTransientBuffer&) = delete;
    PrecisionCudaTransientBuffer& operator=(const PrecisionCudaTransientBuffer&) = delete;

    ~PrecisionCudaTransientBuffer() {
        static_cast<void>(cudaFree(pointer_));
    }

    int allocate(uint64_t count) {
        if (count == 0) return GAFIME_STATUS_OK;
        if (!fits_size_t(count, sizeof(T))) return GAFIME_STATUS_OUT_OF_MEMORY;
        return cuda_status(cudaMalloc(
            reinterpret_cast<void**>(&pointer_), static_cast<size_t>(count) * sizeof(T)));
    }

    T* get() const { return pointer_; }

private:
    T* pointer_ = nullptr;
};

template <typename T>
int update_exponent(int current, T value) {
    if (!std::isfinite(value) || value == static_cast<T>(0)) return current;
    // Keep the fp32 lane's numerical diagnostic in fp32 too.  The returned
    // exponent is structural planning metadata, but it must not introduce an
    // otherwise hidden f64 value conversion into the fp32 execution path.
    return std::max(current, std::ilogb(std::fabs(value)));
}

template <typename Storage, typename Accumulation>
void build_host_resident(
    const Storage* source,
    uint64_t rows,
    uint32_t cols,
    std::vector<Storage>* resident,
    std::vector<Storage>* means,
    std::vector<int>* exponents,
    bool* finite_out
) {
    resident->assign(static_cast<size_t>(rows) * cols, static_cast<Storage>(0));
    means->assign(cols, static_cast<Storage>(0));
    exponents->assign(cols, gafime_gpu_abi::kZeroMagnitudeExponent);
    bool finite = true;
    for (uint32_t column = 0; column < cols; ++column) {
        Accumulation sum = static_cast<Accumulation>(0);
        for (uint64_t row = 0; row < rows; ++row) {
            const Storage value = source[static_cast<size_t>(row) * cols + column];
            sum += static_cast<Accumulation>(value);
            finite = finite && std::isfinite(value);
            (*exponents)[column] = update_exponent((*exponents)[column], value);
            (*resident)[static_cast<size_t>(column) * rows + row] = value;
        }
        // Mixed deliberately rounds this stored centering constant to fp32 so
        // each pointwise interaction remains a genuine fp32 operation.
        (*means)[column] = static_cast<Storage>(sum / static_cast<Accumulation>(rows));
    }
    *finite_out = finite;
}

template <typename Storage>
bool all_finite_host(const Storage* values, uint64_t count) {
    for (uint64_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) return false;
    }
    return true;
}

template <typename Storage>
int host_abs_exponent(const Storage* values, uint64_t count) {
    int exponent = gafime_gpu_abi::kZeroMagnitudeExponent;
    for (uint64_t index = 0; index < count; ++index) {
        exponent = update_exponent(exponent, values[index]);
    }
    return exponent;
}

int refresh_precision_feature_stats(PrecisionCudaMatrix* matrix, cudaStream_t stream) {
    const int status = cuda_status(matrix->kernels->feature_stats(
        matrix->features, matrix->rows, matrix->cols, matrix->feature_stats,
        matrix->launch_policy, stream));
    if (status != GAFIME_STATUS_OK) return status;
    matrix->feature_stats_profile = static_cast<uint32_t>(matrix->profile);
    return cuda_status(cudaStreamSynchronize(stream));
}

int refresh_precision_target_stats(PrecisionCudaMatrix* matrix, cudaStream_t stream) {
    const int status = cuda_status(matrix->kernels->target_stats(
        matrix->target, matrix->rows, matrix->target_stats, matrix->launch_policy, stream));
    if (status != GAFIME_STATUS_OK) return status;
    matrix->target_stats_profile = static_cast<uint32_t>(matrix->profile);
    return cuda_status(cudaStreamSynchronize(stream));
}

bool metric_supported(uint32_t metric) {
    return metric == GAFIME_METRIC_PEARSON || metric == GAFIME_METRIC_R2 ||
        metric == GAFIME_METRIC_MUTUAL_INFO || metric == GAFIME_METRIC_SPEARMAN;
}

bool family_supported(uint32_t family) {
    return family == GAFIME_FAMILY_CONTINUOUS || family == GAFIME_FAMILY_TIME_SERIES ||
        family == GAFIME_FAMILY_DECISION_PATH;
}

uint64_t planned_row_count(const GafimeLaunchProtocol* protocol) {
    uint64_t rows = 0;
    for (uint32_t index = 0; index < protocol->chunk_count; ++index) {
        rows += protocol->chunks[index].combo_count;
    }
    return rows;
}

uint64_t output_row_count(const GafimeLaunchProtocol* protocol, uint64_t planned_rows) {
    return protocol->rank.top_k == 0 ? planned_rows :
        std::min<uint64_t>(planned_rows, protocol->rank.top_k);
}

int validate_precision_protocol(
    const GafimePrecisionLaunchProtocol* protocol,
    const PrecisionCudaMatrix* matrix
) {
    if (protocol == nullptr || matrix == nullptr ||
        protocol->abi_version != GAFIME_PRECISION_ABI_VERSION ||
        protocol->profile != static_cast<uint32_t>(matrix->profile) || protocol->base == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t reserved : protocol->reserved) {
        if (reserved != 0) return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const GafimeLaunchProtocol* base = protocol->base;
    if (base->abi_version != GAFIME_ABI_VERSION || base->backend_kind != GAFIME_BACKEND_CUDA ||
        base->n_samples != matrix->rows || base->n_features != matrix->cols ||
        base->family_count == 0 || base->family_count > 3 || base->max_arity == 0 ||
        base->max_arity > matrix->cols ||
        base->metric_ids.ptr == nullptr || base->metric_ids.len == 0 ||
        base->metric_ids.len > std::numeric_limits<uint32_t>::max() ||
        base->combo_indices.ptr == nullptr || base->chunks == nullptr || base->chunk_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    constexpr uint32_t known_flags = GAFIME_LAUNCH_FLAG_GRAPH | GAFIME_LAUNCH_FLAG_MI_APPROX |
        GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    if ((base->flags & ~known_flags) != 0 ||
        (base->shape_hint_count != 0 && base->shape_hints == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (base->rank.top_k != 0 && base->rank.include_ties != 0) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    bool primary_found = base->rank.top_k == 0;
    bool has_mutual_info = false;
    for (uint64_t index = 0; index < base->metric_ids.len; ++index) {
        if (!metric_supported(base->metric_ids.ptr[index])) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        if (base->metric_ids.ptr[index] == base->rank.primary_metric) primary_found = true;
        has_mutual_info = has_mutual_info || base->metric_ids.ptr[index] == GAFIME_METRIC_MUTUAL_INFO;
    }
    if (!primary_found) return GAFIME_STATUS_INVALID_ARGUMENT;
    // MI histograms deliberately use integer counters.  Reject a sample count
    // that cannot be represented exactly by that established control domain
    // instead of widening it into a floating-point bookkeeping path.
    if (has_mutual_info && !matrix->legacy_abi10 &&
        matrix->rows > std::numeric_limits<uint32_t>::max()) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (base->permutations.permutation_count != 0) {
        uint64_t expected_offsets = 0;
        if (!checked_mul(base->permutations.permutation_count, matrix->rows, &expected_offsets) ||
            (base->permutations.target_offsets.len != 0 &&
                base->permutations.target_offsets.len != expected_offsets) ||
            (base->permutations.target_offsets.len != 0 &&
                base->permutations.target_offsets.ptr == nullptr)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    uint64_t total_rows = 0;
    uint64_t expected_offset = 0;
    for (uint32_t chunk_index = 0; chunk_index < base->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = base->chunks[chunk_index];
        if (!family_supported(chunk.family) || chunk.arity == 0 ||
            chunk.arity > base->max_arity || chunk.combo_count == 0 ||
            chunk.descriptor_count != chunk.combo_count || chunk.combo_row_offset != total_rows ||
            chunk.descriptor_offset != expected_offset || chunk.local_chunk_id != chunk_index ||
            (base->shape_hint_count != 0 && chunk.shape_hint_index >= base->shape_hint_count)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        uint64_t descriptor_span = 0;
        uint64_t descriptor_end = 0;
        if (!checked_mul(chunk.combo_count, chunk.arity, &descriptor_span) ||
            !checked_add(chunk.descriptor_offset, descriptor_span, &descriptor_end) ||
            descriptor_end > base->combo_indices.len) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint64_t descriptor = chunk.descriptor_offset; descriptor < descriptor_end; ++descriptor) {
            if (base->combo_indices.ptr[descriptor] >= matrix->cols) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
        if (chunk.combo_count > std::numeric_limits<uint32_t>::max() ||
            total_rows > std::numeric_limits<uint32_t>::max() - chunk.combo_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        total_rows += chunk.combo_count;
        expected_offset = descriptor_end;
    }
    if (expected_offset != base->combo_indices.len || !fits_size_t(base->combo_indices.len, sizeof(uint32_t)) ||
        !fits_size_t(base->metric_ids.len, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t metric_values = 0;
    if (!checked_mul(total_rows, base->metric_ids.len, &metric_values) ||
        !fits_size_t(metric_values, matrix->kernels->result_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_OK;
}

int validate_precision_result_f32(
    const GafimeLaunchProtocol* protocol, const GafimeResultTable* result
) {
    if (result == nullptr || result->abi_version != GAFIME_ABI_VERSION ||
        result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = output_row_count(protocol, planned_row_count(protocol));
    if (result->capacity < rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (rows != 0 && (result->combo_indices == nullptr || result->metric_values == nullptr ||
        result->ranks == nullptr || result->families == nullptr || result->candidate_ids == nullptr ||
        result->row_flags == nullptr)) return GAFIME_STATUS_INVALID_ARGUMENT;
    return GAFIME_STATUS_OK;
}

int validate_precision_result_f64(
    const GafimeLaunchProtocol* protocol, const GafimeResultTableF64* result
) {
    if (result == nullptr || result->abi_version != GAFIME_PRECISION_ABI_VERSION ||
        result->max_arity < protocol->max_arity || result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = output_row_count(protocol, planned_row_count(protocol));
    if (result->capacity < rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (rows != 0 && (result->combo_indices == nullptr || result->metric_values == nullptr ||
        result->ranks == nullptr || result->families == nullptr || result->candidate_ids == nullptr ||
        result->row_flags == nullptr)) return GAFIME_STATUS_INVALID_ARGUMENT;
    return GAFIME_STATUS_OK;
}

bool has_covariance_metric(const GafimeLaunchProtocol* protocol) {
    for (uint64_t index = 0; index < protocol->metric_ids.len; ++index) {
        const uint32_t metric = protocol->metric_ids.ptr[index];
        if (metric == GAFIME_METRIC_PEARSON || metric == GAFIME_METRIC_R2) return true;
    }
    return false;
}

bool has_spearman_metric(const GafimeLaunchProtocol* protocol) {
    for (uint64_t index = 0; index < protocol->metric_ids.len; ++index) {
        if (protocol->metric_ids.ptr[index] == GAFIME_METRIC_SPEARMAN) return true;
    }
    return false;
}

uint32_t mi_bins_for_chunk(const GafimeLaunchProtocol* protocol, const GafimeArityChunk& chunk) {
    uint32_t bins = 96;
    if (protocol->shape_hints != nullptr && chunk.shape_hint_index < protocol->shape_hint_count) {
        const uint32_t candidate = protocol->shape_hints[chunk.shape_hint_index].vendor_hint;
        if (candidate == 2 || candidate == 4 || candidate == 8 || candidate == 12 ||
            candidate == 16 || candidate == 24 || candidate == 32 || candidate == 48 ||
            candidate == 64 || candidate == 96) {
            bins = candidate;
        }
    }
    return bins;
}

std::vector<uint8_t> covariance_modes_for_protocol(
    const PrecisionCudaMatrix* matrix, const GafimeLaunchProtocol* protocol
) {
    std::vector<uint8_t> modes(protocol->chunk_count, 0);
    if (!has_covariance_metric(protocol)) return modes;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        for (uint64_t row = 0; row < chunk.combo_count; ++row) {
            const uint64_t descriptor = chunk.descriptor_offset + row * chunk.arity;
            const int feature_exponent = gafime_gpu_abi::interaction_abs_exponent(
                matrix->feature_abs_exponents.data(), protocol->combo_indices.ptr + descriptor, chunk.arity);
            if (gafime_gpu_abi::covariance_requires_scaled_path(
                    matrix->rows, feature_exponent, matrix->target_abs_exponent)) {
                modes[chunk_index] = 1;
                break;
            }
        }
    }
    return modes;
}

bool descriptors_resident(
    const PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol
) {
    const GafimeLaunchProtocol* base = protocol->base;
    const bool immutable = (base->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0;
    const uint64_t generation = base->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
    return immutable && generation != 0 && matrix->descriptor_profile == protocol->profile &&
        matrix->descriptor_generation == generation &&
        matrix->descriptor_combo_len == base->combo_indices.len &&
        matrix->descriptor_metric_id_len == base->metric_ids.len;
}

int prepare_precision_buffers(
    PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t metric_value_count
) {
    int status = ensure_device_buffer(
        &matrix->metric_values, &matrix->metric_value_capacity, metric_value_count,
        matrix->kernels->result_bytes);
    if (status != GAFIME_STATUS_OK) return status;
    if (cudaMemset(matrix->metric_values, 0,
            static_cast<size_t>(metric_value_count) * matrix->kernels->result_bytes) != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    if (descriptors_resident(matrix, protocol)) return GAFIME_STATUS_OK;

    // Invalidate the identity before either descriptor allocation can replace
    // its backing storage.  If the second allocation fails after the first
    // one succeeds, a retry must not mistake the new combo buffer for the
    // previously uploaded descriptor set.
    invalidate_precision_descriptors(matrix);
    const GafimeLaunchProtocol* base = protocol->base;
    std::vector<uint8_t> next_covariance_modes = covariance_modes_for_protocol(matrix, base);
    status = ensure_device_buffer(
        reinterpret_cast<void**>(&matrix->combo_indices), &matrix->combo_capacity,
        base->combo_indices.len, sizeof(uint32_t));
    if (status != GAFIME_STATUS_OK) return status;
    status = ensure_device_buffer(
        reinterpret_cast<void**>(&matrix->metric_ids), &matrix->metric_id_capacity,
        base->metric_ids.len, sizeof(uint32_t));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(
        matrix->combo_indices, base->combo_indices.ptr,
        static_cast<size_t>(base->combo_indices.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(
        matrix->metric_ids, base->metric_ids.ptr,
        static_cast<size_t>(base->metric_ids.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;

    matrix->covariance_modes.swap(next_covariance_modes);
    const bool cacheable = (base->flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL) != 0 &&
        base->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] != 0;
    // Profile is an identity component even for an uncached descriptor upload.
    // A zero generation records that it cannot be replayed; it does not erase
    // which precision-specialized descriptor buffer owns the upload.
    matrix->descriptor_profile = protocol->profile;
    if (cacheable) {
        matrix->descriptor_generation = base->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT];
        matrix->descriptor_combo_len = base->combo_indices.len;
        matrix->descriptor_metric_id_len = base->metric_ids.len;
    }
    return GAFIME_STATUS_OK;
}

uint32_t primary_metric_index(const GafimeLaunchProtocol* protocol) {
    if (protocol->rank.top_k == 0) return 0;
    for (uint32_t index = 0; index < protocol->metric_ids.len; ++index) {
        if (protocol->metric_ids.ptr[index] == protocol->rank.primary_metric) return index;
    }
    return 0;
}

uint32_t topk_partial_block_count(uint64_t row_count, uint64_t top_k, uint32_t threads) {
    if (row_count == 0 || top_k == 0 || threads == 0) return 0;
    const uint64_t by_threads = 1 + (row_count - 1) / threads;
    const uint64_t by_storage = 1 + (row_count - 1) / top_k;
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::min(by_threads, by_storage), gafime_cuda_v1::kTopKMaxPartialBlocks));
}

bool spearman_rank_cache_eligible(
    const PrecisionCudaMatrix* matrix, const GafimeLaunchProtocol* protocol
) {
    if (!matrix->features_are_finite || !matrix->target_is_finite ||
        matrix->spearman_target_ranks_twice == nullptr ||
        matrix->rows < gafime_cuda_v1::kSpearmanTargetRankCacheMinSamples ||
        matrix->rows > gafime_cuda_v1::kSpearmanTargetRankCacheMaxSamples ||
        protocol->permutations.permutation_count != 0 || !has_spearman_metric(protocol)) {
        return false;
    }
    uint64_t unary = 0;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        if (chunk.arity == 1) unary += chunk.combo_count;
    }
    return unary >= gafime_cuda_v1::kSpearmanTargetRankCacheMinUnaryCandidates;
}

int prepare_spearman_rank_cache(PrecisionCudaMatrix* matrix, const GafimeLaunchProtocol* protocol) {
    if (!spearman_rank_cache_eligible(matrix, protocol) || matrix->spearman_target_ranks_ready) {
        return GAFIME_STATUS_OK;
    }
    int status = cuda_status(matrix->kernels->build_target_ranks(
        matrix->target, matrix->rows, matrix->spearman_target_ranks_twice,
        matrix->launch_policy, nullptr));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    if (status == GAFIME_STATUS_OK) matrix->spearman_target_ranks_ready = true;
    return status;
}

int launch_precision_score_kernels(
    PrecisionCudaMatrix* matrix, const GafimeLaunchProtocol* protocol, cudaStream_t stream
) {
    const uint64_t metric_count = protocol->metric_ids.len;
    const uint64_t* cached_ranks = spearman_rank_cache_eligible(matrix, protocol) &&
        matrix->spearman_target_ranks_ready ? matrix->spearman_target_ranks_twice : nullptr;
    uint64_t metric_offset = 0;
    for (uint32_t chunk_index = 0; chunk_index < protocol->chunk_count; ++chunk_index) {
        const GafimeArityChunk& chunk = protocol->chunks[chunk_index];
        void* output = static_cast<unsigned char*>(matrix->metric_values) +
            metric_offset * metric_count * matrix->kernels->result_bytes;
        if (has_covariance_metric(protocol)) {
            if (matrix->covariance_modes.size() != protocol->chunk_count) return GAFIME_STATUS_DEVICE_ERROR;
            const bool unary_fast_path = chunk.arity == 1 &&
                matrix->covariance_modes[chunk_index] == 0 && matrix->features_are_finite &&
                matrix->target_is_finite && matrix->kernels->continuous_unary != nullptr;
            const int status = unary_fast_path
                ? cuda_status(matrix->kernels->continuous_unary(
                    matrix->features, matrix->target, matrix->target_stats, matrix->feature_stats,
                    matrix->combo_indices, matrix->rows, chunk.descriptor_offset, chunk.combo_count,
                    matrix->metric_ids, static_cast<uint32_t>(metric_count), output,
                    matrix->launch_policy, stream))
                : cuda_status(matrix->kernels->continuous(
                    matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                    matrix->rows, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                    matrix->covariance_modes[chunk_index], matrix->metric_ids,
                    static_cast<uint32_t>(metric_count), output, matrix->launch_policy, stream));
            if (status != GAFIME_STATUS_OK) return status;
        }
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            if (protocol->metric_ids.ptr[metric_index] != GAFIME_METRIC_MUTUAL_INFO) continue;
            const int status = cuda_status(matrix->kernels->mutual_info(
                matrix->features, matrix->target, matrix->column_means, matrix->combo_indices,
                matrix->rows, chunk.arity, chunk.descriptor_offset, chunk.combo_count,
                static_cast<uint32_t>(metric_count), metric_index, mi_bins_for_chunk(protocol, chunk),
                output, matrix->launch_policy, stream));
            if (status != GAFIME_STATUS_OK) return status;
        }
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            if (protocol->metric_ids.ptr[metric_index] != GAFIME_METRIC_SPEARMAN) continue;
            // A materialized higher-order interaction can discard rows after
            // fp32 overflow. Its target ranks therefore belong to that
            // candidate's finite row set, not the matrix-wide unary cache.
            const uint64_t* chunk_cached_ranks = chunk.arity == 1 ? cached_ranks : nullptr;
            const auto spearman_kernel = matrix->legacy_abi10 &&
                matrix->kernels->legacy_spearman != nullptr
                ? matrix->kernels->legacy_spearman
                : matrix->kernels->spearman;
            const int status = cuda_status(spearman_kernel(
                matrix->features, matrix->target, matrix->column_means, chunk_cached_ranks,
                matrix->combo_indices, matrix->rows, chunk.arity, chunk.descriptor_offset,
                chunk.combo_count, static_cast<uint32_t>(metric_count), metric_index,
                output, matrix->launch_policy, stream));
            if (status != GAFIME_STATUS_OK) return status;
        }
        metric_offset += chunk.combo_count;
    }
    return GAFIME_STATUS_OK;
}

uint64_t metric_signature(const GafimePrecisionLaunchProtocol* protocol) {
    const GafimeLaunchProtocol* base = protocol->base;
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(protocol->profile);
    for (uint64_t index = 0; index < base->metric_ids.len; ++index) mix(base->metric_ids.ptr[index]);
    for (uint32_t index = 0; index < base->chunk_count; ++index) {
        mix(base->chunks[index].metric_mask);
        mix(base->chunks[index].family);
        mix(base->chunks[index].arity);
        mix(mi_bins_for_chunk(base, base->chunks[index]));
    }
    return hash;
}

bool graph_requested(const GafimeLaunchProtocol* protocol) {
    return (protocol->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0 && protocol->rank.top_k == 0;
}

bool graph_shape_matches(
    const PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* protocol,
    uint64_t metric_count, bool target_copy
) {
    const GafimeLaunchProtocol* base = protocol->base;
    const uintptr_t target_pointer = target_copy
        ? reinterpret_cast<uintptr_t>(matrix->permutation_target_host) : 0;
    if (!matrix->graph_valid || matrix->graph_exec == nullptr ||
        matrix->graph_profile != protocol->profile || matrix->graph_has_target_copy != target_copy ||
        matrix->graph_target_copy_ptr != target_pointer || matrix->graph_chunk_count != base->chunk_count ||
        matrix->graph_chunk_shapes.size() != base->chunk_count ||
        matrix->graph_combo_len != base->combo_indices.len ||
        matrix->graph_metric_id_len != base->metric_ids.len ||
        matrix->graph_metric_value_count != metric_count ||
        matrix->graph_combo_ptr != reinterpret_cast<uintptr_t>(matrix->combo_indices) ||
        matrix->graph_metric_ids_ptr != reinterpret_cast<uintptr_t>(matrix->metric_ids) ||
        matrix->graph_metric_values_ptr != reinterpret_cast<uintptr_t>(matrix->metric_values) ||
        matrix->graph_metric_signature != metric_signature(protocol)) return false;
    for (uint32_t index = 0; index < base->chunk_count; ++index) {
        const auto& expected = matrix->graph_chunk_shapes[index];
        const auto& actual = base->chunks[index];
        if (expected.arity != actual.arity || expected.family != actual.family ||
            expected.scaled_covariance != matrix->covariance_modes[index] ||
            expected.descriptor_offset != actual.descriptor_offset || expected.combo_count != actual.combo_count) {
            return false;
        }
    }
    return true;
}

int store_graph_shape(
    PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* protocol,
    uint64_t metric_count, bool target_copy
) try {
    const GafimeLaunchProtocol* base = protocol->base;
    std::vector<PrecisionGraphChunkShape> shapes;
    shapes.reserve(base->chunk_count);
    for (uint32_t index = 0; index < base->chunk_count; ++index) {
        const auto& chunk = base->chunks[index];
        shapes.push_back({
            chunk.arity, chunk.family, matrix->covariance_modes[index], chunk.descriptor_offset,
            chunk.combo_count,
        });
    }
    matrix->graph_profile = protocol->profile;
    matrix->graph_has_target_copy = target_copy;
    matrix->graph_chunk_count = base->chunk_count;
    matrix->graph_combo_len = base->combo_indices.len;
    matrix->graph_metric_id_len = base->metric_ids.len;
    matrix->graph_metric_value_count = metric_count;
    matrix->graph_target_copy_ptr = target_copy ? reinterpret_cast<uintptr_t>(matrix->permutation_target_host) : 0;
    matrix->graph_combo_ptr = reinterpret_cast<uintptr_t>(matrix->combo_indices);
    matrix->graph_metric_ids_ptr = reinterpret_cast<uintptr_t>(matrix->metric_ids);
    matrix->graph_metric_values_ptr = reinterpret_cast<uintptr_t>(matrix->metric_values);
    matrix->graph_metric_signature = metric_signature(protocol);
    matrix->graph_chunk_shapes.swap(shapes);
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
}

int execute_precision_score_kernels(
    PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* protocol,
    uint64_t metric_value_count, bool* graph_replayed
) {
    *graph_replayed = false;
    const GafimeLaunchProtocol* base = protocol->base;
    int status = prepare_spearman_rank_cache(matrix, base);
    if (status != GAFIME_STATUS_OK) return status;
    if (!graph_requested(base)) {
        status = launch_precision_score_kernels(matrix, base, nullptr);
        return status == GAFIME_STATUS_OK ? cuda_status(cudaDeviceSynchronize()) : status;
    }
    if (matrix->graph_stream == nullptr) {
        status = cuda_status(cudaStreamCreateWithFlags(&matrix->graph_stream, cudaStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) return status;
    }
    if (!graph_shape_matches(matrix, protocol, metric_value_count, false)) {
        destroy_precision_graph(matrix);
        status = cuda_status(cudaStreamBeginCapture(
            matrix->graph_stream, cudaStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) return status;
        status = launch_precision_score_kernels(matrix, base, matrix->graph_stream);
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(cudaStreamEndCapture(matrix->graph_stream, &matrix->graph));
            destroy_precision_graph(matrix);
            return status;
        }
        cudaGraph_t graph = nullptr;
        status = cuda_status(cudaStreamEndCapture(matrix->graph_stream, &graph));
        if (status != GAFIME_STATUS_OK) {
            destroy_precision_graph(matrix);
            return status;
        }
        cudaGraphExec_t graph_exec = nullptr;
        status = cuda_status(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(cudaGraphDestroy(graph));
            destroy_precision_graph(matrix);
            return status;
        }
        matrix->graph = graph;
        matrix->graph_exec = graph_exec;
        status = store_graph_shape(matrix, protocol, metric_value_count, false);
        if (status != GAFIME_STATUS_OK) {
            destroy_precision_graph(matrix);
            return status;
        }
        matrix->graph_valid = true;
    }
    status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) {
        *graph_replayed = true;
    } else {
        destroy_precision_graph(matrix);
    }
    return status;
}

bool locate_combo(
    const GafimeLaunchProtocol* protocol, uint64_t global_row,
    const GafimeArityChunk** chunk_out, uint64_t* local_row_out
) {
    uint64_t offset = 0;
    for (uint32_t index = 0; index < protocol->chunk_count; ++index) {
        const auto& chunk = protocol->chunks[index];
        if (global_row < offset + chunk.combo_count) {
            *chunk_out = &chunk;
            *local_row_out = global_row - offset;
            return true;
        }
        offset += chunk.combo_count;
    }
    return false;
}

template <typename Result>
int write_result_rows(
    const GafimeLaunchProtocol* protocol,
    uint32_t max_arity,
    uint32_t metric_capacity,
    uint64_t* row_count_out,
    uint32_t* combo_indices_out,
    Result* metric_values_out,
    uint32_t* ranks_out,
    uint32_t* families_out,
    uint64_t* candidate_ids_out,
    uint32_t* flags_out,
    const std::vector<Result>& metric_values,
    const std::vector<uint32_t>* selected_indices
) {
    if (selected_indices == nullptr && protocol->rank.top_k != 0) return GAFIME_STATUS_DEVICE_ERROR;
    const uint64_t rows = selected_indices == nullptr ? planned_row_count(protocol) :
        static_cast<uint64_t>(selected_indices->size());
    for (uint64_t output_row = 0; output_row < rows; ++output_row) {
        const uint64_t global_row = selected_indices == nullptr ? output_row :
            static_cast<uint64_t>((*selected_indices)[output_row]);
        const GafimeArityChunk* chunk = nullptr;
        uint64_t local_row = 0;
        if (!locate_combo(protocol, global_row, &chunk, &local_row)) return GAFIME_STATUS_DEVICE_ERROR;
        const uint64_t combo_base = chunk->descriptor_offset + local_row * chunk->arity;
        for (uint32_t slot = 0; slot < max_arity; ++slot) {
            combo_indices_out[output_row * max_arity + slot] = slot < chunk->arity
                ? protocol->combo_indices.ptr[combo_base + slot]
                : UINT32_MAX;
        }
        for (uint32_t metric = 0; metric < metric_capacity; ++metric) {
            metric_values_out[output_row * metric_capacity + metric] = metric < protocol->metric_ids.len
                ? metric_values[output_row * protocol->metric_ids.len + metric]
                : static_cast<Result>(0);
        }
        ranks_out[output_row] = static_cast<uint32_t>(output_row);
        families_out[output_row] = chunk->family;
        candidate_ids_out[output_row] = global_row;
        flags_out[output_row] = 0;
    }
    *row_count_out = rows;
    return GAFIME_STATUS_OK;
}

void update_graph_result_flag(GafimeResultTable* result, bool replayed) {
    result->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    if (replayed) result->flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
}

void update_graph_result_flag(GafimeResultTableF64* result, bool replayed) {
    result->flags &= ~GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
    if (replayed) result->flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
}

template <GafimePrecisionProfile Profile, typename Storage, typename Accumulation>
int upload_precision(
    PrecisionCudaMatrix* matrix, const Storage* features_host, const Storage* target_host,
    uint64_t rows, uint32_t cols
) try {
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        features_host == nullptr || target_host == nullptr || rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    ScopedCudaDevice device(matrix->device_id);
    int status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    std::vector<Storage> resident;
    std::vector<Storage> means;
    std::vector<int> exponents;
    bool features_finite = true;
    // ABI 1.0 historically formed its f32 centering constants from an f64
    // host sum before narrowing them to the resident f32 buffer.  The modern
    // fp32 profile intentionally uses f32 host accumulation; retain the
    // legacy arithmetic only for the adapter lane so diagnostics and metric
    // results keep their frozen behavior without creating another engine.
    if constexpr (std::is_same_v<Storage, float>) {
        if (matrix->legacy_abi10) {
            build_host_resident<Storage, double>(
                features_host, rows, cols, &resident, &means, &exponents, &features_finite);
        } else {
            build_host_resident<Storage, Accumulation>(
                features_host, rows, cols, &resident, &means, &exponents, &features_finite);
        }
    } else {
        build_host_resident<Storage, Accumulation>(
            features_host, rows, cols, &resident, &means, &exponents, &features_finite);
    }
    const bool target_finite = all_finite_host(target_host, rows);
    const int target_exponent = host_abs_exponent(target_host, rows);
    matrix->content_valid = false;
    destroy_precision_graph(matrix);
    invalidate_precision_descriptors(matrix);
    matrix->spearman_target_ranks_ready = false;
    const size_t feature_bytes = static_cast<size_t>(rows) * cols * sizeof(Storage);
    const size_t target_bytes = static_cast<size_t>(rows) * sizeof(Storage);
    const size_t mean_bytes = static_cast<size_t>(cols) * sizeof(Storage);
    status = cuda_status(cudaMemcpy(matrix->features, resident.data(), feature_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(matrix->target, target_host, target_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(matrix->column_means, means.data(), mean_bytes, cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    status = refresh_precision_feature_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) return status;
    status = refresh_precision_target_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) return status;
    matrix->feature_generation = next_precision_generation();
    matrix->target_generation = next_precision_generation();
    matrix->features_are_finite = features_finite;
    matrix->target_is_finite = target_finite;
    matrix->feature_abs_exponents.swap(exponents);
    matrix->target_abs_exponent = target_exponent;
    if constexpr (std::is_same_v<Storage, float>) {
        matrix->target_host_f32.assign(target_host, target_host + rows);
        matrix->target_host_f64.clear();
    } else {
        matrix->target_host_f64.assign(target_host, target_host + rows);
        matrix->target_host_f32.clear();
    }
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

template <GafimePrecisionProfile Profile, typename Storage, typename Accumulation>
int update_precision_target(PrecisionCudaMatrix* matrix, const Storage* target_host, uint64_t rows) try {
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || target_host == nullptr ||
        rows != matrix->rows || !matrix->content_valid) return GAFIME_STATUS_INVALID_ARGUMENT;
    ScopedCudaDevice device(matrix->device_id);
    int status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    const bool target_finite = all_finite_host(target_host, rows);
    const int target_exponent = host_abs_exponent(target_host, rows);
    matrix->content_valid = false;
    destroy_precision_graph(matrix);
    invalidate_precision_descriptors(matrix);
    matrix->spearman_target_ranks_ready = false;
    status = cuda_status(cudaMemcpy(
        matrix->target, target_host, static_cast<size_t>(rows) * sizeof(Storage), cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    status = refresh_precision_target_stats(matrix, nullptr);
    if (status != GAFIME_STATUS_OK) return status;
    matrix->target_generation = next_precision_generation();
    matrix->target_is_finite = target_finite;
    matrix->target_abs_exponent = target_exponent;
    if constexpr (std::is_same_v<Storage, float>) {
        matrix->target_host_f32.assign(target_host, target_host + rows);
    } else {
        matrix->target_host_f64.assign(target_host, target_host + rows);
    }
    matrix->content_valid = true;
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

template <GafimePrecisionProfile Profile, typename Result, typename ResultTable>
int execute_precision(
    PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    ResultTable* result
) try {
    int status = validate_precision_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    const GafimeLaunchProtocol* base = protocol->base;
    if constexpr (std::is_same_v<ResultTable, GafimeResultTable>) {
        status = validate_precision_result_f32(base, result);
    } else {
        status = validate_precision_result_f64(base, result);
    }
    if (status != GAFIME_STATUS_OK) return status;
    status = require_precision_matrix(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    ScopedCudaDevice device(matrix->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;

    const uint64_t total_rows = planned_row_count(base);
    if (total_rows == 0) {
        result->row_count = 0;
        return GAFIME_STATUS_OK;
    }
    uint64_t metric_value_count = 0;
    if (!checked_mul(total_rows, base->metric_ids.len, &metric_value_count)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = prepare_precision_buffers(matrix, protocol, metric_value_count);
    if (status != GAFIME_STATUS_OK) return status;

    bool graph_replayed = false;
    status = execute_precision_score_kernels(matrix, protocol, metric_value_count, &graph_replayed);
    if (status != GAFIME_STATUS_OK) return status;
    if (base->rank.top_k == 0) {
        std::vector<Result> scores(static_cast<size_t>(metric_value_count), static_cast<Result>(0));
        status = cuda_status(cudaMemcpy(
            scores.data(), matrix->metric_values, static_cast<size_t>(metric_value_count) * sizeof(Result),
            cudaMemcpyDeviceToHost));
        if (status != GAFIME_STATUS_OK) return status;
        update_graph_result_flag(result, graph_replayed);
        return write_result_rows<Result>(
            base, result->max_arity, result->metric_count, &result->row_count,
            result->combo_indices, result->metric_values, result->ranks, result->families,
            result->candidate_ids, result->row_flags, scores, nullptr);
    }

    const uint64_t requested_rows = output_row_count(base, total_rows);
    status = ensure_device_buffer(
        reinterpret_cast<void**>(&matrix->selected_indices), &matrix->selected_index_capacity,
        requested_rows, sizeof(uint32_t));
    if (status != GAFIME_STATUS_OK) return status;
    const uint32_t partial_blocks = topk_partial_block_count(
        total_rows, requested_rows, matrix->launch_policy.threads_per_block);
    const uint64_t partial_items = static_cast<uint64_t>(partial_blocks) * requested_rows;
    status = ensure_device_buffer(
        &matrix->topk_partial_scores, &matrix->topk_partial_score_capacity, partial_items,
        matrix->kernels->result_bytes);
    if (status != GAFIME_STATUS_OK) return status;
    status = ensure_device_buffer(
        reinterpret_cast<void**>(&matrix->topk_partial_indices), &matrix->topk_partial_index_capacity,
        partial_items, sizeof(uint32_t));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(matrix->kernels->select_topk(
        matrix->metric_values, total_rows, static_cast<uint32_t>(base->metric_ids.len),
        primary_metric_index(base), static_cast<uint32_t>(requested_rows), base->rank.descending,
        matrix->selected_indices, matrix->topk_partial_scores, matrix->topk_partial_indices,
        partial_blocks, matrix->launch_policy, nullptr));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) return status;

    std::vector<uint32_t> selected_indices(static_cast<size_t>(requested_rows), UINT32_MAX);
    status = cuda_status(cudaMemcpy(
        selected_indices.data(), matrix->selected_indices,
        static_cast<size_t>(requested_rows) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t selected_count = 0;
    while (selected_count < requested_rows && selected_indices[selected_count] != UINT32_MAX) ++selected_count;
    selected_indices.resize(static_cast<size_t>(selected_count));
    if (selected_count == 0) {
        result->row_count = 0;
        update_graph_result_flag(result, graph_replayed);
        return GAFIME_STATUS_OK;
    }
    uint64_t selected_metric_count = 0;
    if (!checked_mul(selected_count, base->metric_ids.len, &selected_metric_count)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_device_buffer(
        &matrix->selected_metric_values, &matrix->selected_metric_value_capacity,
        selected_metric_count, matrix->kernels->result_bytes);
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(matrix->kernels->copy_selected_rows(
        matrix->metric_values, matrix->selected_indices, selected_count,
        static_cast<uint32_t>(base->metric_ids.len), matrix->selected_metric_values,
        matrix->launch_policy, nullptr));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) return status;
    std::vector<Result> selected_scores(static_cast<size_t>(selected_metric_count), static_cast<Result>(0));
    status = cuda_status(cudaMemcpy(
        selected_scores.data(), matrix->selected_metric_values,
        static_cast<size_t>(selected_metric_count) * sizeof(Result), cudaMemcpyDeviceToHost));
    if (status != GAFIME_STATUS_OK) return status;
    update_graph_result_flag(result, graph_replayed);
    return write_result_rows<Result>(
        base, result->max_arity, result->metric_count, &result->row_count,
        result->combo_indices, result->metric_values, result->ranks, result->families,
        result->candidate_ids, result->row_flags, selected_scores, &selected_indices);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

uint64_t resident_precision_bytes(const PrecisionCudaMatrix* matrix) {
    using gafime_gpu_abi::allocation_bytes;
    using gafime_gpu_abi::saturating_add_u64;
    using gafime_gpu_abi::saturating_mul_u64;
    uint64_t bytes = 0;
    const auto add = [&bytes](uint64_t count, size_t element_size) {
        bytes = saturating_add_u64(bytes, allocation_bytes(count, element_size));
    };
    add(saturating_mul_u64(matrix->rows, matrix->cols), matrix->kernels->storage_bytes);
    add(matrix->rows, matrix->kernels->storage_bytes);
    add(matrix->cols, matrix->kernels->storage_bytes);
    add(1, matrix->kernels->target_stats_bytes);
    add(matrix->cols, matrix->kernels->feature_stats_bytes);
    add(matrix->spearman_target_ranks_twice != nullptr ? matrix->rows : 0, sizeof(uint64_t));
    add(matrix->combo_capacity, sizeof(uint32_t));
    add(matrix->metric_id_capacity, sizeof(uint32_t));
    add(matrix->metric_value_capacity, matrix->kernels->result_bytes);
    add(matrix->selected_index_capacity, sizeof(uint32_t));
    add(matrix->selected_metric_value_capacity, matrix->kernels->result_bytes);
    add(matrix->topk_partial_score_capacity, matrix->kernels->result_bytes);
    add(matrix->topk_partial_index_capacity, sizeof(uint32_t));
    add(matrix->significance_observed_value_capacity, matrix->kernels->result_bytes);
    add(matrix->significance_metric_max_capacity, matrix->kernels->result_bytes);
    add(matrix->significance_exceedance_count_capacity, sizeof(uint32_t));
    return bytes;
}

void track_precision_execution_allocations(
    gafime_gpu_abi::DeviceMemoryPeakTracker* tracker,
    const PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol
) {
    using gafime_gpu_abi::saturating_mul_u64;
    const GafimeLaunchProtocol* base = protocol->base;
    const uint64_t total_rows = planned_row_count(base);
    const uint64_t metric_values = saturating_mul_u64(total_rows, base->metric_ids.len);
    tracker->grow(matrix->metric_value_capacity, metric_values, matrix->kernels->result_bytes);
    if (!descriptors_resident(matrix, protocol)) {
        tracker->reserve_pair(
            matrix->combo_capacity, base->combo_indices.len, sizeof(uint32_t),
            matrix->metric_id_capacity, base->metric_ids.len, sizeof(uint32_t));
    }
    if (base->rank.top_k == 0) return;
    const uint64_t output_rows = output_row_count(base, total_rows);
    tracker->grow(matrix->selected_index_capacity, output_rows, sizeof(uint32_t));
    const uint64_t partial_items = saturating_mul_u64(
        topk_partial_block_count(total_rows, output_rows, matrix->launch_policy.threads_per_block), output_rows);
    tracker->grow(matrix->topk_partial_score_capacity, partial_items, matrix->kernels->result_bytes);
    tracker->grow(matrix->topk_partial_index_capacity, partial_items, sizeof(uint32_t));
    tracker->grow(
        matrix->selected_metric_value_capacity,
        saturating_mul_u64(output_rows, base->metric_ids.len), matrix->kernels->result_bytes);
}

uint64_t precision_execution_peak(const PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* protocol) {
    gafime_gpu_abi::DeviceMemoryPeakTracker tracker(resident_precision_bytes(matrix));
    track_precision_execution_allocations(&tracker, matrix, protocol);
    return tracker.peak_bytes();
}

uint64_t precision_permutation_peak(
    const PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t selected_rows
) {
    using gafime_gpu_abi::saturating_mul_u64;
    gafime_gpu_abi::DeviceMemoryPeakTracker tracker(resident_precision_bytes(matrix));
    track_precision_execution_allocations(&tracker, matrix, protocol);
    const uint64_t values = saturating_mul_u64(selected_rows, protocol->base->metric_ids.len);
    tracker.grow(matrix->significance_observed_value_capacity, values, matrix->kernels->result_bytes);
    tracker.grow(matrix->significance_metric_max_capacity, protocol->base->metric_ids.len, matrix->kernels->result_bytes);
    tracker.grow(matrix->significance_exceedance_count_capacity, values, sizeof(uint32_t));
    return tracker.peak_bytes();
}

uint64_t splitmix64_next(uint64_t* state) {
    uint64_t value = (*state += 0x9E3779B97F4A7C15ull);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

uint64_t bounded_random(uint64_t* state, uint64_t bound) {
    return bound <= 1 ? 0 : splitmix64_next(state) % bound;
}

uint64_t mix_permutation_seed(
    uint64_t base_seed,
    uint64_t permutation_index,
    bool legacy_abi10
) {
    if (legacy_abi10) {
        // Frozen ABI 1.0 performs one SplitMix step before the Fisher-Yates
        // draws.  Keep that exact sequence in the thin adapter even though
        // ABI 1.1 established a different route-owned seed mixer.
        uint64_t state = base_seed ^
            0xA5A5A5A5ull * 0x9E3779B97F4A7C15ull ^
            permutation_index * 0xD1B54A32D192ED03ull;
        return splitmix64_next(&state);
    }
    return base_seed ^ permutation_index * 0xD1B54A32D192ED03ull ^
        0xA5A5A5A59E3779B9ull;
}

template <typename Storage>
const std::vector<Storage>& precision_target_host(const PrecisionCudaMatrix* matrix) {
    if constexpr (std::is_same_v<Storage, float>) {
        return matrix->target_host_f32;
    } else {
        return matrix->target_host_f64;
    }
}

template <typename Storage>
int ensure_pinned_target(PrecisionCudaMatrix* matrix, uint64_t required) {
    if (required <= matrix->permutation_target_capacity) return GAFIME_STATUS_OK;
    if (!fits_size_t(required, sizeof(Storage))) return GAFIME_STATUS_OUT_OF_MEMORY;
    const uint64_t max_capacity = std::numeric_limits<size_t>::max() / sizeof(Storage);
    const uint64_t doubled = matrix->permutation_target_capacity > max_capacity / 2
        ? max_capacity
        : matrix->permutation_target_capacity * 2;
    const uint64_t next_capacity = std::max(
        required, matrix->permutation_target_capacity == 0 ? required : doubled);
    void* next = nullptr;
    int status = cuda_status(cudaHostAlloc(
        &next, static_cast<size_t>(next_capacity) * sizeof(Storage), cudaHostAllocDefault));
    if (status != GAFIME_STATUS_OK) return status;
    static_cast<void>(cudaFreeHost(matrix->permutation_target_host));
    matrix->permutation_target_host = next;
    matrix->permutation_target_capacity = next_capacity;
    destroy_precision_graph(matrix);
    return GAFIME_STATUS_OK;
}

template <typename Storage>
int fill_permutation_target(
    PrecisionCudaMatrix* matrix, const GafimeLaunchProtocol* protocol, uint32_t permutation_index
) try {
    const auto& target = precision_target_host<Storage>(matrix);
    if (target.size() != matrix->rows || matrix->permutation_target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    auto* out = static_cast<Storage*>(matrix->permutation_target_host);
    const uint64_t rows = matrix->rows;
    if (protocol->permutations.target_offsets.len != 0) {
        const uint64_t base = static_cast<uint64_t>(permutation_index) * rows;
        for (uint64_t row = 0; row < rows; ++row) {
            const uint64_t source = protocol->permutations.target_offsets.ptr[base + row];
            if (source >= rows) return GAFIME_STATUS_INVALID_ARGUMENT;
            out[row] = target[source];
        }
        return GAFIME_STATUS_OK;
    }
    std::vector<uint64_t> order(static_cast<size_t>(rows));
    for (uint64_t row = 0; row < rows; ++row) order[row] = row;
    uint64_t state = mix_permutation_seed(
        protocol->permutations.seed,
        static_cast<uint64_t>(permutation_index),
        matrix->legacy_abi10);
    for (uint64_t row = rows; row > 1; --row) {
        std::swap(order[row - 1], order[bounded_random(&state, row)]);
    }
    for (uint64_t row = 0; row < rows; ++row) out[row] = target[order[row]];
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
}

template <typename Storage>
int execute_permutation_iteration(
    PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    uint64_t metric_value_count,
    bool use_graph
) {
    const GafimeLaunchProtocol* base = protocol->base;
    const size_t target_bytes = static_cast<size_t>(matrix->rows) * sizeof(Storage);
    if (!use_graph) {
        int status = cuda_status(cudaMemcpy(
            matrix->target, matrix->permutation_target_host, target_bytes, cudaMemcpyHostToDevice));
        if (status != GAFIME_STATUS_OK) return status;
        status = launch_precision_score_kernels(matrix, base, nullptr);
        return status == GAFIME_STATUS_OK ? cuda_status(cudaDeviceSynchronize()) : status;
    }
    if (matrix->graph_stream == nullptr) {
        const int status = cuda_status(cudaStreamCreateWithFlags(&matrix->graph_stream, cudaStreamNonBlocking));
        if (status != GAFIME_STATUS_OK) return status;
    }
    if (!graph_shape_matches(matrix, protocol, metric_value_count, true)) {
        destroy_precision_graph(matrix);
        int status = cuda_status(cudaStreamBeginCapture(
            matrix->graph_stream, cudaStreamCaptureModeThreadLocal));
        if (status != GAFIME_STATUS_OK) return status;
        status = cuda_status(cudaMemcpyAsync(
            matrix->target, matrix->permutation_target_host, target_bytes, cudaMemcpyHostToDevice,
            matrix->graph_stream));
        if (status == GAFIME_STATUS_OK) {
            status = launch_precision_score_kernels(matrix, base, matrix->graph_stream);
        }
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(cudaStreamEndCapture(matrix->graph_stream, &matrix->graph));
            destroy_precision_graph(matrix);
            return status;
        }
        cudaGraph_t graph = nullptr;
        status = cuda_status(cudaStreamEndCapture(matrix->graph_stream, &graph));
        if (status != GAFIME_STATUS_OK) {
            destroy_precision_graph(matrix);
            return status;
        }
        cudaGraphExec_t graph_exec = nullptr;
        status = cuda_status(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        if (status != GAFIME_STATUS_OK) {
            static_cast<void>(cudaGraphDestroy(graph));
            destroy_precision_graph(matrix);
            return status;
        }
        matrix->graph = graph;
        matrix->graph_exec = graph_exec;
        status = store_graph_shape(matrix, protocol, metric_value_count, true);
        if (status != GAFIME_STATUS_OK) {
            destroy_precision_graph(matrix);
            return status;
        }
        matrix->graph_valid = true;
    }
    int status = cuda_status(cudaGraphLaunch(matrix->graph_exec, matrix->graph_stream));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(matrix->graph_stream));
    if (status != GAFIME_STATUS_OK) destroy_precision_graph(matrix);
    return status;
}

template <typename Storage, typename Result, typename SignificanceTable>
int validate_precision_significance(
    const GafimeLaunchProtocol* protocol,
    const PrecisionCudaMatrix* matrix,
    const SignificanceTable* significance,
    uint64_t total_rows
) {
    const uint32_t expected_abi = std::is_same_v<Result, float>
        ? GAFIME_ABI_VERSION
        : GAFIME_PRECISION_ABI_VERSION;
    if (significance == nullptr || significance->abi_version != expected_abi ||
        protocol->permutations.permutation_count == 0 ||
        significance->metric_count != protocol->metric_ids.len || significance->row_count > total_rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->row_count == 0) return GAFIME_STATUS_OK;
    if (significance->candidate_ids == nullptr || significance->observed_metric_values == nullptr ||
        significance->p_values == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    uint64_t values = 0;
    if (!checked_mul(significance->row_count, significance->metric_count, &values) ||
        !fits_size_t(values, sizeof(Result)) || !fits_size_t(values, sizeof(uint32_t))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    for (uint64_t row = 0; row < significance->row_count; ++row) {
        if (significance->candidate_ids[row] >= total_rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const bool target_present = std::is_same_v<Storage, float>
        ? matrix->target_host_f32.size() == matrix->rows
        : matrix->target_host_f64.size() == matrix->rows;
    return target_present ? GAFIME_STATUS_OK : GAFIME_STATUS_INVALID_ARGUMENT;
}

template <GafimePrecisionProfile Profile, typename Storage, typename Result, typename SignificanceTable>
int permutation_precision(
    PrecisionCudaMatrix* matrix,
    const GafimePrecisionLaunchProtocol* protocol,
    SignificanceTable* significance
) try {
    int status = validate_precision_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    status = require_precision_matrix(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    const GafimeLaunchProtocol* base = protocol->base;
    const uint64_t total_rows = planned_row_count(base);
    status = validate_precision_significance<Storage, Result>(base, matrix, significance, total_rows);
    if (status != GAFIME_STATUS_OK) return status;
    if (total_rows == 0 || significance->row_count == 0) return GAFIME_STATUS_OK;
    ScopedCudaDevice device(matrix->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t metric_values = 0;
    if (!checked_mul(total_rows, base->metric_ids.len, &metric_values)) return GAFIME_STATUS_OUT_OF_MEMORY;
    status = prepare_precision_buffers(matrix, protocol, metric_values);
    if (status != GAFIME_STATUS_OK) return status;
    status = ensure_pinned_target<Storage>(matrix, matrix->rows);
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t observed_values = 0;
    if (!checked_mul(significance->row_count, base->metric_ids.len, &observed_values)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_device_buffer(
        &matrix->significance_observed_values, &matrix->significance_observed_value_capacity,
        observed_values, sizeof(Result));
    if (status != GAFIME_STATUS_OK) return status;
    status = ensure_device_buffer(
        &matrix->significance_metric_max, &matrix->significance_metric_max_capacity,
        base->metric_ids.len, sizeof(Result));
    if (status != GAFIME_STATUS_OK) return status;
    status = ensure_device_buffer(
        reinterpret_cast<void**>(&matrix->significance_exceedance_counts),
        &matrix->significance_exceedance_count_capacity, observed_values, sizeof(uint32_t));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(
        matrix->significance_observed_values, significance->observed_metric_values,
        static_cast<size_t>(observed_values) * sizeof(Result), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(
            matrix->significance_exceedance_counts, 0,
            static_cast<size_t>(observed_values) * sizeof(uint32_t)));
    }
    if (status != GAFIME_STATUS_OK) return status;

    matrix->content_valid = false;
    const bool use_graph = (base->flags & GAFIME_LAUNCH_FLAG_GRAPH) != 0;
    for (uint32_t permutation = 0; permutation < base->permutations.permutation_count; ++permutation) {
        status = fill_permutation_target<Storage>(matrix, base, permutation);
        if (status != GAFIME_STATUS_OK) break;
        status = execute_permutation_iteration<Storage>(matrix, protocol, metric_values, use_graph);
        if (status != GAFIME_STATUS_OK) break;
        status = cuda_status(matrix->kernels->selected_metric_max(
            matrix->metric_values, total_rows, matrix->metric_ids,
            static_cast<uint32_t>(base->metric_ids.len), matrix->significance_metric_max,
            matrix->launch_policy, nullptr));
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(matrix->kernels->accumulate_exceedances(
                matrix->significance_metric_max, matrix->metric_ids,
                static_cast<uint32_t>(base->metric_ids.len), matrix->significance_observed_values,
                significance->row_count, matrix->significance_exceedance_counts,
                matrix->launch_policy, nullptr));
        }
        if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
        if (status != GAFIME_STATUS_OK) break;
    }
    const auto& original_target = precision_target_host<Storage>(matrix);
    const int restore_status = cuda_status(cudaMemcpy(
        matrix->target, original_target.data(), static_cast<size_t>(matrix->rows) * sizeof(Storage),
        cudaMemcpyHostToDevice));
    if (restore_status == GAFIME_STATUS_OK) {
        matrix->content_valid = true;
        matrix->spearman_target_ranks_ready = false;
        const int stats_status = refresh_precision_target_stats(matrix, nullptr);
        if (stats_status != GAFIME_STATUS_OK) return stats_status;
    } else {
        destroy_precision_graph(matrix);
    }
    if (status != GAFIME_STATUS_OK) return status;
    if (restore_status != GAFIME_STATUS_OK) return restore_status;
    std::vector<uint32_t> counts(static_cast<size_t>(observed_values), 0);
    status = cuda_status(cudaMemcpy(
        counts.data(), matrix->significance_exceedance_counts,
        counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (status != GAFIME_STATUS_OK) return status;
    const Result denominator = static_cast<Result>(base->permutations.permutation_count) +
        static_cast<Result>(1);
    for (uint64_t index = 0; index < observed_values; ++index) {
        significance->p_values[index] =
            (static_cast<Result>(counts[index]) + static_cast<Result>(1)) / denominator;
    }
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

int fp32_upload_f32(PrecisionCudaMatrix* matrix, const float* x, const float* y, uint64_t rows, uint32_t cols) {
    return upload_precision<GAFIME_PRECISION_FP32, float, float>(matrix, x, y, rows, cols);
}
int mixed_upload_f32(PrecisionCudaMatrix* matrix, const float* x, const float* y, uint64_t rows, uint32_t cols) {
    return upload_precision<GAFIME_PRECISION_MIXED, float, double>(matrix, x, y, rows, cols);
}
int fp64_upload_f64(PrecisionCudaMatrix* matrix, const double* x, const double* y, uint64_t rows, uint32_t cols) {
    return upload_precision<GAFIME_PRECISION_FP64, double, double>(matrix, x, y, rows, cols);
}
int fp32_update_f32(PrecisionCudaMatrix* matrix, const float* y, uint64_t rows) {
    return update_precision_target<GAFIME_PRECISION_FP32, float, float>(matrix, y, rows);
}
int mixed_update_f32(PrecisionCudaMatrix* matrix, const float* y, uint64_t rows) {
    return update_precision_target<GAFIME_PRECISION_MIXED, float, double>(matrix, y, rows);
}
int fp64_update_f64(PrecisionCudaMatrix* matrix, const double* y, uint64_t rows) {
    return update_precision_target<GAFIME_PRECISION_FP64, double, double>(matrix, y, rows);
}
int fp32_execute_f32(PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* p, GafimeResultTable* r) {
    return execute_precision<GAFIME_PRECISION_FP32, float>(matrix, p, r);
}
int mixed_execute_f64(PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* p, GafimeResultTableF64* r) {
    return execute_precision<GAFIME_PRECISION_MIXED, double>(matrix, p, r);
}
int fp64_execute_f64(PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* p, GafimeResultTableF64* r) {
    return execute_precision<GAFIME_PRECISION_FP64, double>(matrix, p, r);
}
int fp32_permutation_f32(
    PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* p,
    GafimePermutationSignificanceTable* s
) {
    return permutation_precision<GAFIME_PRECISION_FP32, float, float>(matrix, p, s);
}
int mixed_permutation_f64(
    PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* p,
    GafimePermutationSignificanceTableF64* s
) {
    return permutation_precision<GAFIME_PRECISION_MIXED, float, double>(matrix, p, s);
}
int fp64_permutation_f64(
    PrecisionCudaMatrix* matrix, const GafimePrecisionLaunchProtocol* p,
    GafimePermutationSignificanceTableF64* s
) {
    return permutation_precision<GAFIME_PRECISION_FP64, double, double>(matrix, p, s);
}

const PrecisionHostOps* precision_host_ops(GafimePrecisionProfile profile) {
    static const PrecisionHostOps fp32{
        fp32_upload_f32, nullptr, fp32_update_f32, nullptr, fp32_execute_f32, nullptr,
        fp32_permutation_f32, nullptr,
    };
    static const PrecisionHostOps mixed{
        mixed_upload_f32, nullptr, mixed_update_f32, nullptr, nullptr, mixed_execute_f64,
        nullptr, mixed_permutation_f64,
    };
    static const PrecisionHostOps fp64{
        nullptr, fp64_upload_f64, nullptr, fp64_update_f64, nullptr, fp64_execute_f64,
        nullptr, fp64_permutation_f64,
    };
    switch (profile) {
    case GAFIME_PRECISION_FP32: return &fp32;
    case GAFIME_PRECISION_MIXED: return &mixed;
    case GAFIME_PRECISION_FP64: return &fp64;
    default: return nullptr;
    }
}

int validate_precision_matrix_desc(const GafimePrecisionMatrixDesc* desc) {
    if (desc == nullptr || desc->abi_version != GAFIME_PRECISION_ABI_VERSION ||
        !profile_is_valid(desc->profile) ||
        !profile_dtype_matches(static_cast<GafimePrecisionProfile>(desc->profile), desc->dtype) ||
        desc->layout != GAFIME_MATRIX_ROW_MAJOR || desc->flags != 0 || desc->reserved32 != 0 ||
        desc->rows == 0 || desc->cols == 0 || desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t reserved : desc->reserved) {
        if (reserved != 0) return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t count = 0;
    uint64_t bytes = 0;
    const size_t element_size = dtype_size(desc->dtype);
    if (element_size == 0 || !checked_mul(desc->rows, desc->cols, &count) ||
        !checked_mul(count, element_size, &bytes) || desc->bytes != bytes ||
        !fits_size_t(count, element_size) || !fits_size_t(desc->rows, element_size) ||
        !fits_size_t(desc->cols, element_size)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_legacy_matrix_desc(const GafimeMatrixDesc* desc) {
    if (desc == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (desc->abi_version != GAFIME_ABI_VERSION) return GAFIME_STATUS_ABI_MISMATCH;
    // Preserve ABI 1.0's distinction between an unsupported storage domain
    // and malformed dimensions/flags.
    if (desc->dtype != GAFIME_DTYPE_F32 || desc->layout != GAFIME_MATRIX_ROW_MAJOR) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (desc->rows == 0 || desc->cols == 0 || desc->flags != 0 ||
        desc->row_stride != desc->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t feature_count = 0;
    uint64_t feature_bytes = 0;
    if (!checked_mul(desc->rows, desc->cols, &feature_count) ||
        !checked_mul(feature_count, sizeof(float), &feature_bytes) ||
        !fits_size_t(feature_count, sizeof(float)) ||
        !fits_size_t(desc->rows, sizeof(float)) ||
        !fits_size_t(desc->cols, sizeof(float))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (desc->bytes != feature_bytes) return GAFIME_STATUS_INVALID_ARGUMENT;
    return GAFIME_STATUS_OK;
}

bool legacy_metric_supported(uint32_t metric) {
    return metric == GAFIME_METRIC_PEARSON || metric == GAFIME_METRIC_SPEARMAN ||
        metric == GAFIME_METRIC_MUTUAL_INFO || metric == GAFIME_METRIC_R2;
}

int validate_legacy_protocol(
    const GafimeLaunchProtocol* protocol,
    const PrecisionCudaMatrix* matrix,
    GafimePrecisionLaunchProtocol* internal_out
) {
    if (internal_out == nullptr || protocol == nullptr || matrix == nullptr ||
        matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->profile != GAFIME_PRECISION_FP32 || !matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (protocol->family_count != 1) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->chunks == nullptr || protocol->chunk_count == 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // The legacy validator returned INVALID_ARGUMENT for arithmetic-size
    // overflow before any engine admission was attempted.
    uint64_t total_rows = 0;
    for (uint32_t index = 0; index < protocol->chunk_count; ++index) {
        const GafimeArityChunk& chunk = protocol->chunks[index];
        if (chunk.family != GAFIME_FAMILY_CONTINUOUS) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (!checked_add(total_rows, chunk.combo_count, &total_rows)) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    uint64_t metric_value_count = 0;
    if (!checked_mul(total_rows, protocol->metric_ids.len, &metric_value_count) ||
        !fits_size_t(metric_value_count, sizeof(float))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (protocol->metric_ids.ptr != nullptr) {
        for (uint64_t index = 0; index < protocol->metric_ids.len; ++index) {
            if (!legacy_metric_supported(protocol->metric_ids.ptr[index])) {
                return GAFIME_STATUS_UNSUPPORTED_BACKEND;
            }
        }
    }
    internal_out->abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal_out->profile = GAFIME_PRECISION_FP32;
    internal_out->base = protocol;
    return validate_precision_protocol(internal_out, matrix);
}

int validate_legacy_result_table(
    const GafimeLaunchProtocol* protocol,
    const GafimeResultTable* result
) {
    if (result == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (result->abi_version != GAFIME_ABI_VERSION) return GAFIME_STATUS_ABI_MISMATCH;
    if (result->max_arity < protocol->max_arity ||
        result->metric_count < protocol->metric_ids.len) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t rows = output_row_count(protocol, planned_row_count(protocol));
    if (result->capacity < rows) return GAFIME_STATUS_INVALID_ARGUMENT;
    uint64_t combo_count = 0;
    uint64_t metric_count = 0;
    if (!checked_mul(rows, result->max_arity, &combo_count) ||
        !checked_mul(rows, result->metric_count, &metric_count) ||
        !fits_size_t(combo_count, sizeof(uint32_t)) ||
        !fits_size_t(metric_count, sizeof(float)) ||
        !fits_size_t(rows, sizeof(uint64_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != 0 && (result->combo_indices == nullptr || result->metric_values == nullptr ||
        result->ranks == nullptr || result->families == nullptr ||
        result->candidate_ids == nullptr || result->row_flags == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int validate_legacy_significance(
    const GafimeLaunchProtocol* protocol,
    const PrecisionCudaMatrix* matrix,
    const GafimePermutationSignificanceTable* significance,
    uint64_t total_rows
) {
    if (significance == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (significance->abi_version != GAFIME_ABI_VERSION) return GAFIME_STATUS_ABI_MISMATCH;
    if (protocol->permutations.permutation_count == 0 ||
        significance->metric_count != protocol->metric_ids.len ||
        significance->row_count > total_rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (significance->row_count == 0) return GAFIME_STATUS_OK;
    if (significance->candidate_ids == nullptr ||
        significance->observed_metric_values == nullptr ||
        significance->p_values == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t selected_metric_count = 0;
    if (!checked_mul(significance->row_count, significance->metric_count, &selected_metric_count) ||
        !fits_size_t(selected_metric_count, sizeof(float)) ||
        !fits_size_t(selected_metric_count, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t row = 0; row < significance->row_count; ++row) {
        if (significance->candidate_ids[row] >= total_rows) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (matrix == nullptr || matrix->target_host_f32.size() != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return GAFIME_STATUS_OK;
}

int precision_interaction_diagnostics(
    PrecisionCudaMatrix* matrix,
    GafimeInteractionDiagnosticBatch* diagnostics
) try {
    int status = require_precision_matrix(matrix);
    if (status != GAFIME_STATUS_OK || diagnostics == nullptr) {
        return status == GAFIME_STATUS_OK ? GAFIME_STATUS_INVALID_ARGUMENT : status;
    }
    if (diagnostics->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (diagnostics->max_arity == 0 || diagnostics->max_arity > 5 ||
        diagnostics->reserved32 != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t reserved : diagnostics->reserved) {
        if (reserved != 0) return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t expected_combo_count = 0;
    if (!checked_mul(
            diagnostics->row_count,
            static_cast<uint64_t>(diagnostics->max_arity),
            &expected_combo_count) || diagnostics->combo_index_count != expected_combo_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!fits_size_t(diagnostics->row_count, sizeof(uint64_t)) ||
        !fits_size_t(diagnostics->row_count, sizeof(uint32_t)) ||
        !fits_size_t(diagnostics->combo_index_count, sizeof(uint32_t))) {
        return matrix->legacy_abi10 ? GAFIME_STATUS_OUT_OF_MEMORY : GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (diagnostics->row_count == 0) return GAFIME_STATUS_OK;
    if (diagnostics->row_count > std::numeric_limits<unsigned int>::max() ||
        diagnostics->combo_indices == nullptr ||
        diagnostics->overflow_row_counts == nullptr || diagnostics->flags == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t row = 0; row < diagnostics->row_count; ++row) {
        const uint32_t* combo = diagnostics->combo_indices +
            static_cast<size_t>(row * diagnostics->max_arity);
        uint32_t arity = 0;
        bool saw_padding = false;
        for (uint32_t index = 0; index < diagnostics->max_arity; ++index) {
            const uint32_t column = combo[index];
            if (column == UINT32_MAX) {
                saw_padding = true;
                continue;
            }
            if (saw_padding || column >= matrix->cols) return GAFIME_STATUS_INVALID_ARGUMENT;
            ++arity;
        }
        if (arity == 0) return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    std::fill_n(
        diagnostics->overflow_row_counts,
        static_cast<size_t>(diagnostics->row_count),
        uint64_t{0});
    std::fill_n(
        diagnostics->flags,
        static_cast<size_t>(diagnostics->row_count),
        uint32_t{0});
    ScopedCudaDevice device(matrix->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;

    PrecisionCudaTransientBuffer<uint32_t> device_combos;
    PrecisionCudaTransientBuffer<uint64_t> device_overflow_rows;
    PrecisionCudaTransientBuffer<uint32_t> device_flags;
    status = device_combos.allocate(diagnostics->combo_index_count);
    if (status == GAFIME_STATUS_OK) {
        status = device_overflow_rows.allocate(diagnostics->row_count);
    }
    if (status == GAFIME_STATUS_OK) {
        status = device_flags.allocate(diagnostics->row_count);
    }
    if (status != GAFIME_STATUS_OK) return status;

    status = cuda_status(cudaMemcpy(
        device_combos.get(), diagnostics->combo_indices,
        static_cast<size_t>(diagnostics->combo_index_count) * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(matrix->kernels->interaction_diagnostics(
        matrix->features,
        matrix->target,
        matrix->column_means,
        device_combos.get(),
        diagnostics->row_count,
        matrix->rows,
        diagnostics->max_arity,
        device_overflow_rows.get(),
        device_flags.get(),
        matrix->launch_policy,
        nullptr));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(
        diagnostics->overflow_row_counts,
        device_overflow_rows.get(),
        static_cast<size_t>(diagnostics->row_count) * sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            diagnostics->flags,
            device_flags.get(),
            static_cast<size_t>(diagnostics->row_count) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
    }
    return status;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

void free_precision_matrix(PrecisionCudaMatrix* matrix) {
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic) return;
    ScopedCudaDevice device(matrix->device_id);
    destroy_precision_graph(matrix);
    if (matrix->graph_stream != nullptr) static_cast<void>(cudaStreamDestroy(matrix->graph_stream));
    static_cast<void>(cudaFree(matrix->features));
    static_cast<void>(cudaFree(matrix->target));
    static_cast<void>(cudaFree(matrix->column_means));
    static_cast<void>(cudaFree(matrix->target_stats));
    static_cast<void>(cudaFree(matrix->feature_stats));
    static_cast<void>(cudaFree(matrix->spearman_target_ranks_twice));
    static_cast<void>(cudaFree(matrix->combo_indices));
    static_cast<void>(cudaFree(matrix->metric_ids));
    static_cast<void>(cudaFree(matrix->metric_values));
    static_cast<void>(cudaFree(matrix->selected_indices));
    static_cast<void>(cudaFree(matrix->selected_metric_values));
    static_cast<void>(cudaFree(matrix->topk_partial_scores));
    static_cast<void>(cudaFree(matrix->topk_partial_indices));
    static_cast<void>(cudaFree(matrix->significance_observed_values));
    static_cast<void>(cudaFree(matrix->significance_metric_max));
    static_cast<void>(cudaFree(matrix->significance_exceedance_counts));
    static_cast<void>(cudaFreeHost(matrix->permutation_target_host));
    matrix->magic = 0;
    delete matrix;
}

SemanticCudaBank* semantic_bank_from_handle(GafimeGpuSemanticBank handle) {
    auto* bank = static_cast<SemanticCudaBank*>(handle);
    return bank != nullptr && bank->magic == kSemanticCudaBankMagic ? bank : nullptr;
}

int require_semantic_bank(const SemanticCudaBank* bank) {
    return bank != nullptr && bank->magic == kSemanticCudaBankMagic && bank->kernels != nullptr &&
            bank->columns != nullptr && profile_is_valid(static_cast<uint32_t>(bank->profile))
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_INVALID_ARGUMENT;
}

bool semantic_slots_initialized(
    const SemanticCudaBank* bank,
    const uint32_t* slots,
    uint64_t count
) {
    if (bank == nullptr || slots == nullptr && count != 0) return false;
    for (uint64_t index = 0; index < count; ++index) {
        if (slots[index] >= bank->initialized_slots.size() || !bank->initialized_slots[slots[index]]) {
            return false;
        }
    }
    return true;
}

int free_semantic_cuda_bank(SemanticCudaBank* bank) {
    if (bank == nullptr || bank->magic != kSemanticCudaBankMagic) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // A failed device switch must not free through the caller's previous
    // device. Keep the handle intact on failure so direct ABI callers receive
    // the failure and retain its ownership instead of a false successful free.
    if (bank->columns != nullptr) {
        ScopedCudaDevice device(bank->device_id);
        int status = cuda_status(device.status());
        if (status != GAFIME_STATUS_OK) return status;
        status = cuda_status(cudaFree(bank->columns));
        if (status != GAFIME_STATUS_OK) return status;
    }
    bank->columns = nullptr;
    bank->magic = 0;
    delete bank;
    return GAFIME_STATUS_OK;
}

int cuda_semantic_bank_alloc_internal(
    uint32_t device_id,
    const GafimeSemanticBankDesc* desc,
    GafimeGpuSemanticBank* bank_out
) {
    if (bank_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *bank_out = nullptr;
    int status = gafime_semantic_abi::validate_bank_desc(desc);
    if (status != GAFIME_STATUS_OK) return status;
    ScopedCudaDevice device(device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    cudaDeviceProp props{};
    status = cuda_status(cudaGetDeviceProperties(&props, static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) return status;
    const auto policy = gafime_cuda_v1::cuda_kernel_launch_policy_for_device(
        static_cast<uint32_t>(props.major), static_cast<uint32_t>(props.maxThreadsPerBlock));
    if (!gafime_cuda_v1::cuda_kernel_launch_policy_supported(policy)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    const auto profile = static_cast<GafimePrecisionProfile>(desc->route.profile);
    const auto* kernels = gafime_cuda_v1::cuda_semantic_kernel_set(profile);
    if (kernels == nullptr || kernels->storage_bytes != dtype_size(desc->route.storage_dtype)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    // Structural allocation may throw before cudaMalloc; keep even this
    // partially initialized host owner under RAII until publishing its handle.
    std::unique_ptr<SemanticCudaBank, decltype(&free_semantic_cuda_bank)> bank(
        new SemanticCudaBank{}, &free_semantic_cuda_bank);
    bank->magic = kSemanticCudaBankMagic;
    bank->profile = profile;
    bank->kernels = kernels;
    bank->device_id = device_id;
    bank->device_flags = precision_cuda_device_flags(props, device_id);
    bank->architecture_class = cuda_architecture_class(props);
    bank->launch_policy = policy;
    bank->rows = desc->rows;
    bank->source_slots = desc->source_slots;
    bank->slot_capacity = desc->slot_capacity;
    bank->route = desc->route;
    bank->initialized_slots.assign(desc->slot_capacity, 0);
    status = cuda_status(cudaMalloc(&bank->columns, static_cast<size_t>(desc->bytes)));
    if (status != GAFIME_STATUS_OK) return status;
    *bank_out = static_cast<GafimeGpuSemanticBank>(bank.release());
    return GAFIME_STATUS_OK;
}

int cuda_semantic_bank_upload_internal(
    SemanticCudaBank* bank,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* source_columns
) {
    int status = require_semantic_bank(bank);
    if (status != GAFIME_STATUS_OK) return status;
    // A semantic bank is one immutable source-content epoch. Re-uploading
    // would leave already materialized derived slots stale, so fail before
    // any device transfer or initialized-state mutation.
    if (bank->sources_uploaded) return GAFIME_STATUS_INVALID_ARGUMENT;
    status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (!gafime_gpu_abi::route_fields_equal(*route, bank->route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t elements = 0;
    if (!checked_mul(bank->rows, bank->source_slots, &elements)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_gpu_abi::validate_const_buffer(
        source_columns, route->storage_dtype, elements);
    if (status != GAFIME_STATUS_OK) return status;
    ScopedCudaDevice device(bank->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    if (elements != 0) {
        status = cuda_status(cudaMemcpy(
            bank->columns,
            source_columns->data,
            static_cast<size_t>(elements) * bank->kernels->storage_bytes,
            cudaMemcpyHostToDevice));
        if (status != GAFIME_STATUS_OK) return status;
    }
    for (uint32_t slot = 0; slot < bank->source_slots; ++slot) {
        bank->initialized_slots[slot] = 1;
    }
    bank->sources_uploaded = true;
    return GAFIME_STATUS_OK;
}

int cuda_semantic_materialize_internal(
    SemanticCudaBank* bank,
    const GafimeSemanticProgramBatch* batch
) {
    int status = require_semantic_bank(bank);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_program_batch(
        batch, bank->profile, bank->source_slots, bank->slot_capacity, bank->initialized_slots);
    if (status != GAFIME_STATUS_OK) return status;
    if (batch->node_count > gafime_semantic_abi::kSemanticMaxProgramNodes) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::vector<uint8_t> initialized = bank->initialized_slots;
    for (uint32_t index = 0; index < batch->node_count; ++index) {
        const auto& node = batch->nodes[index];
        for (uint32_t operand = 0; operand < node.operand_count; ++operand) {
            const uint32_t slot = batch->operand_slots.ptr[node.operand_offset + operand];
            if (!initialized[slot]) return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        initialized[node.output_slot] = 1;
    }
    if (batch->node_count == 0) return GAFIME_STATUS_OK;
    ScopedCudaDevice device(bank->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    PrecisionCudaTransientBuffer<uint32_t> operand_slots;
    PrecisionCudaTransientBuffer<uint64_t> mean_bits;
    // Every nonempty program batch reserves this one-word device flag so the
    // forecast has an exact, profile-independent accounting term.  Only
    // derived outputs are scanned; source nodes leave it clear.
    PrecisionCudaTransientBuffer<uint32_t> derived_nonfinite;
    // Upload each flattened descriptor exactly once before any arithmetic
    // kernel launches.  A subsequent node must never overwrite a descriptor
    // which an earlier asynchronous centered-product kernel still reads.
    status = operand_slots.allocate(batch->operand_slots.len);
    if (status == GAFIME_STATUS_OK) status = mean_bits.allocate(batch->mean_bits.len);
    if (status == GAFIME_STATUS_OK) status = derived_nonfinite.allocate(1);
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemset(derived_nonfinite.get(), 0, sizeof(uint32_t)));
    if (status == GAFIME_STATUS_OK && batch->operand_slots.len != 0) {
        status = cuda_status(cudaMemcpy(
            operand_slots.get(), batch->operand_slots.ptr,
            static_cast<size_t>(batch->operand_slots.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK && batch->mean_bits.len != 0) {
        status = cuda_status(cudaMemcpy(
            mean_bits.get(), batch->mean_bits.ptr,
            static_cast<size_t>(batch->mean_bits.len) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) return status;
    for (uint32_t index = 0; index < batch->node_count; ++index) {
        const auto& node = batch->nodes[index];
        if (node.opcode == GAFIME_SEMANTIC_PROGRAM_SOURCE) continue;
        switch (node.opcode) {
        case GAFIME_SEMANTIC_PROGRAM_ABSOLUTE_DIFFERENCE:
            status = cuda_status(bank->kernels->absolute_difference(
                bank->columns, bank->rows, batch->operand_slots.ptr[node.operand_offset],
                batch->operand_slots.ptr[node.operand_offset + 1], node.output_slot,
                bank->launch_policy, nullptr));
            break;
        case GAFIME_SEMANTIC_PROGRAM_SOFTSIGN:
            status = cuda_status(bank->kernels->softsign(
                bank->columns, bank->rows, batch->operand_slots.ptr[node.operand_offset],
                node.output_slot, bank->launch_policy, nullptr));
            break;
        case GAFIME_SEMANTIC_PROGRAM_CENTERED_PRODUCT:
            status = cuda_status(bank->kernels->centered_product(
                bank->columns, bank->rows, operand_slots.get() + node.operand_offset,
                mean_bits.get() + node.mean_offset, node.operand_count,
                node.output_slot, bank->launch_policy, nullptr));
            break;
        default: return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (status != GAFIME_STATUS_OK) return status;
        status = cuda_status(bank->kernels->reject_nonfinite_output(
            bank->columns, bank->rows, node.output_slot, derived_nonfinite.get(),
            bank->launch_policy, nullptr));
        if (status != GAFIME_STATUS_OK) return status;
    }
    status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) return status;
    uint32_t has_nonfinite = 0;
    status = cuda_status(cudaMemcpy(
        &has_nonfinite, derived_nonfinite.get(), sizeof(has_nonfinite), cudaMemcpyDeviceToHost));
    if (status != GAFIME_STATUS_OK) return status;
    if (has_nonfinite != 0) return GAFIME_STATUS_INVALID_ARGUMENT;
    bank->initialized_slots = std::move(initialized);
    return GAFIME_STATUS_OK;
}

int cuda_semantic_pairwise_pearson_internal(
    SemanticCudaBank* left,
    SemanticCudaBank* right,
    const GafimeSemanticPearsonBatch* batch,
    GafimeSemanticScalarResultTable* results
) {
    int status = require_semantic_bank(left);
    if (status != GAFIME_STATUS_OK || require_semantic_bank(right) != GAFIME_STATUS_OK) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (left->device_id != right->device_id || left->rows != right->rows ||
        !gafime_gpu_abi::route_fields_equal(left->route, right->route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_semantic_abi::validate_pearson_batch(
        batch, left->slot_capacity, right->slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_scalar_results(results, left->route, batch->left_slots.len);
    if (status != GAFIME_STATUS_OK) return status;
    if (!semantic_slots_initialized(left, batch->left_slots.ptr, batch->left_slots.len) ||
        !semantic_slots_initialized(right, batch->right_slots.ptr, batch->right_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->left_slots.len == 0) return GAFIME_STATUS_OK;
    uint64_t value_bytes = 0;
    if (!checked_mul(batch->left_slots.len, left->kernels->result_bytes, &value_bytes) ||
        !fits_size_t(value_bytes, sizeof(uint8_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    ScopedCudaDevice device(left->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    PrecisionCudaTransientBuffer<uint32_t> left_slots;
    PrecisionCudaTransientBuffer<uint32_t> right_slots;
    PrecisionCudaTransientBuffer<uint32_t> states;
    PrecisionCudaTransientBuffer<uint64_t> supports;
    PrecisionCudaTransientBuffer<uint8_t> values;
    status = left_slots.allocate(batch->left_slots.len);
    if (status == GAFIME_STATUS_OK) status = right_slots.allocate(batch->right_slots.len);
    if (status == GAFIME_STATUS_OK) status = states.allocate(batch->left_slots.len);
    if (status == GAFIME_STATUS_OK) status = supports.allocate(batch->left_slots.len);
    if (status == GAFIME_STATUS_OK) status = values.allocate(value_bytes);
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(left_slots.get(), batch->left_slots.ptr,
        static_cast<size_t>(batch->left_slots.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(right_slots.get(), batch->right_slots.ptr,
        static_cast<size_t>(batch->right_slots.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) status = cuda_status(left->kernels->pairwise_pearson(
        left->columns, right->columns, left->rows, left_slots.get(), right_slots.get(),
        batch->left_slots.len, batch->mode, values.get(), states.get(), supports.get(),
        left->launch_policy, nullptr));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(
        results->values.data, values.get(), static_cast<size_t>(value_bytes), cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(
        results->states, states.get(), static_cast<size_t>(batch->left_slots.len) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(
        results->supports, supports.get(), static_cast<size_t>(batch->left_slots.len) * sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) results->count = batch->left_slots.len;
    return status;
}

int cuda_semantic_edge_energy_internal(
    SemanticCudaBank* bank,
    const GafimeSemanticEdgeEnergyBatch* batch,
    GafimeSemanticScalarResultTable* results
) {
    int status = require_semantic_bank(bank);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_edge_energy_batch(
        batch, bank->route, bank->rows, bank->slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_scalar_results(
        results, bank->route, batch->candidate_slots.len);
    if (status != GAFIME_STATUS_OK) return status;
    if (!semantic_slots_initialized(bank, batch->candidate_slots.ptr, batch->candidate_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->candidate_slots.len == 0) return GAFIME_STATUS_OK;
    uint64_t value_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!checked_mul(batch->candidate_slots.len, bank->kernels->result_bytes, &value_bytes) ||
        !checked_mul(batch->edge_count, bank->kernels->storage_bytes, &weight_bytes) ||
        !fits_size_t(value_bytes, sizeof(uint8_t)) || !fits_size_t(weight_bytes, sizeof(uint8_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    ScopedCudaDevice device(bank->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    PrecisionCudaTransientBuffer<uint32_t> candidates;
    PrecisionCudaTransientBuffer<GafimeSemanticEdge> edges;
    PrecisionCudaTransientBuffer<uint8_t> weights;
    PrecisionCudaTransientBuffer<uint8_t> values;
    PrecisionCudaTransientBuffer<uint32_t> states;
    PrecisionCudaTransientBuffer<uint64_t> supports;
    status = candidates.allocate(batch->candidate_slots.len);
    if (status == GAFIME_STATUS_OK) status = edges.allocate(batch->edge_count);
    if (status == GAFIME_STATUS_OK) status = weights.allocate(weight_bytes);
    if (status == GAFIME_STATUS_OK) status = values.allocate(value_bytes);
    if (status == GAFIME_STATUS_OK) status = states.allocate(batch->candidate_slots.len);
    if (status == GAFIME_STATUS_OK) status = supports.allocate(batch->candidate_slots.len);
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(candidates.get(), batch->candidate_slots.ptr,
        static_cast<size_t>(batch->candidate_slots.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK && batch->edge_count != 0) status = cuda_status(cudaMemcpy(
        edges.get(), batch->edges, static_cast<size_t>(batch->edge_count) * sizeof(GafimeSemanticEdge),
        cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK && weight_bytes != 0) status = cuda_status(cudaMemcpy(
        weights.get(), batch->weights.data, static_cast<size_t>(weight_bytes), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) status = cuda_status(bank->kernels->ordered_edge_energy(
        bank->columns, bank->rows, candidates.get(), batch->candidate_slots.len, edges.get(), weights.get(),
        batch->edge_count, values.get(), states.get(), supports.get(), bank->launch_policy, nullptr));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(
        results->values.data, values.get(), static_cast<size_t>(value_bytes), cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(
        results->states, states.get(), static_cast<size_t>(batch->candidate_slots.len) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(
        results->supports, supports.get(), static_cast<size_t>(batch->candidate_slots.len) * sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    if (status == GAFIME_STATUS_OK) results->count = batch->candidate_slots.len;
    return status;
}

int cuda_semantic_sparse_gather_internal(
    SemanticCudaBank* source,
    SemanticCudaBank* destination,
    const GafimeSemanticSparseGatherBatch* batch
) {
    int status = require_semantic_bank(source);
    if (status != GAFIME_STATUS_OK || require_semantic_bank(destination) != GAFIME_STATUS_OK) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (source == destination) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (source->device_id != destination->device_id ||
        !gafime_gpu_abi::route_fields_equal(source->route, destination->route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_semantic_abi::validate_gather_batch(
        batch, source->rows, source->slot_capacity, destination->rows, destination->slot_capacity);
    if (status != GAFIME_STATUS_OK) return status;
    if (batch->row_indices.len > gafime_semantic_abi::kSemanticMaxGatherRows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!semantic_slots_initialized(source, batch->source_slots.ptr, batch->source_slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint64_t index = 0; index < batch->destination_slots.len; ++index) {
        const uint32_t slot = batch->destination_slots.ptr[index];
        if (destination->initialized_slots[slot]) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    if (batch->source_slots.len == 0 || batch->row_indices.len == 0) return GAFIME_STATUS_OK;
    ScopedCudaDevice device(source->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    PrecisionCudaTransientBuffer<uint32_t> source_slots;
    PrecisionCudaTransientBuffer<uint32_t> destination_slots;
    PrecisionCudaTransientBuffer<uint64_t> rows;
    status = source_slots.allocate(batch->source_slots.len);
    if (status == GAFIME_STATUS_OK) status = destination_slots.allocate(batch->destination_slots.len);
    if (status == GAFIME_STATUS_OK) status = rows.allocate(batch->row_indices.len);
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaMemcpy(source_slots.get(), batch->source_slots.ptr,
        static_cast<size_t>(batch->source_slots.len) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(destination_slots.get(),
        batch->destination_slots.ptr, static_cast<size_t>(batch->destination_slots.len) * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMemcpy(rows.get(), batch->row_indices.ptr,
        static_cast<size_t>(batch->row_indices.len) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) status = cuda_status(source->kernels->sparse_gather(
        source->columns, source->rows, destination->columns, destination->rows, source_slots.get(),
        destination_slots.get(), batch->source_slots.len, rows.get(), source->launch_policy, nullptr));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) return status;
    for (uint64_t index = 0; index < batch->destination_slots.len; ++index) {
        destination->initialized_slots[batch->destination_slots.ptr[index]] = 1;
    }
    return GAFIME_STATUS_OK;
}

int cuda_semantic_forecast_internal(
    SemanticCudaBank* bank,
    const GafimeSemanticForecastRequest* request,
    GafimeSemanticMemoryForecast* forecast
) {
    int status = require_semantic_bank(bank);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_forecast_request(request);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_forecast(forecast);
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t resident = 0;
    uint64_t pair_bytes = 0;
    uint64_t graph_bytes = 0;
    uint64_t retained = 0;
    uint64_t program_operand_bytes = 0;
    uint64_t program_mean_bytes = 0;
    uint64_t program_descriptor_bytes = 0;
    uint64_t program_nonfinite_flag_bytes = 0;
    uint64_t gather_descriptor_bytes = 0;
    uint64_t pair_slot_bytes = 0;
    uint64_t pair_result_bytes = 0;
    uint64_t graph_candidate_bytes = 0;
    uint64_t graph_edge_bytes = 0;
    uint64_t gather_slot_bytes = 0;
    uint64_t gather_row_bytes = 0;
    if (!checked_mul(bank->rows, bank->slot_capacity, &resident) ||
        !checked_mul(resident, bank->kernels->storage_bytes, &resident) ||
        !checked_mul(request->program_operand_count, sizeof(uint32_t), &program_operand_bytes) ||
        !checked_mul(request->program_mean_count, sizeof(uint64_t), &program_mean_bytes) ||
        !checked_add(program_operand_bytes, program_mean_bytes, &program_descriptor_bytes) ||
        !checked_mul(request->program_operand_count == 0 ? 0 : 1,
            sizeof(uint32_t), &program_nonfinite_flag_bytes) ||
        !checked_mul(request->pair_count, 2 * sizeof(uint32_t), &pair_slot_bytes) ||
        !checked_mul(request->pair_count,
            bank->kernels->result_bytes + sizeof(uint32_t) + sizeof(uint64_t), &pair_result_bytes) ||
        !checked_mul(request->graph_candidate_count,
            sizeof(uint32_t) + bank->kernels->result_bytes + sizeof(uint32_t) + sizeof(uint64_t),
            &graph_candidate_bytes) ||
        !checked_mul(request->graph_edge_count,
            sizeof(GafimeSemanticEdge) + bank->kernels->storage_bytes, &graph_edge_bytes) ||
        !checked_mul(request->gather_slot_count, 2 * sizeof(uint32_t), &gather_slot_bytes) ||
        !checked_mul(request->gather_row_count, sizeof(uint64_t), &gather_row_bytes) ||
        !checked_mul(bank->rows, request->retained_slot_count, &retained) ||
        !checked_mul(retained, bank->kernels->storage_bytes, &retained) ||
        !checked_add(pair_slot_bytes, pair_result_bytes, &pair_bytes) ||
        !checked_add(graph_candidate_bytes, graph_edge_bytes, &graph_bytes) ||
        !checked_add(gather_slot_bytes, gather_row_bytes, &gather_descriptor_bytes) ||
        !checked_add(program_descriptor_bytes, program_nonfinite_flag_bytes,
            &program_descriptor_bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // Native launchers return before device allocation for no-op pair, graph,
    // or gather work.  Preserve that exact zero-count lifetime here.
    if (request->pair_count == 0) pair_bytes = 0;
    if (request->graph_candidate_count == 0) graph_bytes = 0;
    if (request->gather_slot_count == 0 || request->gather_row_count == 0) {
        gather_descriptor_bytes = 0;
    }
    const uint64_t transient = std::max(
        std::max(pair_bytes, graph_bytes), std::max(program_descriptor_bytes, gather_descriptor_bytes));
    forecast->resident_bytes = resident;
    forecast->transient_bytes = transient;
    forecast->retained_bytes = retained;
    return GAFIME_STATUS_OK;
}

int cuda_semantic_retain_internal(
    SemanticCudaBank* source,
    GafimeSliceU32 slots,
    GafimeGpuSemanticBank* retained_out
) {
    if (retained_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *retained_out = nullptr;
    int status = require_semantic_bank(source);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_slot_slice(slots, source->slot_capacity);
    if (status != GAFIME_STATUS_OK || slots.len == 0 ||
        slots.len > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        !semantic_slots_initialized(source, slots.ptr, slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t elements = 0;
    uint64_t bytes = 0;
    if (!checked_mul(source->rows, slots.len, &elements) ||
        !checked_mul(elements, source->kernels->storage_bytes, &bytes)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    GafimeSemanticBankDesc desc{};
    desc.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.route = source->route;
    desc.layout = GAFIME_MATRIX_COLUMN_MAJOR;
    desc.rows = source->rows;
    desc.source_slots = static_cast<uint32_t>(slots.len);
    desc.slot_capacity = static_cast<uint32_t>(slots.len);
    desc.bytes = bytes;
    status = cuda_semantic_bank_alloc_internal(source->device_id, &desc, retained_out);
    if (status != GAFIME_STATUS_OK) return status;
    auto* retained = semantic_bank_from_handle(*retained_out);
    ScopedCudaDevice device(source->device_id);
    status = cuda_status(device.status());
    if (status == GAFIME_STATUS_OK) {
        for (uint64_t index = 0; index < slots.len; ++index) {
            status = cuda_status(cudaMemcpy(
                static_cast<uint8_t*>(retained->columns) +
                    static_cast<size_t>(index * source->rows * source->kernels->storage_bytes),
                static_cast<const uint8_t*>(source->columns) +
                    static_cast<size_t>(static_cast<uint64_t>(slots.ptr[index]) * source->rows *
                        source->kernels->storage_bytes),
                static_cast<size_t>(source->rows * source->kernels->storage_bytes),
                cudaMemcpyDeviceToDevice));
            if (status != GAFIME_STATUS_OK) break;
        }
    }
    if (status != GAFIME_STATUS_OK) {
        const int cleanup_status = free_semantic_cuda_bank(retained);
        // If cleanup itself fails, preserve the owned handle for the caller
        // instead of losing an allocation behind a null error output.
        if (cleanup_status != GAFIME_STATUS_OK) return cleanup_status;
        *retained_out = nullptr;
        return status;
    }
    std::fill(retained->initialized_slots.begin(), retained->initialized_slots.end(), uint8_t{1});
    retained->sources_uploaded = true;
    return GAFIME_STATUS_OK;
}

int cuda_semantic_download_internal(
    SemanticCudaBank* bank,
    GafimeSliceU32 slots,
    const GafimeNumericRoute* route,
    GafimeMutableBufferView* columns_out
) {
    int status = require_semantic_bank(bank);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_semantic_abi::validate_download(
        slots, bank->slot_capacity, bank->rows, route, bank->profile, columns_out);
    if (status != GAFIME_STATUS_OK || !gafime_gpu_abi::route_fields_equal(*route, bank->route) ||
        !semantic_slots_initialized(bank, slots.ptr, slots.len)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    ScopedCudaDevice device(bank->device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    for (uint64_t index = 0; index < slots.len; ++index) {
        status = cuda_status(cudaMemcpy(
            static_cast<uint8_t*>(columns_out->data) +
                static_cast<size_t>(index * bank->rows * bank->kernels->storage_bytes),
            static_cast<const uint8_t*>(bank->columns) +
                static_cast<size_t>(static_cast<uint64_t>(slots.ptr[index]) * bank->rows *
                    bank->kernels->storage_bytes),
            static_cast<size_t>(bank->rows * bank->kernels->storage_bytes), cudaMemcpyDeviceToHost));
        if (status != GAFIME_STATUS_OK) return status;
    }
    return GAFIME_STATUS_OK;
}

}  // namespace

namespace gafime_cuda_v1::detail {

bool free_precision_cuda_matrix(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic) return false;
    free_precision_matrix(matrix);
    return true;
}

bool interaction_diagnostics_precision_cuda_matrix(
    GafimeGpuMatrix matrix_handle,
    GafimeInteractionDiagnosticBatch* diagnostics,
    int* status_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic) return false;
    if (status_out == nullptr) return false;
    if (!matrix->legacy_abi10) {
        *status_out = GAFIME_STATUS_INVALID_ARGUMENT;
        return true;
    }
    *status_out = precision_interaction_diagnostics(matrix, diagnostics);
    return true;
}

int inspect_cuda_matrix(
    GafimeGpuMatrix matrix_handle,
    CudaMatrixView* view_out
) {
    if (view_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    const int status = require_precision_matrix(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    // The local RT bridge consumes the historical f32 feature/target view.
    // Mixed has the same f32 storage, while fp64 is deliberately rejected by
    // the RT-only path rather than silently narrowing resident data.
    if (matrix->kernels == nullptr || matrix->kernels->storage_bytes != sizeof(float)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    *view_out = {
        static_cast<float*>(matrix->features),
        static_cast<float*>(matrix->target),
        matrix->rows,
        matrix->cols,
        matrix->device_id,
        matrix->architecture_class,
        matrix->device_flags,
        matrix->features_are_finite,
        matrix->feature_generation,
        matrix->target_generation,
    };
    return GAFIME_STATUS_OK;
}

}  // namespace gafime_cuda_v1::detail

extern "C" {

static int cuda_matrix_alloc_internal(
    uint32_t device_id, const GafimePrecisionMatrixDesc* desc, GafimeGpuMatrix* matrix_out,
    bool legacy_abi10 = false
) try {
    if (matrix_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *matrix_out = nullptr;
    int status = validate_precision_matrix_desc(desc);
    if (status != GAFIME_STATUS_OK) return status;
    ScopedCudaDevice device(device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    cudaDeviceProp props{};
    status = cuda_status(cudaGetDeviceProperties(&props, static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) return status;
    const auto launch_policy = gafime_cuda_v1::cuda_kernel_launch_policy_for_device(
        static_cast<uint32_t>(props.major), static_cast<uint32_t>(props.maxThreadsPerBlock));
    if (!gafime_cuda_v1::cuda_kernel_launch_policy_supported(launch_policy)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    const auto profile = static_cast<GafimePrecisionProfile>(desc->profile);
    const auto* kernels = legacy_abi10
        ? gafime_cuda_v1::cuda_legacy_kernel_set()
        : gafime_cuda_v1::cuda_precision_kernel_set(profile);
    const auto* host_ops = precision_host_ops(profile);
    if (kernels == nullptr || host_ops == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    auto* matrix = new PrecisionCudaMatrix{};
    matrix->magic = kPrecisionCudaMatrixMagic;
    matrix->profile = profile;
    matrix->kernels = kernels;
    matrix->host_ops = host_ops;
    matrix->device_id = device_id;
    matrix->device_flags = precision_cuda_device_flags(props, device_id);
    matrix->architecture_class = cuda_architecture_class(props);
    matrix->launch_policy = launch_policy;
    matrix->rows = desc->rows;
    matrix->cols = desc->cols;
    matrix->legacy_abi10 = legacy_abi10;
    matrix->features_are_finite = true;
    matrix->target_is_finite = true;
    matrix->target_abs_exponent = gafime_gpu_abi::kZeroMagnitudeExponent;
    const uint64_t features = desc->rows * static_cast<uint64_t>(desc->cols);
    status = cuda_status(cudaMalloc(&matrix->features, static_cast<size_t>(features) * kernels->storage_bytes));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMalloc(
        &matrix->target, static_cast<size_t>(desc->rows) * kernels->storage_bytes));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMalloc(
        &matrix->column_means, static_cast<size_t>(desc->cols) * kernels->storage_bytes));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMalloc(&matrix->target_stats, kernels->target_stats_bytes));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaMalloc(
        &matrix->feature_stats, static_cast<size_t>(desc->cols) * kernels->feature_stats_bytes));
    if (status != GAFIME_STATUS_OK) {
        free_precision_matrix(matrix);
        return status;
    }
    if (desc->rows <= gafime_cuda_v1::kSpearmanTargetRankCacheMaxSamples) {
        if (cudaMalloc(&matrix->spearman_target_ranks_twice,
                static_cast<size_t>(desc->rows) * sizeof(uint64_t)) != cudaSuccess) {
            matrix->spearman_target_ranks_twice = nullptr;
        }
    }
    *matrix_out = static_cast<GafimeGpuMatrix>(matrix);
    return GAFIME_STATUS_OK;
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

static int cuda_matrix_upload_f32_internal(
    GafimeGpuMatrix matrix_handle, const float* features, const float* target,
    uint64_t rows, uint32_t cols
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->upload_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->upload_f32(matrix, features, target, rows, cols);
}

static int cuda_matrix_upload_f64_internal(
    GafimeGpuMatrix matrix_handle, const double* features, const double* target,
    uint64_t rows, uint32_t cols
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->upload_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->upload_f64(matrix, features, target, rows, cols);
}

static int cuda_matrix_update_target_f32_internal(
    GafimeGpuMatrix matrix_handle, const float* target, uint64_t rows
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->update_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->update_f32(matrix, target, rows);
}

static int cuda_matrix_update_target_f64_internal(
    GafimeGpuMatrix matrix_handle, const double* target, uint64_t rows
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->update_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->update_f64(matrix, target, rows);
}

static int cuda_execute_f32_internal(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->execute_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->execute_f32(matrix, protocol, result_out);
}

static int cuda_execute_f64_internal(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTableF64* result_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->execute_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->execute_f64(matrix, protocol, result_out);
}

static int cuda_execution_memory_peak_internal(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    if (peak_bytes_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *peak_bytes_out = 0;
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    int status = validate_precision_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    status = require_precision_matrix(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    *peak_bytes_out = precision_execution_peak(matrix, protocol);
    return GAFIME_STATUS_OK;
}

static int cuda_permutation_memory_peak_internal(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    uint64_t selected_row_count, uint64_t* peak_bytes_out
) {
    if (peak_bytes_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *peak_bytes_out = 0;
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    int status = validate_precision_protocol(protocol, matrix);
    if (status != GAFIME_STATUS_OK) return status;
    status = require_precision_matrix(matrix);
    if (status != GAFIME_STATUS_OK) return status;
    const uint64_t total_rows = planned_row_count(protocol->base);
    if (protocol->base->permutations.permutation_count == 0 || selected_row_count > total_rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *peak_bytes_out = precision_permutation_peak(matrix, protocol, selected_row_count);
    return GAFIME_STATUS_OK;
}

static int cuda_permutation_pvalues_f32_internal(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->permutation_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->permutation_f32(matrix, protocol, significance_out);
}

static int cuda_permutation_pvalues_f64_internal(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTableF64* significance_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->permutation_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->permutation_f64(matrix, protocol, significance_out);
}

/* -------------------------------------------------------------------------
 * Stable ABI 1.0 adapters.
 *
 * ABI 1.0 keeps its frozen C layouts and status contract, but the resident
 * matrix, allocation, graph, ranking, permutation, diagnostics, and device
 * kernels are all owned by the profile engine above.  The adapter deliberately
 * constructs only the private precision protocol wrapper and never exposes a
 * second matrix or kernel implementation.
 * ------------------------------------------------------------------------- */

GAFIME_GPU_API int gafime_gpu_device_info(
    uint32_t device_id,
    GafimeGpuDeviceInfo* info_out
) {
    if (info_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    ScopedCudaDevice device(device_id);
    if (device.status() != cudaSuccess) return cuda_status(device.status());
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, static_cast<int>(device_id)) != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    std::memset(info_out, 0, sizeof(*info_out));
    info_out->abi_version = GAFIME_ABI_VERSION;
    info_out->backend_kind = GAFIME_BACKEND_CUDA;
    info_out->device_id = device_id;
    info_out->flags = precision_cuda_device_flags(props, device_id);
    std::strncpy(info_out->name, props.name, sizeof(info_out->name) - 1);
    info_out->name[sizeof(info_out->name) - 1] = '\0';
    info_out->total_global_mem_bytes = static_cast<uint64_t>(props.totalGlobalMem);
    info_out->multiprocessor_count = static_cast<uint32_t>(props.multiProcessorCount);
    info_out->warp_size = static_cast<uint32_t>(props.warpSize);
    info_out->compute_major = static_cast<uint32_t>(props.major);
    info_out->compute_minor = static_cast<uint32_t>(props.minor);
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        info_out->driver_version = static_cast<uint32_t>(driver_version);
    }
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        info_out->runtime_version = static_cast<uint32_t>(runtime_version);
    }
    info_out->reserved[0] = cuda_architecture_class(props);
    const int shared_optin = cuda_device_attr(
        device_id, cudaDevAttrMaxSharedMemoryPerBlockOptin);
    info_out->reserved[1] = static_cast<uint64_t>(
        shared_optin > 0 ? shared_optin :
        cuda_device_attr(device_id, cudaDevAttrMaxSharedMemoryPerBlock));
    info_out->reserved[2] = static_cast<uint64_t>(
        cuda_device_attr(device_id, cudaDevAttrMaxRegistersPerBlock));
    info_out->reserved[3] = static_cast<uint64_t>(
        cuda_device_attr(device_id, cudaDevAttrL2CacheSize));
    info_out->reserved[4] = static_cast<uint64_t>(
        cuda_device_attr(device_id, cudaDevAttrGlobalMemoryBusWidth));
    info_out->reserved[5] = static_cast<uint64_t>(
        cuda_device_attr(device_id, cudaDevAttrMemoryClockRate));
    info_out->reserved[6] = static_cast<uint64_t>(
        cuda_device_attr(device_id, cudaDevAttrMaxThreadsPerMultiProcessor));
    info_out->reserved[7] = static_cast<uint64_t>(props.maxThreadsPerBlock);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_graph_capability(
    uint32_t device_id,
    GafimeGpuGraphCapability* capability_out
) {
    (void)device_id;
    int status = gafime_gpu_abi::fill_graph_capability(
        GAFIME_BACKEND_CUDA, GAFIME_GRAPH_STREAM_CAPTURE, capability_out);
    if (status == GAFIME_STATUS_OK) {
        capability_out->supports_kernel_param_update = 0;
        capability_out->supports_device_ranking = 1;
        capability_out->max_captured_nodes = 64;
        capability_out->stable_pointer_flags = 1;
    }
    return status;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc(
    uint32_t device_id,
    const GafimeMatrixDesc* matrix_desc,
    GafimeGpuMatrix* matrix_out
) try {
    if (matrix_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *matrix_out = nullptr;
    int status = validate_legacy_matrix_desc(matrix_desc);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionMatrixDesc internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = GAFIME_PRECISION_FP32;
    internal.dtype = GAFIME_DTYPE_F32;
    internal.layout = matrix_desc->layout;
    internal.rows = matrix_desc->rows;
    internal.cols = matrix_desc->cols;
    internal.row_stride = matrix_desc->row_stride;
    internal.bytes = matrix_desc->bytes;
    return cuda_matrix_alloc_internal(device_id, &internal, matrix_out, true);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_matrix_upload(
    GafimeGpuMatrix matrix_handle,
    const float* features_host,
    const float* target_host,
    uint64_t rows,
    uint32_t cols
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        !matrix->legacy_abi10 || features_host == nullptr || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return cuda_matrix_upload_f32_internal(
        matrix_handle, features_host, target_host, rows, cols);
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target(
    GafimeGpuMatrix matrix_handle,
    const float* target_host,
    uint64_t rows
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        !matrix->legacy_abi10 || target_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return cuda_matrix_update_target_f32_internal(matrix_handle, target_host, rows);
}

GAFIME_GPU_API void gafime_gpu_matrix_free(GafimeGpuMatrix matrix_handle) {
    // The historical free symbol is void and intentionally remains a no-op for
    // null/unknown handles.  The shared owner also accepts ABI 1.1 handles so
    // RT teardown and explicit v2 cleanup cannot double-own device memory.
    static_cast<void>(gafime_cuda_v1::detail::free_precision_cuda_matrix(matrix_handle));
}

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics(
    GafimeGpuMatrix matrix_handle,
    GafimeInteractionDiagnosticBatch* diagnostics
) {
    int status = GAFIME_STATUS_INVALID_ARGUMENT;
    if (gafime_cuda_v1::detail::interaction_diagnostics_precision_cuda_matrix(
            matrix_handle, diagnostics, &status)) {
        return status;
    }
    return GAFIME_STATUS_INVALID_ARGUMENT;
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    GafimePrecisionLaunchProtocol internal{};
    int status = validate_legacy_protocol(protocol, matrix, &internal);
    if (status != GAFIME_STATUS_OK) return status;
    return cuda_execution_memory_peak_internal(matrix_handle, &internal, peak_bytes_out);
}

GAFIME_GPU_API int gafime_gpu_permutation_memory_peak(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    GafimePrecisionLaunchProtocol internal{};
    int status = validate_legacy_protocol(protocol, matrix, &internal);
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t selected_metric_count = 0;
    if (!checked_mul(selected_row_count, protocol->metric_ids.len, &selected_metric_count) ||
        !fits_size_t(selected_metric_count, sizeof(float)) ||
        !fits_size_t(selected_metric_count, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return cuda_permutation_memory_peak_internal(
        matrix_handle, &internal, selected_row_count, peak_bytes_out);
}

GAFIME_GPU_API int gafime_gpu_execute(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    GafimePrecisionLaunchProtocol internal{};
    int status = validate_legacy_protocol(protocol, matrix, &internal);
    if (status != GAFIME_STATUS_OK) return status;
    status = validate_legacy_result_table(protocol, result_out);
    if (status != GAFIME_STATUS_OK) return status;
    return cuda_execute_f32_internal(matrix_handle, &internal, result_out);
}

GAFIME_GPU_API int gafime_gpu_permutation_pvalues(
    GafimeGpuMatrix matrix_handle,
    const GafimeLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    GafimePrecisionLaunchProtocol internal{};
    int status = validate_legacy_protocol(protocol, matrix, &internal);
    if (status != GAFIME_STATUS_OK) return status;
    const uint64_t total_rows = planned_row_count(protocol);
    status = validate_legacy_significance(protocol, matrix, significance_out, total_rows);
    if (status != GAFIME_STATUS_OK) return status;
    return cuda_permutation_pvalues_f32_internal(
        matrix_handle, &internal, significance_out);
}

GAFIME_GPU_API int gafime_gpu_numeric_routes_v2(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    uint32_t route_stride,
    GafimeNumericRoute* routes_out,
    uint32_t route_capacity,
    uint32_t* route_count_out
) {
    ScopedCudaDevice device(device_id);
    if (device.status() != cudaSuccess) return cuda_status(device.status());
    constexpr uint32_t profiles[] = {
        GAFIME_PRECISION_FP32,
        GAFIME_PRECISION_MIXED,
        GAFIME_PRECISION_FP64,
    };
    return gafime_gpu_abi::enumerate_numeric_routes(
        consumer_abi_version,
        route_stride,
        routes_out,
        route_capacity,
        route_count_out,
        profiles,
        static_cast<uint32_t>(sizeof(profiles) / sizeof(profiles[0]))
    );
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc_v2(
    uint32_t device_id,
    const GafimeNumericMatrixDesc* desc,
    GafimeGpuMatrix* matrix_out
) {
    int status = gafime_gpu_abi::validate_numeric_matrix_desc(desc);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionMatrixDesc internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = desc->route.profile;
    internal.dtype = desc->route.storage_dtype;
    internal.layout = desc->layout;
    internal.rows = desc->rows;
    internal.cols = desc->cols;
    internal.row_stride = desc->row_stride;
    internal.bytes = desc->bytes;
    return cuda_matrix_alloc_internal(device_id, &internal, matrix_out);
}

GAFIME_GPU_API int gafime_gpu_matrix_upload_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* features,
    const GafimeConstBufferView* target,
    uint64_t rows,
    uint32_t cols
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->profile != matrix->profile || rows != matrix->rows || cols != matrix->cols) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    uint64_t feature_count = 0;
    if (!checked_mul(rows, cols, &feature_count)) return GAFIME_STATUS_INVALID_ARGUMENT;
    status = gafime_gpu_abi::validate_const_buffer(
        features, route->storage_dtype, feature_count);
    if (status == GAFIME_STATUS_OK) {
        status = gafime_gpu_abi::validate_const_buffer(target, route->storage_dtype, rows);
    }
    if (status != GAFIME_STATUS_OK) return status;
    if (route->storage_dtype == GAFIME_DTYPE_F32) {
        return cuda_matrix_upload_f32_internal(
            matrix_handle,
            static_cast<const float*>(features->data),
            static_cast<const float*>(target->data),
            rows,
            cols);
    }
    if (route->storage_dtype == GAFIME_DTYPE_F64) {
        return cuda_matrix_upload_f64_internal(
            matrix_handle,
            static_cast<const double*>(features->data),
            static_cast<const double*>(target->data),
            rows,
            cols);
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* target,
    uint64_t rows
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_route(route);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->profile != matrix->profile || rows != matrix->rows) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    status = gafime_gpu_abi::validate_const_buffer(target, route->storage_dtype, rows);
    if (status != GAFIME_STATUS_OK) return status;
    if (route->storage_dtype == GAFIME_DTYPE_F32) {
        return cuda_matrix_update_target_f32_internal(
            matrix_handle, static_cast<const float*>(target->data), rows);
    }
    if (route->storage_dtype == GAFIME_DTYPE_F64) {
        return cuda_matrix_update_target_f64_internal(
            matrix_handle, static_cast<const double*>(target->data), rows);
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_execute_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    GafimeNumericResultTable* result_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_gpu_abi::validate_numeric_result_table(
        result_out, protocol->route.result_dtype);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionLaunchProtocol internal_protocol{};
    internal_protocol.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal_protocol.profile = protocol->route.profile;
    internal_protocol.base = protocol->base;
    if (protocol->route.result_dtype == GAFIME_DTYPE_F32) {
        GafimeResultTable internal{};
        internal.abi_version = GAFIME_ABI_VERSION;
        internal.max_arity = result_out->max_arity;
        internal.metric_count = result_out->metric_count;
        internal.flags = result_out->flags;
        internal.capacity = result_out->capacity;
        internal.row_count = result_out->row_count;
        internal.combo_indices = result_out->combo_indices;
        internal.metric_values = static_cast<float*>(result_out->metric_values.data);
        internal.ranks = result_out->ranks;
        internal.families = result_out->families;
        internal.candidate_ids = result_out->candidate_ids;
        internal.row_flags = result_out->row_flags;
        status = cuda_execute_f32_internal(matrix_handle, &internal_protocol, &internal);
        result_out->flags = internal.flags;
        result_out->row_count = internal.row_count;
        return status;
    }
    if (protocol->route.result_dtype == GAFIME_DTYPE_F64) {
        GafimeResultTableF64 internal{};
        internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
        internal.max_arity = result_out->max_arity;
        internal.metric_count = result_out->metric_count;
        internal.flags = result_out->flags;
        internal.capacity = result_out->capacity;
        internal.row_count = result_out->row_count;
        internal.combo_indices = result_out->combo_indices;
        internal.metric_values = static_cast<double*>(result_out->metric_values.data);
        internal.ranks = result_out->ranks;
        internal.families = result_out->families;
        internal.candidate_ids = result_out->candidate_ids;
        internal.row_flags = result_out->row_flags;
        status = cuda_execute_f64_internal(matrix_handle, &internal_protocol, &internal);
        result_out->flags = internal.flags;
        result_out->row_count = internal.row_count;
        return status;
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    uint64_t* peak_bytes_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionLaunchProtocol internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = protocol->route.profile;
    internal.base = protocol->base;
    return cuda_execution_memory_peak_internal(matrix_handle, &internal, peak_bytes_out);
}

GAFIME_GPU_API int gafime_gpu_permutation_memory_peak_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    uint64_t selected_row_count,
    uint64_t* peak_bytes_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionLaunchProtocol internal{};
    internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal.profile = protocol->route.profile;
    internal.base = protocol->base;
    return cuda_permutation_memory_peak_internal(
        matrix_handle, &internal, selected_row_count, peak_bytes_out);
}

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_v2(
    GafimeGpuMatrix matrix_handle,
    const GafimeNumericLaunchProtocol* protocol,
    GafimeNumericSignificanceTable* significance_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_launch_protocol(protocol, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    status = gafime_gpu_abi::validate_numeric_significance_table(
        significance_out, protocol->route.result_dtype);
    if (status != GAFIME_STATUS_OK) return status;
    GafimePrecisionLaunchProtocol internal_protocol{};
    internal_protocol.abi_version = GAFIME_PRECISION_ABI_VERSION;
    internal_protocol.profile = protocol->route.profile;
    internal_protocol.base = protocol->base;
    if (protocol->route.result_dtype == GAFIME_DTYPE_F32) {
        GafimePermutationSignificanceTable internal{};
        internal.abi_version = GAFIME_ABI_VERSION;
        internal.metric_count = significance_out->metric_count;
        internal.row_count = significance_out->row_count;
        internal.candidate_ids = significance_out->candidate_ids;
        internal.observed_metric_values = static_cast<const float*>(
            significance_out->observed_metric_values.data);
        internal.p_values = static_cast<float*>(significance_out->p_values.data);
        return cuda_permutation_pvalues_f32_internal(
            matrix_handle, &internal_protocol, &internal);
    }
    if (protocol->route.result_dtype == GAFIME_DTYPE_F64) {
        GafimePermutationSignificanceTableF64 internal{};
        internal.abi_version = GAFIME_PRECISION_ABI_VERSION;
        internal.metric_count = significance_out->metric_count;
        internal.row_count = significance_out->row_count;
        internal.candidate_ids = significance_out->candidate_ids;
        internal.observed_metric_values = static_cast<const double*>(
            significance_out->observed_metric_values.data);
        internal.p_values = static_cast<double*>(significance_out->p_values.data);
        return cuda_permutation_pvalues_f64_internal(
            matrix_handle, &internal_protocol, &internal);
    }
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

GAFIME_GPU_API int gafime_gpu_interaction_diagnostics_v2(
    GafimeGpuMatrix matrix_handle,
    GafimeNumericInteractionDiagnosticBatch* diagnostics
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic || matrix->legacy_abi10) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_gpu_abi::validate_numeric_diagnostics(diagnostics, matrix->profile);
    if (status != GAFIME_STATUS_OK) return status;
    GafimeInteractionDiagnosticBatch internal{};
    internal.abi_version = GAFIME_ABI_VERSION;
    internal.max_arity = diagnostics->max_arity;
    internal.row_count = diagnostics->row_count;
    internal.combo_indices = diagnostics->combo_indices;
    internal.combo_index_count = diagnostics->combo_index_count;
    internal.overflow_row_counts = diagnostics->overflow_row_counts;
    internal.flags = diagnostics->row_flags;
    return precision_interaction_diagnostics(matrix, &internal);
}

GAFIME_GPU_API int gafime_gpu_matrix_free_v2(GafimeGpuMatrix matrix_handle) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // Teardown is generation-neutral: after magic validation either ABI
    // generation may release the one canonical resident owner.
    free_precision_matrix(matrix);
    return GAFIME_STATUS_OK;
}

/* -------------------------------------------------------------------------
 * Optional resident semantic-arithmetic table.  This table is deliberately
 * separate from the frozen matrix/target ABI: all operands are generic typed
 * slots, and the Rust semantic owner supplies their context and policy.
 * ------------------------------------------------------------------------- */

GAFIME_GPU_API int gafime_gpu_semantic_capabilities_v1(
    uint32_t device_id,
    uint32_t consumer_abi_version,
    GafimeSemanticCapabilities* capabilities_out
) {
    if (capabilities_out == nullptr ||
        !gafime_gpu_abi::naturally_aligned(capabilities_out)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (GAFIME_ABI_VERSION_MAJOR_OF(consumer_abi_version) !=
            GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MAJOR ||
        GAFIME_ABI_VERSION_MINOR_OF(consumer_abi_version) <
            GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION_MINOR) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    ScopedCudaDevice device(device_id);
    int status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    cudaDeviceProp props{};
    status = cuda_status(cudaGetDeviceProperties(&props, static_cast<int>(device_id)));
    if (status != GAFIME_STATUS_OK) return status;
    const auto policy = gafime_cuda_v1::cuda_kernel_launch_policy_for_device(
        static_cast<uint32_t>(props.major), static_cast<uint32_t>(props.maxThreadsPerBlock));
    if (!gafime_cuda_v1::cuda_kernel_launch_policy_supported(policy)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    std::memset(capabilities_out, 0, sizeof(*capabilities_out));
    capabilities_out->abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    capabilities_out->struct_size = sizeof(*capabilities_out);
    capabilities_out->backend_kind = GAFIME_BACKEND_CUDA;
    capabilities_out->device_id = device_id;
    capabilities_out->profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32 |
        GAFIME_PRECISION_PROFILE_MASK_MIXED | GAFIME_PRECISION_PROFILE_MASK_FP64;
    capabilities_out->program_op_mask = GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOURCE |
        GAFIME_SEMANTIC_PROGRAM_OP_MASK_ABSOLUTE_DIFFERENCE |
        GAFIME_SEMANTIC_PROGRAM_OP_MASK_SOFTSIGN |
        GAFIME_SEMANTIC_PROGRAM_OP_MASK_CENTERED_PRODUCT;
    capabilities_out->primitive_mask = GAFIME_SEMANTIC_PRIMITIVE_MASK_PAIRWISE_PEARSON |
        GAFIME_SEMANTIC_PRIMITIVE_MASK_ORDERED_EDGE_ENERGY |
        GAFIME_SEMANTIC_PRIMITIVE_MASK_SPARSE_GATHER;
    capabilities_out->association_statistic_mask = GAFIME_SEMANTIC_STATISTIC_MASK_PEARSON;
    capabilities_out->max_program_nodes = gafime_semantic_abi::kSemanticMaxProgramNodes;
    capabilities_out->max_slot_count = std::numeric_limits<uint32_t>::max();
    capabilities_out->max_rows = std::numeric_limits<uint64_t>::max() / sizeof(double);
    capabilities_out->max_gather_rows = gafime_semantic_abi::kSemanticMaxGatherRows;
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_alloc_v1(
    uint32_t device_id,
    const GafimeSemanticBankDesc* desc,
    GafimeGpuSemanticBank* bank_out
) try {
    return cuda_semantic_bank_alloc_internal(device_id, desc, bank_out);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_upload_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeNumericRoute* route,
    const GafimeConstBufferView* source_columns
) {
    return cuda_semantic_bank_upload_internal(
        semantic_bank_from_handle(bank_handle), route, source_columns);
}

GAFIME_GPU_API int gafime_gpu_semantic_materialize_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticProgramBatch* batch
) try {
    return cuda_semantic_materialize_internal(semantic_bank_from_handle(bank_handle), batch);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_pairwise_pearson_v1(
    GafimeGpuSemanticBank left_bank_handle,
    GafimeGpuSemanticBank right_bank_handle,
    const GafimeSemanticPearsonBatch* batch,
    GafimeSemanticScalarResultTable* results_out
) {
    return cuda_semantic_pairwise_pearson_internal(
        semantic_bank_from_handle(left_bank_handle), semantic_bank_from_handle(right_bank_handle),
        batch, results_out);
}

GAFIME_GPU_API int gafime_gpu_semantic_ordered_edge_energy_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticEdgeEnergyBatch* batch,
    GafimeSemanticScalarResultTable* results_out
) {
    return cuda_semantic_edge_energy_internal(
        semantic_bank_from_handle(bank_handle), batch, results_out);
}

GAFIME_GPU_API int gafime_gpu_semantic_sparse_gather_v1(
    GafimeGpuSemanticBank source_bank_handle,
    GafimeGpuSemanticBank destination_bank_handle,
    const GafimeSemanticSparseGatherBatch* batch
) try {
    return cuda_semantic_sparse_gather_internal(
        semantic_bank_from_handle(source_bank_handle), semantic_bank_from_handle(destination_bank_handle),
        batch);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_forecast_v1(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticForecastRequest* request,
    GafimeSemanticMemoryForecast* forecast_out
) {
    return cuda_semantic_forecast_internal(
        semantic_bank_from_handle(bank_handle), request, forecast_out);
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_retain_v1(
    GafimeGpuSemanticBank source_bank_handle,
    GafimeSliceU32 slots,
    GafimeGpuSemanticBank* retained_bank_out
) try {
    return cuda_semantic_retain_internal(
        semantic_bank_from_handle(source_bank_handle), slots, retained_bank_out);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_download_v1(
    GafimeGpuSemanticBank bank_handle,
    GafimeSliceU32 slots,
    const GafimeNumericRoute* route,
    GafimeMutableBufferView* columns_out
) {
    return cuda_semantic_download_internal(
        semantic_bank_from_handle(bank_handle), slots, route, columns_out);
}

GAFIME_GPU_API int gafime_gpu_semantic_bank_free_v1(GafimeGpuSemanticBank bank_handle) {
    auto* bank = semantic_bank_from_handle(bank_handle);
    if (bank == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    return free_semantic_cuda_bank(bank);
}

}  // extern "C"
