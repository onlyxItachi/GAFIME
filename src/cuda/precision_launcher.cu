#include "precision_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/covariance_policy.hpp"
#include "../common/gpu_abi_impl.hpp"

namespace {

constexpr uint64_t kPrecisionCudaMatrixMagic = 0x4741465052454332ull;  // "GAFPREC2"

struct PrecisionGraphChunkShape {
    uint32_t arity;
    uint32_t family;
    uint32_t scaled_covariance;
    uint64_t descriptor_offset;
    uint64_t combo_count;
};

struct PrecisionCudaMatrix;

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
    const int integrated = cuda_device_attr(device_id, cudaDevAttrIntegrated);
    const int managed = cuda_device_attr(device_id, cudaDevAttrManagedMemory);
    const int unified = cuda_device_attr(device_id, cudaDevAttrUnifiedAddressing);
    if (props.integrated != 0 || integrated != 0) {
        flags |= GAFIME_GPU_DEVICE_FLAG_INTEGRATED | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    } else {
        flags |= GAFIME_GPU_DEVICE_FLAG_DISCRETE;
    }
    if (props.managedMemory != 0 || managed != 0) flags |= GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY;
    if (props.unifiedAddressing != 0 || unified != 0) flags |= GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY;
    // ABI 1.0's F64 flag was reserved for its float-pointer entry points.  The
    // profile capability query is the additive, unambiguous f64 advertisement.
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
        base->max_arity > gafime_cuda_v1::kTemplateMaxArity || base->max_arity > matrix->cols ||
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
    if (has_mutual_info && matrix->rows > std::numeric_limits<uint32_t>::max()) {
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
    invalidate_precision_descriptors(matrix);
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
            const int status = cuda_status(matrix->kernels->continuous(
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
            const int status = cuda_status(matrix->kernels->spearman(
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
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
    if (status != GAFIME_STATUS_OK) return status;
    std::vector<Storage> resident;
    std::vector<Storage> means;
    std::vector<int> exponents;
    bool features_finite = true;
    build_host_resident<Storage, Accumulation>(
        features_host, rows, cols, &resident, &means, &exponents, &features_finite);
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
    int status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
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
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
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
    uint64_t state = protocol->permutations.seed ^
        static_cast<uint64_t>(permutation_index) * 0xD1B54A32D192ED03ull ^
        0xA5A5A5A59E3779B9ull;
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
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
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
            &expected_combo_count) ||
        diagnostics->combo_index_count != expected_combo_count ||
        !fits_size_t(diagnostics->row_count, sizeof(uint64_t)) ||
        !fits_size_t(diagnostics->row_count, sizeof(uint32_t)) ||
        !fits_size_t(diagnostics->combo_index_count, sizeof(uint32_t))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
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
    status = cuda_status(cudaSetDevice(static_cast<int>(matrix->device_id)));
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
    int previous_device = 0;
    const bool restore = cudaGetDevice(&previous_device) == cudaSuccess;
    const cudaError_t selected = cudaSetDevice(static_cast<int>(matrix->device_id));
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
    if (restore && selected == cudaSuccess) static_cast<void>(cudaSetDevice(previous_device));
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
    *status_out = precision_interaction_diagnostics(matrix, diagnostics);
    return true;
}

int inspect_precision_cuda_matrix(
    GafimeGpuMatrix matrix_handle,
    PrecisionCudaMatrixIdentity* identity_out
) {
    if (identity_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *identity_out = {
        static_cast<uint32_t>(matrix->profile),
        matrix->feature_stats_profile,
        matrix->target_stats_profile,
        matrix->descriptor_profile,
        matrix->graph_profile,
        matrix->graph_valid ? 1u : 0u,
        static_cast<uint32_t>(matrix->kernels->storage_bytes),
        static_cast<uint32_t>(matrix->kernels->accumulation_bytes),
        static_cast<uint32_t>(matrix->kernels->result_bytes),
        matrix->feature_generation,
        matrix->target_generation,
        matrix->descriptor_generation,
        matrix->graph_metric_signature,
        reinterpret_cast<uintptr_t>(matrix->features),
        reinterpret_cast<uintptr_t>(matrix->target),
        reinterpret_cast<uintptr_t>(matrix->combo_indices),
        reinterpret_cast<uintptr_t>(matrix->graph_exec),
    };
    return GAFIME_STATUS_OK;
}

}  // namespace gafime_cuda_v1::detail

extern "C" {

GAFIME_GPU_API int gafime_gpu_precision_capabilities(
    uint32_t device_id, GafimePrecisionCapabilities* capabilities_out
) {
    if (capabilities_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (cudaSetDevice(static_cast<int>(device_id)) != cudaSuccess) return GAFIME_STATUS_DEVICE_ERROR;
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, static_cast<int>(device_id)) != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    std::memset(capabilities_out, 0, sizeof(*capabilities_out));
    capabilities_out->abi_version = GAFIME_PRECISION_ABI_VERSION;
    capabilities_out->backend_kind = GAFIME_BACKEND_CUDA;
    capabilities_out->profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32 |
        GAFIME_PRECISION_PROFILE_MASK_MIXED | GAFIME_PRECISION_PROFILE_MASK_FP64;
    capabilities_out->storage_dtype_mask = GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64;
    capabilities_out->result_dtype_mask = GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64;
    capabilities_out->flags = precision_cuda_device_flags(props, device_id);
    capabilities_out->reserved[0] = cuda_architecture_class(props);
    capabilities_out->reserved[1] = static_cast<uint64_t>(props.maxThreadsPerBlock);
    return GAFIME_STATUS_OK;
}

GAFIME_GPU_API int gafime_gpu_matrix_alloc_v2(
    uint32_t device_id, const GafimePrecisionMatrixDesc* desc, GafimeGpuMatrix* matrix_out
) try {
    if (matrix_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *matrix_out = nullptr;
    int status = validate_precision_matrix_desc(desc);
    if (status != GAFIME_STATUS_OK) return status;
    status = cuda_status(cudaSetDevice(static_cast<int>(device_id)));
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
    const auto* kernels = gafime_cuda_v1::cuda_precision_kernel_set(profile);
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

GAFIME_GPU_API int gafime_gpu_matrix_upload_f32_v2(
    GafimeGpuMatrix matrix_handle, const float* features, const float* target,
    uint64_t rows, uint32_t cols
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->upload_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->upload_f32(matrix, features, target, rows, cols);
}

GAFIME_GPU_API int gafime_gpu_matrix_upload_f64_v2(
    GafimeGpuMatrix matrix_handle, const double* features, const double* target,
    uint64_t rows, uint32_t cols
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->upload_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->upload_f64(matrix, features, target, rows, cols);
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target_f32_v2(
    GafimeGpuMatrix matrix_handle, const float* target, uint64_t rows
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->update_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->update_f32(matrix, target, rows);
}

GAFIME_GPU_API int gafime_gpu_matrix_update_target_f64_v2(
    GafimeGpuMatrix matrix_handle, const double* target, uint64_t rows
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->update_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->update_f64(matrix, target, rows);
}

GAFIME_GPU_API int gafime_gpu_execute_f32_v2(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTable* result_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->execute_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->execute_f32(matrix, protocol, result_out);
}

GAFIME_GPU_API int gafime_gpu_execute_f64_v2(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimeResultTableF64* result_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->execute_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->execute_f64(matrix, protocol, result_out);
}

GAFIME_GPU_API int gafime_gpu_execution_memory_peak_v2(
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

GAFIME_GPU_API int gafime_gpu_permutation_memory_peak_v2(
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

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_f32_v2(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTable* significance_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->permutation_f32 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->permutation_f32(matrix, protocol, significance_out);
}

GAFIME_GPU_API int gafime_gpu_permutation_pvalues_f64_v2(
    GafimeGpuMatrix matrix_handle, const GafimePrecisionLaunchProtocol* protocol,
    GafimePermutationSignificanceTableF64* significance_out
) {
    auto* matrix = static_cast<PrecisionCudaMatrix*>(matrix_handle);
    if (matrix == nullptr || matrix->magic != kPrecisionCudaMatrixMagic ||
        matrix->host_ops->permutation_f64 == nullptr) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    return matrix->host_ops->permutation_f64(matrix, protocol, significance_out);
}

}  // extern "C"
