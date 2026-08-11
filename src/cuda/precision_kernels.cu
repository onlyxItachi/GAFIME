#include "precision_kernels.cuh"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace gafime_cuda_v1::precision_kernel {

constexpr int kThreads = gafime_cuda_v1::kThreadsPerBlock;
constexpr int kMaxMiBins = static_cast<int>(gafime_cuda_v1::kMaxMutualInfoBins);
constexpr int kCudaWarpSize = 32;
constexpr int kMiWarpsPerBlock =
    (gafime_cuda_v1::kMiThreadsPerBlock + kCudaWarpSize - 1) / kCudaWarpSize;

#define GAFIME_CUDA_PRECISION_INLINE __device__ __forceinline__

template <GafimePrecisionProfile Profile>
struct PrecisionTraits;

template <>
struct PrecisionTraits<GAFIME_PRECISION_FP32> {
    using Storage = float;
    using Accumulation = float;
    using Result = float;
};

template <>
struct PrecisionTraits<GAFIME_PRECISION_MIXED> {
    using Storage = float;
    using Accumulation = double;
    using Result = double;
};

template <>
struct PrecisionTraits<GAFIME_PRECISION_FP64> {
    using Storage = double;
    using Accumulation = double;
    using Result = double;
};

// Compile-time lane gates.  These intentionally make an accidental widening
// of fp32, a mixed-result downcast, or an fp64 storage regression fail the
// CUDA payload build rather than merely changing a tolerance test.
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_FP32>::Storage, float>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_FP32>::Accumulation, float>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_FP32>::Result, float>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_MIXED>::Storage, float>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_MIXED>::Accumulation, double>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_MIXED>::Result, double>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_FP64>::Storage, double>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_FP64>::Accumulation, double>);
static_assert(std::is_same_v<PrecisionTraits<GAFIME_PRECISION_FP64>::Result, double>);

template <typename Accumulation>
struct TargetStatsDevice {
    Accumulation mean_y;
    Accumulation syy;
    uint64_t finite_count;
    uint32_t finite;
    uint32_t reserved;
};

template <typename Accumulation>
struct UnaryFeatureStatsDevice {
    Accumulation mean_x;
    Accumulation sxx;
    uint64_t finite_count;
    uint32_t finite;
    uint32_t reserved;
};

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T device_sqrt(T value);

template <>
GAFIME_CUDA_PRECISION_INLINE float device_sqrt(float value) {
    return sqrtf(value);
}

template <>
GAFIME_CUDA_PRECISION_INLINE double device_sqrt(double value) {
    return sqrt(value);
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T device_log(T value);

template <>
GAFIME_CUDA_PRECISION_INLINE float device_log(float value) {
    return logf(value);
}

template <>
GAFIME_CUDA_PRECISION_INLINE double device_log(double value) {
    return log(value);
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T device_abs(T value);

template <>
GAFIME_CUDA_PRECISION_INLINE float device_abs(float value) {
    return fabsf(value);
}

template <>
GAFIME_CUDA_PRECISION_INLINE double device_abs(double value) {
    return fabs(value);
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE bool device_isfinite(T value) {
    return isfinite(value);
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T warp_reduce_sum(T value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T warp_reduce_min(T value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        const T other = __shfl_down_sync(0xffffffffu, value, offset);
        value = other < value ? other : value;
    }
    return value;
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T warp_reduce_max(T value) {
#pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        const T other = __shfl_down_sync(0xffffffffu, value, offset);
        value = other > value ? other : value;
    }
    return value;
}

template <typename T>
GAFIME_CUDA_PRECISION_INLINE T device_nan() {
    return static_cast<T>(NAN);
}

template <typename Accumulation, typename Result>
GAFIME_CUDA_PRECISION_INLINE Result finalize_correlation(
    Accumulation variance_x,
    Accumulation variance_y,
    Accumulation covariance
) {
    if (!device_isfinite(variance_x) || !device_isfinite(variance_y) ||
        !device_isfinite(covariance)) {
        return device_nan<Result>();
    }
    if (variance_x == static_cast<Accumulation>(0) ||
        variance_y == static_cast<Accumulation>(0)) {
        return static_cast<Result>(0);
    }
    if (variance_x < static_cast<Accumulation>(0) ||
        variance_y < static_cast<Accumulation>(0)) {
        return device_nan<Result>();
    }
    const Accumulation denominator = device_sqrt(variance_x * variance_y);
    if (!device_isfinite(denominator) || denominator <= static_cast<Accumulation>(0)) {
        return device_nan<Result>();
    }
    Accumulation correlation = covariance / denominator;
    if (!device_isfinite(correlation)) {
        return device_nan<Result>();
    }
    if (correlation > static_cast<Accumulation>(1)) {
        correlation = static_cast<Accumulation>(1);
    } else if (correlation < static_cast<Accumulation>(-1)) {
        correlation = static_cast<Accumulation>(-1);
    }
    return static_cast<Result>(correlation);
}

template <typename Result>
GAFIME_CUDA_PRECISION_INLINE Result finalize_r2(Result pearson) {
    if (!device_isfinite(pearson)) {
        return device_nan<Result>();
    }
    Result result = pearson * pearson;
    if (result < static_cast<Result>(0)) {
        result = static_cast<Result>(0);
    }
    if (result > static_cast<Result>(1)) {
        result = static_cast<Result>(1);
    }
    return result;
}

template <typename Storage>
GAFIME_CUDA_PRECISION_INLINE Storage interaction_value(
    const Storage* features,
    const Storage* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo,
    uint32_t arity
) {
    if (arity == 1) {
        return features[static_cast<uint64_t>(combo[0]) * rows + row];
    }
    // The type of `value` deliberately is Storage.  In the mixed profile that
    // preserves true fp32 centering and sequential materialization before its
    // value enters any fp64 statistical reduction.
    Storage value = static_cast<Storage>(1);
    for (uint32_t index = 0; index < arity; ++index) {
        const uint32_t column = combo[index];
        value *= features[static_cast<uint64_t>(column) * rows + row] -
            column_means[column];
    }
    return value;
}

template <uint32_t StaticArity, typename Storage>
GAFIME_CUDA_PRECISION_INLINE Storage kernel_interaction_value(
    const Storage* features,
    const Storage* column_means,
    uint64_t row,
    uint64_t rows,
    const uint32_t* combo,
    uint32_t arity
) {
    if constexpr (StaticArity == 0) {
        return interaction_value(features, column_means, row, rows, combo, arity);
    } else {
        static_assert(StaticArity <= gafime_cuda_v1::kTemplateMaxArity);
        if constexpr (StaticArity == 1) {
            return features[static_cast<uint64_t>(combo[0]) * rows + row];
        } else {
            Storage value = static_cast<Storage>(1);
#pragma unroll
            for (uint32_t index = 0; index < StaticArity; ++index) {
                const uint32_t column = combo[index];
                value *= features[static_cast<uint64_t>(column) * rows + row] -
                    column_means[column];
            }
            return value;
        }
    }
}

template <typename Accumulation>
GAFIME_CUDA_PRECISION_INLINE uint32_t fixed_mi_bin(
    Accumulation value,
    Accumulation minimum,
    Accumulation inverse_span,
    uint32_t bins
) {
    const Accumulation scaled = (value - minimum) * inverse_span;
    if (isnan(scaled) || scaled <= static_cast<Accumulation>(0)) {
        return 0;
    }
    const uint32_t max_bin = bins - 1;
    if (!device_isfinite(scaled) || scaled >= static_cast<Accumulation>(max_bin)) {
        return max_bin;
    }
    return static_cast<uint32_t>(scaled);
}

template <typename Storage, typename Accumulation>
__global__ void target_stats_kernel(
    const Storage* target,
    uint64_t n_samples,
    TargetStatsDevice<Accumulation>* target_stats
) {
    Accumulation local_sum = static_cast<Accumulation>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage value = target[row];
        if (device_isfinite(value)) {
            local_sum += static_cast<Accumulation>(value);
            ++local_count;
        }
    }

    __shared__ Accumulation sums[kThreads];
    __shared__ uint64_t counts[kThreads];
    __shared__ Accumulation mean;
    __shared__ Accumulation squares[kThreads];
    sums[threadIdx.x] = local_sum;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean = counts[0] == 0 ? static_cast<Accumulation>(0) :
            sums[0] / static_cast<Accumulation>(counts[0]);
    }
    __syncthreads();

    Accumulation local_squares = static_cast<Accumulation>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage value = target[row];
        if (device_isfinite(value)) {
            const Accumulation centered = static_cast<Accumulation>(value) - mean;
            local_squares += centered * centered;
        }
    }
    squares[threadIdx.x] = local_squares;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            squares[threadIdx.x] += squares[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        target_stats->mean_y = mean;
        target_stats->syy = squares[0];
        target_stats->finite_count = counts[0];
        target_stats->finite = counts[0] == n_samples ? 1u : 0u;
        target_stats->reserved = 0u;
    }
}

template <typename Storage, typename Accumulation>
__global__ void unary_feature_stats_kernel(
    const Storage* features,
    uint64_t n_samples,
    uint32_t n_features,
    UnaryFeatureStatsDevice<Accumulation>* feature_stats
) {
    const uint32_t column = blockIdx.x;
    if (column >= n_features) {
        return;
    }
    const uint64_t base = static_cast<uint64_t>(column) * n_samples;
    Accumulation local_sum = static_cast<Accumulation>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage value = features[base + row];
        if (device_isfinite(value)) {
            local_sum += static_cast<Accumulation>(value);
            ++local_count;
        }
    }
    __shared__ Accumulation sums[kThreads];
    __shared__ uint64_t counts[kThreads];
    __shared__ Accumulation mean;
    __shared__ Accumulation squares[kThreads];
    sums[threadIdx.x] = local_sum;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean = counts[0] == 0 ? static_cast<Accumulation>(0) :
            sums[0] / static_cast<Accumulation>(counts[0]);
    }
    __syncthreads();
    Accumulation local_squares = static_cast<Accumulation>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage value = features[base + row];
        if (device_isfinite(value)) {
            const Accumulation centered = static_cast<Accumulation>(value) - mean;
            local_squares += centered * centered;
        }
    }
    squares[threadIdx.x] = local_squares;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            squares[threadIdx.x] += squares[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        feature_stats[column].mean_x = mean;
        feature_stats[column].sxx = squares[0];
        feature_stats[column].finite_count = counts[0];
        feature_stats[column].finite = counts[0] == n_samples ? 1u : 0u;
        feature_stats[column].reserved = 0u;
    }
}

// One block diagnoses one caller-selected combo using exactly the storage
// arithmetic selected for the profile.  Mixed therefore materializes the
// centered product in fp32, while fp64 never narrows to fp32.  The target and
// histogram/control outputs remain outside the floating-point policy.
template <typename Storage>
__global__ void interaction_diagnostics_kernel(
    const Storage* features,
    const Storage* target,
    const Storage* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t max_arity,
    uint64_t* overflow_row_counts,
    uint32_t* flags
) {
    const uint64_t combo_row = static_cast<uint64_t>(blockIdx.x);
    const uint32_t* combo = combo_indices + combo_row * max_arity;
    uint32_t arity = 0;
    while (arity < max_arity && combo[arity] != UINT32_MAX) {
        ++arity;
    }

    uint64_t local_overflow_rows = 0;
    uint32_t local_flags = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        bool feature_sources_finite = true;
        if (!device_isfinite(target[row])) {
            local_flags |= GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
        }
        for (uint32_t index = 0; index < arity; ++index) {
            const uint32_t column = combo[index];
            const Storage source = features[static_cast<uint64_t>(column) * n_samples + row];
            const Storage mean = column_means[column];
            if (!device_isfinite(source) || !device_isfinite(mean)) {
                feature_sources_finite = false;
                local_flags |= GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE;
            }
        }
        if (arity > 1 && feature_sources_finite) {
            Storage value = static_cast<Storage>(1);
            bool materialization_nonfinite = false;
            for (uint32_t index = 0; index < arity; ++index) {
                const uint32_t column = combo[index];
                const Storage centered =
                    features[static_cast<uint64_t>(column) * n_samples + row] -
                    column_means[column];
                materialization_nonfinite =
                    materialization_nonfinite || !device_isfinite(centered);
                value *= centered;
                materialization_nonfinite =
                    materialization_nonfinite || !device_isfinite(value);
            }
            if (materialization_nonfinite) {
                ++local_overflow_rows;
            }
        }
    }

    __shared__ uint64_t overflow_partials[kThreads];
    __shared__ uint32_t flag_partials[kThreads];
    overflow_partials[threadIdx.x] = local_overflow_rows;
    flag_partials[threadIdx.x] = local_flags;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            overflow_partials[threadIdx.x] += overflow_partials[threadIdx.x + stride];
            flag_partials[threadIdx.x] |= flag_partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        overflow_row_counts[combo_row] = overflow_partials[0];
        flags[combo_row] = flag_partials[0];
    }
}

template <typename Storage, typename Accumulation, typename Result>
__global__ void continuous_unary_kernel(
    const Storage* features,
    const Storage* target,
    const TargetStatsDevice<Accumulation>* target_stats,
    const UnaryFeatureStatsDevice<Accumulation>* feature_stats,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    Result* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) return;
    const uint32_t column = combo_indices[descriptor_offset + combo_row];
    const uint64_t base = static_cast<uint64_t>(column) * n_samples;
    const Accumulation mean_x = feature_stats[column].mean_x;
    const Accumulation mean_y = target_stats->mean_y;
    const Accumulation feature_sxx = feature_stats[column].sxx;

    __shared__ Accumulation covariance[kThreads];
    Accumulation local_covariance = static_cast<Accumulation>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Accumulation dx = static_cast<Accumulation>(features[base + row]) - mean_x;
        const Accumulation dy = static_cast<Accumulation>(target[row]) - mean_y;
        local_covariance += dx * dy;
    }
    covariance[threadIdx.x] = local_covariance;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            covariance[threadIdx.x] += covariance[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const Result pearson = finalize_correlation<Accumulation, Result>(
            feature_sxx, target_stats->syy, covariance[0]);
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            const uint32_t metric_id = metric_ids[metric_index];
            if (metric_id == GAFIME_METRIC_PEARSON) {
                metric_values[combo_row * metric_count + metric_index] = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                metric_values[combo_row * metric_count + metric_index] = finalize_r2(pearson);
            }
        }
    }
}

// Small inputs use one deterministic row-order reduction.  Select covariance
// scaling at launch time so the serial row loops contain no runtime branch for
// a mode that is already fixed by the descriptor's covariance policy.
template <typename Storage, typename Accumulation, typename Result, bool Scaled>
__global__ void continuous_serial_kernel(
    const Storage* features,
    const Storage* target,
    const Storage* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    Result* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count || threadIdx.x != 0) {
        return;
    }
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * arity;

    Accumulation serial_scale_x = static_cast<Accumulation>(0);
    Accumulation serial_scale_y = static_cast<Accumulation>(0);
    uint64_t serial_count = 0;
    for (uint64_t row = 0; row < n_samples; ++row) {
        const Storage x = interaction_value(
            features, column_means, row, n_samples, combo, arity);
        const Storage y = target[row];
        if (device_isfinite(x) && device_isfinite(y)) {
            if constexpr (Scaled) {
                const Accumulation abs_x = device_abs(static_cast<Accumulation>(x));
                const Accumulation abs_y = device_abs(static_cast<Accumulation>(y));
                if (abs_x > serial_scale_x) {
                    serial_scale_x = abs_x;
                }
                if (abs_y > serial_scale_y) {
                    serial_scale_y = abs_y;
                }
            }
            ++serial_count;
        }
    }

    Accumulation serial_sum_x = static_cast<Accumulation>(0);
    Accumulation serial_sum_y = static_cast<Accumulation>(0);
    for (uint64_t row = 0; row < n_samples; ++row) {
        const Storage raw_x = interaction_value(
            features, column_means, row, n_samples, combo, arity);
        const Storage raw_y = target[row];
        if (device_isfinite(raw_x) && device_isfinite(raw_y)) {
            Accumulation x = static_cast<Accumulation>(raw_x);
            Accumulation y = static_cast<Accumulation>(raw_y);
            if constexpr (Scaled) {
                x = serial_scale_x > static_cast<Accumulation>(0)
                    ? x / serial_scale_x : static_cast<Accumulation>(0);
                y = serial_scale_y > static_cast<Accumulation>(0)
                    ? y / serial_scale_y : static_cast<Accumulation>(0);
            }
            serial_sum_x += x;
            serial_sum_y += y;
        }
    }

    const Accumulation serial_count_value = static_cast<Accumulation>(serial_count);
    const Accumulation serial_mean_x = serial_count == 0
        ? static_cast<Accumulation>(0) : serial_sum_x / serial_count_value;
    const Accumulation serial_mean_y = serial_count == 0
        ? static_cast<Accumulation>(0) : serial_sum_y / serial_count_value;

    Accumulation serial_sum_xx = static_cast<Accumulation>(0);
    Accumulation serial_sum_yy = static_cast<Accumulation>(0);
    Accumulation serial_sum_xy = static_cast<Accumulation>(0);
    for (uint64_t row = 0; row < n_samples; ++row) {
        const Storage raw_x = interaction_value(
            features, column_means, row, n_samples, combo, arity);
        const Storage raw_y = target[row];
        if (device_isfinite(raw_x) && device_isfinite(raw_y)) {
            Accumulation x = static_cast<Accumulation>(raw_x);
            Accumulation y = static_cast<Accumulation>(raw_y);
            if constexpr (Scaled) {
                x = serial_scale_x > static_cast<Accumulation>(0)
                    ? x / serial_scale_x : static_cast<Accumulation>(0);
                y = serial_scale_y > static_cast<Accumulation>(0)
                    ? y / serial_scale_y : static_cast<Accumulation>(0);
            }
            const Accumulation delta_x = x - serial_mean_x;
            const Accumulation delta_y = y - serial_mean_y;
            serial_sum_xx += delta_x * delta_x;
            serial_sum_yy += delta_y * delta_y;
            serial_sum_xy += delta_x * delta_y;
        }
    }

    const Result pearson = finalize_correlation<Accumulation, Result>(
        serial_sum_xx, serial_sum_yy, serial_sum_xy);
    for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
        const uint32_t metric_id = metric_ids[metric_index];
        if (metric_id == GAFIME_METRIC_PEARSON) {
            metric_values[combo_row * metric_count + metric_index] = pearson;
        } else if (metric_id == GAFIME_METRIC_R2) {
            metric_values[combo_row * metric_count + metric_index] = finalize_r2(pearson);
        }
    }
}

template <
    typename Storage,
    typename Accumulation,
    typename Result,
    bool Scaled,
    uint32_t StaticArity = 0
>
__global__ void continuous_kernel(
    const Storage* features,
    const Storage* target,
    const Storage* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    Result* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t effective_arity = StaticArity == 0 ? arity : StaticArity;
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * effective_arity;

    __shared__ Accumulation sx[kThreads];
    __shared__ Accumulation sy[kThreads];
    __shared__ Accumulation sxx[kThreads];
    __shared__ Accumulation syy[kThreads];
    __shared__ Accumulation sxy[kThreads];
    __shared__ uint64_t counts[kThreads];
    __shared__ Accumulation scale_x;
    __shared__ Accumulation scale_y;
    __shared__ Accumulation mean_x;
    __shared__ Accumulation mean_y;

    Accumulation local_scale_x = static_cast<Accumulation>(0);
    Accumulation local_scale_y = static_cast<Accumulation>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage x = kernel_interaction_value<StaticArity>(
            features, column_means, row, n_samples, combo, arity);
        const Storage y = target[row];
        if (device_isfinite(x) && device_isfinite(y)) {
            if constexpr (Scaled) {
                const Accumulation ax = device_abs(static_cast<Accumulation>(x));
                const Accumulation ay = device_abs(static_cast<Accumulation>(y));
                if (ax > local_scale_x) {
                    local_scale_x = ax;
                }
                if (ay > local_scale_y) {
                    local_scale_y = ay;
                }
            }
            ++local_count;
        }
    }
    sx[threadIdx.x] = local_scale_x;
    sy[threadIdx.x] = local_scale_y;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if constexpr (Scaled) {
                if (sx[threadIdx.x + stride] > sx[threadIdx.x]) {
                    sx[threadIdx.x] = sx[threadIdx.x + stride];
                }
                if (sy[threadIdx.x + stride] > sy[threadIdx.x]) {
                    sy[threadIdx.x] = sy[threadIdx.x + stride];
                }
            }
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scale_x = Scaled ? sx[0] : static_cast<Accumulation>(1);
        scale_y = Scaled ? sy[0] : static_cast<Accumulation>(1);
    }
    __syncthreads();

    Accumulation local_sx = static_cast<Accumulation>(0);
    Accumulation local_sy = static_cast<Accumulation>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage raw_x = kernel_interaction_value<StaticArity>(
            features, column_means, row, n_samples, combo, arity);
        const Storage raw_y = target[row];
        if (device_isfinite(raw_x) && device_isfinite(raw_y)) {
            Accumulation x = static_cast<Accumulation>(raw_x);
            Accumulation y = static_cast<Accumulation>(raw_y);
            if constexpr (Scaled) {
                x = scale_x > static_cast<Accumulation>(0) ? x / scale_x : static_cast<Accumulation>(0);
                y = scale_y > static_cast<Accumulation>(0) ? y / scale_y : static_cast<Accumulation>(0);
            }
            local_sx += x;
            local_sy += y;
        }
    }
    sx[threadIdx.x] = local_sx;
    sy[threadIdx.x] = local_sy;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (counts[0] == 0) {
            mean_x = static_cast<Accumulation>(0);
            mean_y = static_cast<Accumulation>(0);
        } else {
            const Accumulation count = static_cast<Accumulation>(counts[0]);
            mean_x = sx[0] / count;
            mean_y = sy[0] / count;
        }
    }
    __syncthreads();

    Accumulation local_sxx = static_cast<Accumulation>(0);
    Accumulation local_syy = static_cast<Accumulation>(0);
    Accumulation local_sxy = static_cast<Accumulation>(0);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage raw_x = kernel_interaction_value<StaticArity>(
            features, column_means, row, n_samples, combo, arity);
        const Storage raw_y = target[row];
        if (device_isfinite(raw_x) && device_isfinite(raw_y)) {
            Accumulation x = static_cast<Accumulation>(raw_x);
            Accumulation y = static_cast<Accumulation>(raw_y);
            if constexpr (Scaled) {
                x = scale_x > static_cast<Accumulation>(0) ? x / scale_x : static_cast<Accumulation>(0);
                y = scale_y > static_cast<Accumulation>(0) ? y / scale_y : static_cast<Accumulation>(0);
            }
            const Accumulation dx = x - mean_x;
            const Accumulation dy = y - mean_y;
            local_sxx += dx * dx;
            local_syy += dy * dy;
            local_sxy += dx * dy;
        }
    }
    sxx[threadIdx.x] = local_sxx;
    syy[threadIdx.x] = local_syy;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sxx[threadIdx.x] += sxx[threadIdx.x + stride];
            syy[threadIdx.x] += syy[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const Result pearson = finalize_correlation<Accumulation, Result>(
            sxx[0], syy[0], sxy[0]);
        for (uint32_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            const uint32_t metric_id = metric_ids[metric_index];
            if (metric_id == GAFIME_METRIC_PEARSON) {
                metric_values[combo_row * metric_count + metric_index] = pearson;
            } else if (metric_id == GAFIME_METRIC_R2) {
                metric_values[combo_row * metric_count + metric_index] = finalize_r2(pearson);
            }
        }
    }
}

template <
    typename Storage,
    typename Accumulation,
    typename Result,
    uint32_t StaticArity = 0,
    uint32_t StaticBins = 0
>
__global__ void mutual_info_kernel(
    const Storage* features,
    const Storage* target,
    const Storage* column_means,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    uint32_t bins,
    Result* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    constexpr uint32_t kBinsStorage = StaticBins == 0 ? kMaxMiBins : StaticBins;
    const uint32_t effective_bins = StaticBins == 0 ? bins : StaticBins;
    if (effective_bins < 2 || effective_bins > kMaxMiBins) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = static_cast<Result>(0);
        }
        return;
    }
    const uint32_t effective_arity = StaticArity == 0 ? arity : StaticArity;
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * effective_arity;
    // Fixed-bin placement is pointwise arithmetic. Storage is therefore the
    // binning type: float for fp32/mixed and double for fp64. Probability,
    // correction, logarithm, normalization, and the final mixed score remain
    // Accumulation below.
    __shared__ Storage minimum_x;
    __shared__ Storage maximum_x;
    __shared__ Storage minimum_y;
    __shared__ Storage maximum_y;
    __shared__ Storage warp_min_x[kMiWarpsPerBlock];
    __shared__ Storage warp_max_x[kMiWarpsPerBlock];
    __shared__ Storage warp_min_y[kMiWarpsPerBlock];
    __shared__ Storage warp_max_y[kMiWarpsPerBlock];
    __shared__ unsigned int warp_count[kMiWarpsPerBlock];
    __shared__ unsigned int hist_x[kBinsStorage];
    __shared__ unsigned int hist_y[kBinsStorage];
    __shared__ unsigned int joint[kBinsStorage * kBinsStorage];
    __shared__ unsigned int valid_count;
    __shared__ Accumulation warp_mi[kMiWarpsPerBlock];
    __shared__ unsigned int warp_active_x[kMiWarpsPerBlock];
    __shared__ unsigned int warp_active_y[kMiWarpsPerBlock];

    Storage local_min_x = static_cast<Storage>(INFINITY);
    Storage local_max_x = static_cast<Storage>(-INFINITY);
    Storage local_min_y = static_cast<Storage>(INFINITY);
    Storage local_max_y = static_cast<Storage>(-INFINITY);
    unsigned int local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage raw_x = kernel_interaction_value<StaticArity>(
            features, column_means, row, n_samples, combo, arity);
        const Storage raw_y = target[row];
        if (device_isfinite(raw_x) && device_isfinite(raw_y)) {
            if (raw_x < local_min_x) local_min_x = raw_x;
            if (raw_x > local_max_x) local_max_x = raw_x;
            if (raw_y < local_min_y) local_min_y = raw_y;
            if (raw_y > local_max_y) local_max_y = raw_y;
            ++local_count;
        }
    }
    local_min_x = warp_reduce_min(local_min_x);
    local_max_x = warp_reduce_max(local_max_x);
    local_min_y = warp_reduce_min(local_min_y);
    local_max_y = warp_reduce_max(local_max_y);
    local_count = warp_reduce_sum(local_count);
    const uint32_t lane = threadIdx.x & (kCudaWarpSize - 1);
    const uint32_t warp = threadIdx.x / kCudaWarpSize;
    const uint32_t warp_count_total =
        (blockDim.x + kCudaWarpSize - 1) / kCudaWarpSize;
    if (lane == 0) {
        warp_min_x[warp] = local_min_x;
        warp_max_x[warp] = local_max_x;
        warp_min_y[warp] = local_min_y;
        warp_max_y[warp] = local_max_y;
        warp_count[warp] = static_cast<unsigned int>(local_count);
    }
    __syncthreads();

    Storage block_min_x = static_cast<Storage>(INFINITY);
    Storage block_max_x = static_cast<Storage>(-INFINITY);
    Storage block_min_y = static_cast<Storage>(INFINITY);
    Storage block_max_y = static_cast<Storage>(-INFINITY);
    unsigned int block_valid_count = 0;
    if (threadIdx.x < warp_count_total) {
        block_min_x = warp_min_x[threadIdx.x];
        block_max_x = warp_max_x[threadIdx.x];
        block_min_y = warp_min_y[threadIdx.x];
        block_max_y = warp_max_y[threadIdx.x];
        block_valid_count = warp_count[threadIdx.x];
    }
    if (warp == 0) {
        block_min_x = warp_reduce_min(block_min_x);
        block_max_x = warp_reduce_max(block_max_x);
        block_min_y = warp_reduce_min(block_min_y);
        block_max_y = warp_reduce_max(block_max_y);
        block_valid_count = warp_reduce_sum(block_valid_count);
        if (lane == 0) {
            minimum_x = block_min_x;
            maximum_x = block_max_x;
            minimum_y = block_min_y;
            maximum_y = block_max_y;
            valid_count = block_valid_count;
        }
    }
    for (uint32_t index = threadIdx.x; index < effective_bins; index += blockDim.x) {
        hist_x[index] = 0;
        hist_y[index] = 0;
    }
    for (uint32_t index = threadIdx.x; index < effective_bins * effective_bins; index += blockDim.x) {
        joint[index] = 0;
    }
    __syncthreads();
    if (valid_count <= 1 || maximum_x <= minimum_x || maximum_y <= minimum_y) {
        if (threadIdx.x == 0) {
            metric_values[combo_row * metric_count + metric_index] = static_cast<Result>(0);
        }
        return;
    }
    const Storage inverse_x = static_cast<Storage>(effective_bins) / (maximum_x - minimum_x);
    const Storage inverse_y = static_cast<Storage>(effective_bins) / (maximum_y - minimum_y);
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage raw_x = kernel_interaction_value<StaticArity>(
            features, column_means, row, n_samples, combo, arity);
        const Storage raw_y = target[row];
        if (!device_isfinite(raw_x) || !device_isfinite(raw_y)) {
            continue;
        }
        const uint32_t x_bin = fixed_mi_bin(raw_x, minimum_x, inverse_x, effective_bins);
        const uint32_t y_bin = fixed_mi_bin(raw_y, minimum_y, inverse_y, effective_bins);
        atomicAdd(&hist_x[x_bin], 1u);
        atomicAdd(&hist_y[y_bin], 1u);
        atomicAdd(&joint[x_bin * effective_bins + y_bin], 1u);
    }
    __syncthreads();
    const Accumulation total = static_cast<Accumulation>(valid_count);
    if (effective_bins <= 4) {
        if (threadIdx.x == 0) {
            Accumulation mutual_information = static_cast<Accumulation>(0);
            uint32_t active_x = 0;
            uint32_t active_y = 0;
            for (uint32_t x_bin = 0; x_bin < effective_bins; ++x_bin) {
                if (hist_x[x_bin] == 0) {
                    continue;
                }
                ++active_x;
                for (uint32_t y_bin = 0; y_bin < effective_bins; ++y_bin) {
                    const unsigned int count = joint[x_bin * effective_bins + y_bin];
                    if (count == 0 || hist_y[y_bin] == 0) {
                        continue;
                    }
                    const Accumulation pxy = static_cast<Accumulation>(count) / total;
                    const Accumulation px = static_cast<Accumulation>(hist_x[x_bin]) / total;
                    const Accumulation py = static_cast<Accumulation>(hist_y[y_bin]) / total;
                    mutual_information += pxy * device_log(pxy / (px * py));
                }
            }
            for (uint32_t y_bin = 0; y_bin < effective_bins; ++y_bin) {
                if (hist_y[y_bin] != 0) {
                    ++active_y;
                }
            }
            const Accumulation correction = active_x > 0 && active_y > 0
                ? (static_cast<Accumulation>(active_x - 1) * static_cast<Accumulation>(active_y - 1)) /
                    (static_cast<Accumulation>(2) * total)
                : static_cast<Accumulation>(0);
            const Accumulation corrected = mutual_information > correction
                ? mutual_information - correction
                : static_cast<Accumulation>(0);
            const uint32_t normalizer_bins = active_x < active_y ? active_x : active_y;
            const Accumulation normalizer = normalizer_bins > 1
                ? device_log(static_cast<Accumulation>(normalizer_bins))
                : static_cast<Accumulation>(0);
            metric_values[combo_row * metric_count + metric_index] =
                normalizer > static_cast<Accumulation>(0)
                    ? static_cast<Result>(corrected / normalizer)
                    : static_cast<Result>(0);
        }
    } else {
        Accumulation local_mi = static_cast<Accumulation>(0);
        uint32_t local_active_x = 0;
        uint32_t local_active_y = 0;
        for (uint32_t index = threadIdx.x; index < effective_bins * effective_bins; index += blockDim.x) {
            const uint32_t x_bin = index / effective_bins;
            const uint32_t y_bin = index - x_bin * effective_bins;
            const unsigned int count = joint[index];
            if (count == 0 || hist_x[x_bin] == 0 || hist_y[y_bin] == 0) {
                continue;
            }
            const Accumulation pxy = static_cast<Accumulation>(count) / total;
            const Accumulation px = static_cast<Accumulation>(hist_x[x_bin]) / total;
            const Accumulation py = static_cast<Accumulation>(hist_y[y_bin]) / total;
            local_mi += pxy * device_log(pxy / (px * py));
        }
        for (uint32_t x_bin = threadIdx.x; x_bin < effective_bins; x_bin += blockDim.x) {
            if (hist_x[x_bin] != 0) {
                ++local_active_x;
            }
        }
        for (uint32_t y_bin = threadIdx.x; y_bin < effective_bins; y_bin += blockDim.x) {
            if (hist_y[y_bin] != 0) {
                ++local_active_y;
            }
        }
        local_mi = warp_reduce_sum(local_mi);
        local_active_x = warp_reduce_sum(local_active_x);
        local_active_y = warp_reduce_sum(local_active_y);
        if (lane == 0) {
            warp_mi[warp] = local_mi;
            warp_active_x[warp] = local_active_x;
            warp_active_y[warp] = local_active_y;
        }
        __syncthreads();

        Accumulation block_mi = static_cast<Accumulation>(0);
        uint32_t block_active_x = 0;
        uint32_t block_active_y = 0;
        if (threadIdx.x < warp_count_total) {
            block_mi = warp_mi[threadIdx.x];
            block_active_x = warp_active_x[threadIdx.x];
            block_active_y = warp_active_y[threadIdx.x];
        }
        if (warp == 0) {
            block_mi = warp_reduce_sum(block_mi);
            block_active_x = warp_reduce_sum(block_active_x);
            block_active_y = warp_reduce_sum(block_active_y);
            if (lane == 0) {
                const Accumulation correction = block_active_x > 0 && block_active_y > 0
                    ? (static_cast<Accumulation>(block_active_x - 1) *
                        static_cast<Accumulation>(block_active_y - 1)) /
                        (static_cast<Accumulation>(2) * total)
                    : static_cast<Accumulation>(0);
                const Accumulation corrected = block_mi > correction
                    ? block_mi - correction
                    : static_cast<Accumulation>(0);
                const uint32_t normalizer_bins = block_active_x < block_active_y
                    ? block_active_x : block_active_y;
                const Accumulation normalizer = normalizer_bins > 1
                    ? device_log(static_cast<Accumulation>(normalizer_bins))
                    : static_cast<Accumulation>(0);
                metric_values[combo_row * metric_count + metric_index] =
                    normalizer > static_cast<Accumulation>(0)
                        ? static_cast<Result>(corrected / normalizer)
                        : static_cast<Result>(0);
            }
        }
    }
}

template <typename Storage>
GAFIME_CUDA_PRECISION_INLINE uint64_t rank_twice_for_value(
    Storage value,
    const Storage* values,
    uint64_t n_samples
) {
    uint64_t less = 0;
    uint64_t equal = 0;
    for (uint64_t index = 0; index < n_samples; ++index) {
        const Storage candidate = values[index];
        if (!device_isfinite(candidate)) {
            continue;
        }
        if (candidate < value) {
            ++less;
        } else if (candidate == value) {
            ++equal;
        }
    }
    return equal == 0 ? 0 : less * 2u + equal - 1u;
}

template <typename Storage>
__global__ void build_target_ranks_kernel(
    const Storage* target,
    uint64_t n_samples,
    uint64_t* target_ranks_twice
) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t row = first; row < n_samples; row += stride) {
        const Storage value = target[row];
        target_ranks_twice[row] = device_isfinite(value)
            ? rank_twice_for_value(value, target, n_samples)
            : 0;
    }
}

template <
    typename Storage,
    typename Accumulation,
    typename Result,
    bool CachedTargetRanks,
    uint32_t StaticArity = 0
>
__global__ void spearman_kernel(
    const Storage* features,
    const Storage* target,
    const Storage* column_means,
    const uint64_t* target_ranks_twice,
    const uint32_t* combo_indices,
    uint64_t n_samples,
    uint32_t arity,
    uint64_t descriptor_offset,
    uint64_t combo_count,
    uint32_t metric_count,
    uint32_t metric_index,
    Result* metric_values
) {
    const uint64_t combo_row = blockIdx.x;
    if (combo_row >= combo_count) {
        return;
    }
    const uint32_t effective_arity = StaticArity == 0 ? arity : StaticArity;
    const uint32_t* combo = combo_indices + descriptor_offset + combo_row * effective_arity;
    Accumulation local_sum_x = static_cast<Accumulation>(0);
    Accumulation local_sum_y = static_cast<Accumulation>(0);
    Accumulation local_sum_xx = static_cast<Accumulation>(0);
    Accumulation local_sum_yy = static_cast<Accumulation>(0);
    Accumulation local_sum_xy = static_cast<Accumulation>(0);
    uint64_t local_count = 0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const Storage x = kernel_interaction_value<StaticArity>(
            features, column_means, row, n_samples, combo, arity);
        const Storage y = target[row];
        if (!device_isfinite(x) || !device_isfinite(y)) {
            continue;
        }
        uint64_t x_less = 0;
        uint64_t x_equal = 0;
        for (uint64_t other = 0; other < n_samples; ++other) {
            const Storage other_x = kernel_interaction_value<StaticArity>(
                features, column_means, other, n_samples, combo, arity);
            const Storage other_y = target[other];
            if (!device_isfinite(other_x) || !device_isfinite(other_y)) {
                continue;
            }
            if (other_x < x) {
                ++x_less;
            } else if (other_x == x) {
                ++x_equal;
            }
        }
        const uint64_t rank_x_twice = x_equal == 0 ? 0 : x_less * 2u + x_equal - 1u;
        uint64_t rank_y_twice = 0;
        if constexpr (CachedTargetRanks) {
            rank_y_twice = target_ranks_twice[row];
        } else {
            uint64_t y_less = 0;
            uint64_t y_equal = 0;
            for (uint64_t other = 0; other < n_samples; ++other) {
                const Storage other_x = kernel_interaction_value<StaticArity>(
                    features, column_means, other, n_samples, combo, arity);
                const Storage other_y = target[other];
                if (!device_isfinite(other_x) || !device_isfinite(other_y)) {
                    continue;
                }
                if (other_y < y) {
                    ++y_less;
                } else if (other_y == y) {
                    ++y_equal;
                }
            }
            rank_y_twice = y_equal == 0 ? 0 : y_less * 2u + y_equal - 1u;
        }
        const Accumulation rank_x = static_cast<Accumulation>(rank_x_twice) * static_cast<Accumulation>(0.5);
        const Accumulation rank_y = static_cast<Accumulation>(rank_y_twice) * static_cast<Accumulation>(0.5);
        local_sum_x += rank_x;
        local_sum_y += rank_y;
        local_sum_xx += rank_x * rank_x;
        local_sum_yy += rank_y * rank_y;
        local_sum_xy += rank_x * rank_y;
        ++local_count;
    }
    __shared__ Accumulation sum_x[kThreads];
    __shared__ Accumulation sum_y[kThreads];
    __shared__ Accumulation sum_xx[kThreads];
    __shared__ Accumulation sum_yy[kThreads];
    __shared__ Accumulation sum_xy[kThreads];
    __shared__ uint64_t counts[kThreads];
    sum_x[threadIdx.x] = local_sum_x;
    sum_y[threadIdx.x] = local_sum_y;
    sum_xx[threadIdx.x] = local_sum_xx;
    sum_yy[threadIdx.x] = local_sum_yy;
    sum_xy[threadIdx.x] = local_sum_xy;
    counts[threadIdx.x] = local_count;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_x[threadIdx.x] += sum_x[threadIdx.x + stride];
            sum_y[threadIdx.x] += sum_y[threadIdx.x + stride];
            sum_xx[threadIdx.x] += sum_xx[threadIdx.x + stride];
            sum_yy[threadIdx.x] += sum_yy[threadIdx.x + stride];
            sum_xy[threadIdx.x] += sum_xy[threadIdx.x + stride];
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        Result result = static_cast<Result>(0);
        if (counts[0] > 1) {
            const Accumulation count = static_cast<Accumulation>(counts[0]);
            const Accumulation covariance = count * sum_xy[0] - sum_x[0] * sum_y[0];
            const Accumulation variance_x = count * sum_xx[0] - sum_x[0] * sum_x[0];
            const Accumulation variance_y = count * sum_yy[0] - sum_y[0] * sum_y[0];
            result = finalize_correlation<Accumulation, Result>(variance_x, variance_y, covariance);
        }
        metric_values[combo_row * metric_count + metric_index] = result;
    }
}

template <typename Result, bool Descending>
GAFIME_CUDA_PRECISION_INLINE bool candidate_better(
    Result candidate_score,
    uint32_t candidate_index,
    Result best_score,
    uint32_t best_index
) {
    if (!device_isfinite(candidate_score)) {
        return false;
    }
    if (best_index == UINT32_MAX) {
        return true;
    }
    if constexpr (Descending) {
        if (candidate_score > best_score) return true;
        if (candidate_score < best_score) return false;
    } else {
        if (candidate_score < best_score) return true;
        if (candidate_score > best_score) return false;
    }
    return candidate_index < best_index;
}

template <typename Result, bool Descending>
__global__ void select_topk_partials_kernel(
    const Result* metric_values,
    uint64_t row_count,
    uint32_t metric_count,
    uint32_t primary_metric_index,
    uint32_t top_k,
    Result* partial_scores,
    uint32_t* partial_indices
) {
    __shared__ Result best_scores[kThreads];
    __shared__ uint32_t best_indices[kThreads];
    __shared__ Result previous_score;
    __shared__ uint32_t previous_index;
    const uint64_t block_base = static_cast<uint64_t>(blockIdx.x) * top_k;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        Result local_score = Descending
            ? static_cast<Result>(-INFINITY)
            : static_cast<Result>(INFINITY);
        uint32_t local_index = UINT32_MAX;
        for (uint64_t row = first; row < row_count; row += stride) {
            const uint32_t row_index = static_cast<uint32_t>(row);
            const Result score = metric_values[row * metric_count + primary_metric_index];
            if (rank != 0 && !candidate_better<Result, Descending>(
                    previous_score, previous_index, score, row_index)) {
                continue;
            }
            if (candidate_better<Result, Descending>(score, row_index, local_score, local_index)) {
                local_score = score;
                local_index = row_index;
            }
        }
        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (uint32_t reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
            if (threadIdx.x < reduce_stride && candidate_better<Result, Descending>(
                    best_scores[threadIdx.x + reduce_stride],
                    best_indices[threadIdx.x + reduce_stride],
                    best_scores[threadIdx.x],
                    best_indices[threadIdx.x])) {
                best_scores[threadIdx.x] = best_scores[threadIdx.x + reduce_stride];
                best_indices[threadIdx.x] = best_indices[threadIdx.x + reduce_stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partial_scores[block_base + rank] = best_scores[0];
            partial_indices[block_base + rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
            __threadfence_block();
        }
        __syncthreads();
    }
}

template <typename Result, bool Descending>
__global__ void merge_topk_partials_kernel(
    const Result* partial_scores,
    const uint32_t* partial_indices,
    uint64_t partial_count,
    uint32_t top_k,
    uint32_t* selected_indices
) {
    __shared__ Result best_scores[kThreads];
    __shared__ uint32_t best_indices[kThreads];
    __shared__ Result previous_score;
    __shared__ uint32_t previous_index;
    for (uint32_t rank = 0; rank < top_k; ++rank) {
        Result local_score = Descending
            ? static_cast<Result>(-INFINITY)
            : static_cast<Result>(INFINITY);
        uint32_t local_index = UINT32_MAX;
        for (uint64_t item = threadIdx.x; item < partial_count; item += blockDim.x) {
            const uint32_t candidate_index = partial_indices[item];
            if (candidate_index == UINT32_MAX) continue;
            const Result score = partial_scores[item];
            if (rank != 0 && !candidate_better<Result, Descending>(
                    previous_score, previous_index, score, candidate_index)) {
                continue;
            }
            if (candidate_better<Result, Descending>(score, candidate_index, local_score, local_index)) {
                local_score = score;
                local_index = candidate_index;
            }
        }
        best_scores[threadIdx.x] = local_score;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (uint32_t reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
            if (threadIdx.x < reduce_stride && candidate_better<Result, Descending>(
                    best_scores[threadIdx.x + reduce_stride],
                    best_indices[threadIdx.x + reduce_stride],
                    best_scores[threadIdx.x],
                    best_indices[threadIdx.x])) {
                best_scores[threadIdx.x] = best_scores[threadIdx.x + reduce_stride];
                best_indices[threadIdx.x] = best_indices[threadIdx.x + reduce_stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected_indices[rank] = best_indices[0];
            previous_score = best_scores[0];
            previous_index = best_indices[0];
            __threadfence_block();
        }
        __syncthreads();
    }
}

template <typename Result>
__global__ void copy_selected_rows_kernel(
    const Result* metric_values,
    const uint32_t* selected_indices,
    uint64_t selected_count,
    uint32_t metric_count,
    Result* selected_metric_values
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = selected_count * metric_count;
    if (index >= total) return;
    const uint64_t selected_row = index / metric_count;
    const uint32_t metric_index = static_cast<uint32_t>(index % metric_count);
    const uint32_t source_row = selected_indices[selected_row];
    selected_metric_values[index] = metric_values[
        static_cast<uint64_t>(source_row) * metric_count + metric_index];
}

template <typename Result>
GAFIME_CUDA_PRECISION_INLINE Result metric_extremeness(uint32_t metric_id, Result value) {
    if (!device_isfinite(value)) return static_cast<Result>(-INFINITY);
    if (metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_SPEARMAN) {
        return device_abs(value);
    }
    return value;
}

template <typename Result>
__global__ void selected_metric_max_kernel(
    const Result* metric_values,
    uint64_t row_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    Result* metric_max
) {
    const uint32_t metric_index = blockIdx.x;
    if (metric_index >= metric_count) return;
    Result local_max = static_cast<Result>(-INFINITY);
    const uint32_t metric_id = metric_ids[metric_index];
    for (uint64_t row = threadIdx.x; row < row_count; row += blockDim.x) {
        const Result value = metric_extremeness(metric_id, metric_values[row * metric_count + metric_index]);
        if (value > local_max) local_max = value;
    }
    __shared__ Result partial[kThreads];
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && partial[threadIdx.x + stride] > partial[threadIdx.x]) {
            partial[threadIdx.x] = partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) metric_max[metric_index] = partial[0];
}

template <typename Result>
__global__ void accumulate_exceedances_kernel(
    const Result* metric_max,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    const Result* observed_metric_values,
    uint64_t selected_count,
    uint32_t* exceedance_counts
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = selected_count * metric_count;
    if (index >= total) return;
    const uint32_t metric_index = static_cast<uint32_t>(index % metric_count);
    const Result observed = metric_extremeness(metric_ids[metric_index], observed_metric_values[index]);
    if (metric_max[metric_index] >= observed) atomicAdd(&exceedance_counts[index], 1u);
}

template <GafimePrecisionProfile Profile>
using StorageFor = typename PrecisionTraits<Profile>::Storage;
template <GafimePrecisionProfile Profile>
using AccumulationFor = typename PrecisionTraits<Profile>::Accumulation;
template <GafimePrecisionProfile Profile>
using ResultFor = typename PrecisionTraits<Profile>::Result;

template <GafimePrecisionProfile Profile>
cudaError_t launch_target_stats_erased(
    const void* target, uint64_t n_samples, void* target_stats,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    target_stats_kernel<StorageFor<Profile>, AccumulationFor<Profile>><<<
        1, policy.threads_per_block, 0, stream
    >>>(static_cast<const StorageFor<Profile>*>(target), n_samples,
        static_cast<TargetStatsDevice<AccumulationFor<Profile>>*>(target_stats));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_feature_stats_erased(
    const void* features, uint64_t n_samples, uint32_t n_features, void* feature_stats,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    unary_feature_stats_kernel<StorageFor<Profile>, AccumulationFor<Profile>><<<
        n_features, policy.threads_per_block, 0, stream
    >>>(static_cast<const StorageFor<Profile>*>(features), n_samples, n_features,
        static_cast<UnaryFeatureStatsDevice<AccumulationFor<Profile>>*>(feature_stats));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_interaction_diagnostics_erased(
    const void* features, const void* target, const void* means,
    const uint32_t* combos, uint64_t combo_count, uint64_t n_samples,
    uint32_t max_arity, uint64_t* overflow_row_counts, uint32_t* flags,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    if (combo_count == 0) return cudaSuccess;
    interaction_diagnostics_kernel<StorageFor<Profile>><<<
        combo_count, policy.threads_per_block, 0, stream
    >>>(static_cast<const StorageFor<Profile>*>(features),
        static_cast<const StorageFor<Profile>*>(target),
        static_cast<const StorageFor<Profile>*>(means), combos, n_samples,
        max_arity, overflow_row_counts, flags);
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_continuous_unary_erased(
    const void* features, const void* target, const void* target_stats,
    const void* feature_stats, const uint32_t* combos, uint64_t n_samples,
    uint64_t descriptor_offset, uint64_t combo_count, const uint32_t* metric_ids,
    uint32_t metric_count, void* metric_values, const CudaKernelLaunchPolicy& policy,
    cudaStream_t stream
) {
    continuous_unary_kernel<StorageFor<Profile>, AccumulationFor<Profile>, ResultFor<Profile>><<<
        combo_count, policy.threads_per_block, 0, stream
    >>>(
        static_cast<const StorageFor<Profile>*>(features),
        static_cast<const StorageFor<Profile>*>(target),
        static_cast<const TargetStatsDevice<AccumulationFor<Profile>>*>(target_stats),
        static_cast<const UnaryFeatureStatsDevice<AccumulationFor<Profile>>*>(feature_stats),
        combos, n_samples, descriptor_offset, combo_count, metric_ids, metric_count,
        static_cast<ResultFor<Profile>*>(metric_values));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_continuous_erased(
    const void* features, const void* target, const void* means, const uint32_t* combos,
    uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset, uint64_t combo_count,
    uint32_t scaled, const uint32_t* metric_ids, uint32_t metric_count, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    using Storage = StorageFor<Profile>;
    using Accumulation = AccumulationFor<Profile>;
    using Result = ResultFor<Profile>;
    // Arity changes only the bounded interaction-value loop.  Cloning the
    // complete covariance engine for each arity duplicates reductions and
    // finalization without changing their arithmetic.  The finite unary path
    // remains separately specialized by continuous_unary_kernel above.
    if (n_samples <= static_cast<uint64_t>(kThreads)) {
        if (scaled != 0u) {
            continuous_serial_kernel<Storage, Accumulation, Result, true><<<
                combo_count, policy.threads_per_block, 0, stream
            >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
                static_cast<const Storage*>(means), combos, n_samples, arity, descriptor_offset,
                combo_count, metric_ids, metric_count, static_cast<Result*>(metric_values));
        } else {
            continuous_serial_kernel<Storage, Accumulation, Result, false><<<
                combo_count, policy.threads_per_block, 0, stream
            >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
                static_cast<const Storage*>(means), combos, n_samples, arity, descriptor_offset,
                combo_count, metric_ids, metric_count, static_cast<Result*>(metric_values));
        }
    } else if (scaled != 0u) {
        continuous_kernel<Storage, Accumulation, Result, true><<<
            combo_count, policy.threads_per_block, 0, stream
        >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
            static_cast<const Storage*>(means), combos, n_samples, arity, descriptor_offset,
            combo_count, metric_ids, metric_count, static_cast<Result*>(metric_values));
    } else {
        continuous_kernel<Storage, Accumulation, Result, false><<<
            combo_count, policy.threads_per_block, 0, stream
        >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
            static_cast<const Storage*>(means), combos, n_samples, arity, descriptor_offset,
            combo_count, metric_ids, metric_count, static_cast<Result*>(metric_values));
    }
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile, uint32_t StaticArity, uint32_t StaticBins>
cudaError_t launch_mutual_info_static(
    const void* features, const void* target, const void* means, const uint32_t* combos,
    uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset, uint64_t combo_count,
    uint32_t metric_count, uint32_t metric_index, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    using Storage = StorageFor<Profile>;
    using Accumulation = AccumulationFor<Profile>;
    using Result = ResultFor<Profile>;
    mutual_info_kernel<Storage, Accumulation, Result, StaticArity, StaticBins><<<
        combo_count, policy.threads_per_block, 0, stream
    >>>(
        static_cast<const Storage*>(features), static_cast<const Storage*>(target),
        static_cast<const Storage*>(means), combos, n_samples, arity, descriptor_offset,
        combo_count, metric_count, metric_index, StaticBins,
        static_cast<Result*>(metric_values));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile, uint32_t StaticBins>
cudaError_t launch_mutual_info_static_bins(
    const void* features, const void* target, const void* means, const uint32_t* combos,
    uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset, uint64_t combo_count,
    uint32_t metric_count, uint32_t metric_index, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    // Bin count determines the large shared histogram layout and score-loop
    // shape, so it remains compile-time specialized. Cloning that complete MI
    // engine again for every arity produced 120 redundant kernels: arity is a
    // bounded control value used only by the pointwise interaction helper and
    // does not change histogram/reduction arithmetic. StaticArity=0 shares the
    // same bin-specialized body across arities 1-5 and also preserves frozen
    // ABI 1.0's accepted arity>5 fallback.
    return launch_mutual_info_static<Profile, 0, StaticBins>(
        features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
        metric_count, metric_index, metric_values, policy, stream);
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_mutual_info_erased(
    const void* features, const void* target, const void* means, const uint32_t* combos,
    uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset, uint64_t combo_count,
    uint32_t metric_count, uint32_t metric_index, uint32_t bins, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    switch (bins) {
        case 2:
            return launch_mutual_info_static_bins<Profile, 2>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 4:
            return launch_mutual_info_static_bins<Profile, 4>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 8:
            return launch_mutual_info_static_bins<Profile, 8>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 12:
            return launch_mutual_info_static_bins<Profile, 12>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 16:
            return launch_mutual_info_static_bins<Profile, 16>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 24:
            return launch_mutual_info_static_bins<Profile, 24>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 32:
            return launch_mutual_info_static_bins<Profile, 32>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 48:
            return launch_mutual_info_static_bins<Profile, 48>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 64:
            return launch_mutual_info_static_bins<Profile, 64>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        case 96:
            return launch_mutual_info_static_bins<Profile, 96>(
                features, target, means, combos, n_samples, arity, descriptor_offset, combo_count,
                metric_count, metric_index, metric_values, policy, stream);
        default:
            break;
    }
    mutual_info_kernel<StorageFor<Profile>, AccumulationFor<Profile>, ResultFor<Profile>><<<
        combo_count, policy.threads_per_block, 0, stream
    >>>(static_cast<const StorageFor<Profile>*>(features),
        static_cast<const StorageFor<Profile>*>(target),
        static_cast<const StorageFor<Profile>*>(means), combos, n_samples, arity,
        descriptor_offset, combo_count, metric_count, metric_index, bins,
        static_cast<ResultFor<Profile>*>(metric_values));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_build_target_ranks_erased(
    const void* target, uint64_t n_samples, uint64_t* ranks,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    const uint64_t blocks64 = (n_samples + policy.threads_per_block - 1) / policy.threads_per_block;
    const uint32_t blocks = static_cast<uint32_t>(blocks64 > 4096 ? 4096 : blocks64);
    if (blocks == 0) return cudaSuccess;
    build_target_ranks_kernel<StorageFor<Profile>><<<blocks, policy.threads_per_block, 0, stream>>>(
        static_cast<const StorageFor<Profile>*>(target), n_samples, ranks);
    return cudaGetLastError();
}

template <typename Storage, typename Accumulation, typename Result>
cudaError_t launch_spearman_typed(
    const void* features, const void* target, const void* means, const uint64_t* ranks,
    const uint32_t* combos, uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset,
    uint64_t combo_count, uint32_t metric_count, uint32_t metric_index, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    if (ranks != nullptr && arity == 1) {
        // Cached target ranks are currently a unary fast path.  Preserve that
        // hot specialization while sharing the complete non-cached engine
        // across arities.
        spearman_kernel<Storage, Accumulation, Result, true, 1><<<
            combo_count, policy.threads_per_block, 0, stream
        >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
            static_cast<const Storage*>(means), ranks, combos, n_samples, arity,
            descriptor_offset, combo_count, metric_count, metric_index,
            static_cast<Result*>(metric_values));
    } else if (ranks != nullptr) {
        spearman_kernel<Storage, Accumulation, Result, true, 0><<<
            combo_count, policy.threads_per_block, 0, stream
        >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
            static_cast<const Storage*>(means), ranks, combos, n_samples, arity,
            descriptor_offset, combo_count, metric_count, metric_index,
            static_cast<Result*>(metric_values));
    } else {
        spearman_kernel<Storage, Accumulation, Result, false, 0><<<
            combo_count, policy.threads_per_block, 0, stream
        >>>(static_cast<const Storage*>(features), static_cast<const Storage*>(target),
            static_cast<const Storage*>(means), nullptr, combos, n_samples, arity,
            descriptor_offset, combo_count, metric_count, metric_index,
            static_cast<Result*>(metric_values));
    }
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_spearman_erased(
    const void* features, const void* target, const void* means, const uint64_t* ranks,
    const uint32_t* combos, uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset,
    uint64_t combo_count, uint32_t metric_count, uint32_t metric_index, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    return launch_spearman_typed<StorageFor<Profile>, AccumulationFor<Profile>, ResultFor<Profile>>(
        features, target, means, ranks, combos, n_samples, arity, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, policy, stream);
}

// The historical ABI 1.0 Spearman lane is intentionally not represented as a
// fourth public precision profile.  Its storage and result domains are FP32,
// while rank/covariance accumulation is FP64.  Keep only this primitive as a
// typed adapter over the same generic Spearman kernel family.
cudaError_t launch_legacy_spearman_erased(
    const void* features, const void* target, const void* means, const uint64_t* ranks,
    const uint32_t* combos, uint64_t n_samples, uint32_t arity, uint64_t descriptor_offset,
    uint64_t combo_count, uint32_t metric_count, uint32_t metric_index, void* metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    return launch_spearman_typed<float, double, float>(
        features, target, means, ranks, combos, n_samples, arity, descriptor_offset,
        combo_count, metric_count, metric_index, metric_values, policy, stream);
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_select_topk_erased(
    const void* metric_values, uint64_t row_count, uint32_t metric_count,
    uint32_t primary_metric_index, uint32_t top_k, uint32_t descending,
    uint32_t* selected_indices, void* partial_scores, uint32_t* partial_indices,
    uint32_t partial_blocks, const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    if (top_k == 0 || partial_blocks == 0) return cudaSuccess;
    using Result = ResultFor<Profile>;
    if (descending != 0u) {
        select_topk_partials_kernel<Result, true><<<partial_blocks, policy.threads_per_block, 0, stream>>>(
            static_cast<const Result*>(metric_values), row_count, metric_count, primary_metric_index,
            top_k, static_cast<Result*>(partial_scores), partial_indices);
    } else {
        select_topk_partials_kernel<Result, false><<<partial_blocks, policy.threads_per_block, 0, stream>>>(
            static_cast<const Result*>(metric_values), row_count, metric_count, primary_metric_index,
            top_k, static_cast<Result*>(partial_scores), partial_indices);
    }
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const uint64_t partial_count = static_cast<uint64_t>(partial_blocks) * top_k;
    if (descending != 0u) {
        merge_topk_partials_kernel<Result, true><<<1, policy.threads_per_block, 0, stream>>>(
            static_cast<const Result*>(partial_scores), partial_indices, partial_count, top_k, selected_indices);
    } else {
        merge_topk_partials_kernel<Result, false><<<1, policy.threads_per_block, 0, stream>>>(
            static_cast<const Result*>(partial_scores), partial_indices, partial_count, top_k, selected_indices);
    }
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_copy_selected_rows_erased(
    const void* metric_values, const uint32_t* selected_indices, uint64_t selected_count,
    uint32_t metric_count, void* selected_metric_values,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    const uint64_t total = selected_count * metric_count;
    if (total == 0) return cudaSuccess;
    const uint32_t blocks = static_cast<uint32_t>((total + policy.threads_per_block - 1) /
        policy.threads_per_block);
    copy_selected_rows_kernel<ResultFor<Profile>><<<blocks, policy.threads_per_block, 0, stream>>>(
        static_cast<const ResultFor<Profile>*>(metric_values), selected_indices, selected_count,
        metric_count, static_cast<ResultFor<Profile>*>(selected_metric_values));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_selected_metric_max_erased(
    const void* metric_values, uint64_t row_count, const uint32_t* metric_ids,
    uint32_t metric_count, void* metric_max, const CudaKernelLaunchPolicy& policy,
    cudaStream_t stream
) {
    selected_metric_max_kernel<ResultFor<Profile>><<<metric_count, policy.threads_per_block, 0, stream>>>(
        static_cast<const ResultFor<Profile>*>(metric_values), row_count, metric_ids, metric_count,
        static_cast<ResultFor<Profile>*>(metric_max));
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
cudaError_t launch_accumulate_exceedances_erased(
    const void* metric_max, const uint32_t* metric_ids, uint32_t metric_count,
    const void* observed_metric_values, uint64_t selected_count, uint32_t* exceedance_counts,
    const CudaKernelLaunchPolicy& policy, cudaStream_t stream
) {
    const uint64_t total = selected_count * metric_count;
    if (total == 0) return cudaSuccess;
    const uint32_t blocks = static_cast<uint32_t>((total + policy.threads_per_block - 1) /
        policy.threads_per_block);
    accumulate_exceedances_kernel<ResultFor<Profile>><<<blocks, policy.threads_per_block, 0, stream>>>(
        static_cast<const ResultFor<Profile>*>(metric_max), metric_ids, metric_count,
        static_cast<const ResultFor<Profile>*>(observed_metric_values), selected_count,
        exceedance_counts);
    return cudaGetLastError();
}

template <GafimePrecisionProfile Profile>
CudaPrecisionKernelSet make_kernel_set() {
    using Storage = StorageFor<Profile>;
    using Accumulation = AccumulationFor<Profile>;
    using Result = ResultFor<Profile>;
    CudaPrecisionKernelSet set{};
    set.storage_bytes = sizeof(Storage);
    set.accumulation_bytes = sizeof(Accumulation);
    set.result_bytes = sizeof(Result);
    set.target_stats_bytes = sizeof(TargetStatsDevice<Accumulation>);
    set.feature_stats_bytes = sizeof(UnaryFeatureStatsDevice<Accumulation>);
    set.target_stats = &launch_target_stats_erased<Profile>;
    set.feature_stats = &launch_feature_stats_erased<Profile>;
    set.interaction_diagnostics = &launch_interaction_diagnostics_erased<Profile>;
    set.continuous = &launch_continuous_erased<Profile>;
    set.continuous_unary = &launch_continuous_unary_erased<Profile>;
    set.mutual_info = &launch_mutual_info_erased<Profile>;
    set.build_target_ranks = &launch_build_target_ranks_erased<Profile>;
    set.spearman = &launch_spearman_erased<Profile>;
    set.legacy_spearman = nullptr;
    set.select_topk = &launch_select_topk_erased<Profile>;
    set.copy_selected_rows = &launch_copy_selected_rows_erased<Profile>;
    set.selected_metric_max = &launch_selected_metric_max_erased<Profile>;
    set.accumulate_exceedances = &launch_accumulate_exceedances_erased<Profile>;
    return set;
}

template <GafimePrecisionProfile Profile>
CudaPrecisionKernelSet make_legacy_kernel_set() {
    CudaPrecisionKernelSet set = make_kernel_set<Profile>();
    // ABI 1.0 Pearson/R2/MI are the historical FP32 arithmetic lane and
    // therefore reuse the canonical FP32 static dispatch.  Spearman remains
    // the sole primitive with a distinct adapter arithmetic domain.
    set.legacy_spearman = launch_legacy_spearman_erased;
    return set;
}

}  // namespace gafime_cuda_v1::precision_kernel

namespace gafime_cuda_v1 {

const CudaPrecisionKernelSet* cuda_precision_kernel_set(GafimePrecisionProfile profile) {
    static const CudaPrecisionKernelSet fp32 = precision_kernel::make_kernel_set<GAFIME_PRECISION_FP32>();
    static const CudaPrecisionKernelSet mixed = precision_kernel::make_kernel_set<GAFIME_PRECISION_MIXED>();
    static const CudaPrecisionKernelSet fp64 = precision_kernel::make_kernel_set<GAFIME_PRECISION_FP64>();
    switch (profile) {
    case GAFIME_PRECISION_FP32:
        return &fp32;
    case GAFIME_PRECISION_MIXED:
        return &mixed;
    case GAFIME_PRECISION_FP64:
        return &fp64;
    default:
        return nullptr;
    }
}

const CudaPrecisionKernelSet* cuda_legacy_kernel_set() {
    // ABI 1.0 is the historical FP32 storage/result route.  Only the
    // Spearman function pointer differs; the other metric pointers are the
    // same canonical FP32 static specializations used by ABI 1.1.
    static const CudaPrecisionKernelSet legacy =
        precision_kernel::make_legacy_kernel_set<GAFIME_PRECISION_FP32>();
    return &legacy;
}

}  // namespace gafime_cuda_v1

#undef GAFIME_CUDA_PRECISION_INLINE
