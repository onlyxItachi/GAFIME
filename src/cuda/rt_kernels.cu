#include "rt_kernels.cuh"

#ifdef GAFIME_CUDA_RT_OPTIX_DEVICE

#include <optix.h>
#include <optix_device.h>

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes;
    const float* target;
    const double* target_stats;
    float* membership;
    uint32_t* membership_words;
    uint32_t* direct_inside_counts;
    double* direct_inside_sum_y;
    uint32_t rows;
    uint32_t path_count;
    uint32_t geometry_mode;
    uint32_t words_per_path;
    const uint32_t* group_path_offsets;
    uint32_t group_count;
    uint32_t point_group_stride;
    uint32_t point_stride;
    uint32_t direct_first_hit;
    const GafimeDecisionPathTerm* semantic_exact_terms;
    const uint32_t* semantic_exact_offsets;
    const uint32_t* semantic_region_ids;
    const float* semantic_counterpart_points_xyz;
    const uint32_t* semantic_label_zero_words;
    const uint32_t* semantic_label_one_words;
    GafimeSemanticRtRegionExactStats* semantic_stats;
    uint32_t* semantic_direct_region_ordinals;
    uint32_t semantic_statistic_mask;
    uint32_t semantic_direct_stats;
    uint32_t semantic_view;
};

extern "C" {
__constant__ GafimeRtParams params;
}

static __forceinline__ __device__ bool inside_dim(
    float value,
    float lo,
    float hi,
    bool lo_open
) {
    return lo_open ? (value > lo && value <= hi) : (value >= lo && value <= hi);
}

static __forceinline__ __device__ uint32_t current_group_index()
{
    return params.group_count > 1u ? optixGetInstanceId() : 0u;
}

static __forceinline__ __device__ uint64_t semantic_point_offset(uint32_t row, uint32_t group_idx)
{
    const uint64_t point_stride = static_cast<uint64_t>(params.point_stride);
    return params.group_count > 1u
        ? static_cast<uint64_t>(group_idx) * params.point_group_stride +
            static_cast<uint64_t>(row) * point_stride
        : static_cast<uint64_t>(row) * point_stride;
}

static __forceinline__ __device__ bool inside_box(float3 point, uint32_t path_idx)
{
    const gafime_cuda_v1::rt_kernel::GafimeRtBox box = params.boxes[path_idx];
    bool inside = inside_dim(point.x, box.lo_x, box.hi_x, (box.open_lo_mask & 1u) != 0u);
    if (box.dims > 1u) {
        inside = inside && inside_dim(point.y, box.lo_y, box.hi_y, (box.open_lo_mask & 2u) != 0u);
    }
    if (box.dims > 2u) {
        inside = inside && inside_dim(point.z, box.lo_z, box.hi_z, (box.open_lo_mask & 4u) != 0u);
    }
    return inside;
}

static __forceinline__ __device__ bool inside_semantic_region(float3 point, uint32_t path_idx)
{
    if (params.semantic_exact_terms == nullptr || params.semantic_exact_offsets == nullptr) {
        return inside_box(point, path_idx);
    }
    const uint32_t begin = params.semantic_exact_offsets[path_idx];
    const uint32_t end = params.semantic_exact_offsets[path_idx + 1u];
    for (uint32_t term_index = begin; term_index < end; ++term_index) {
        const GafimeDecisionPathTerm term = params.semantic_exact_terms[term_index];
        if (term.feature >= 3u) return false;
        const float value = term.feature == 0u
            ? point.x
            : term.feature == 1u ? point.y : point.z;
        const bool holds = term.sign == GAFIME_DECISION_PATH_SIGN_LE
            ? value <= term.threshold
            : value > term.threshold;
        if (!holds) return false;
    }
    return true;
}

extern "C" __global__ void __raygen__gafime_dp()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint32_t row = launch_idx.x;
    const uint32_t group_idx = launch_idx.y;
    if (row >= params.rows || (params.group_count > 1u && group_idx >= params.group_count)) {
        return;
    }

    const uint64_t point_offset = semantic_point_offset(row, group_idx);
    const float x = params.points_xyz[point_offset + 0u];
    const float y = params.points_xyz[point_offset + 1u];
    const bool triangle_2d_instanced = params.geometry_mode == 2u;
    uint32_t payload_row = row;
    const float group_z = params.geometry_mode != 0u ? static_cast<float>(group_idx) * 4.0f : 0.0f;
    const float3 origin = triangle_2d_instanced
        ? make_float3(x, y, group_z - 1.0f)
        : make_float3(
            static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(x)),
            static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(y)),
            group_z - 2.0f
        );
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const float tmax = triangle_2d_instanced ? 2.0f : 4.0f;

    const unsigned int ray_flags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
        (params.direct_first_hit != 0u ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT : 0u);

    optixTrace(
        params.handle,
        origin,
        direction,
        0.0f,
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        ray_flags,
        0,
        1,
        0,
        payload_row
    );
}

extern "C" __global__ void __miss__gafime_dp() {}

extern "C" __global__ void __intersection__gafime_dp_box()
{
    const uint32_t row = optixGetPayload_0();
    const uint32_t primitive_idx = optixGetPrimitiveIndex();
    const uint32_t group_idx = current_group_index();
    const uint32_t path_base = params.group_path_offsets != nullptr ? params.group_path_offsets[group_idx] : 0u;
    const uint32_t path_idx = path_base + primitive_idx;
    if (path_idx >= params.path_count) {
        return;
    }
    const uint64_t point_offset = semantic_point_offset(row, group_idx);
    const float3 point = make_float3(
        params.points_xyz[point_offset + 0u],
        params.points_xyz[point_offset + 1u],
        params.points_xyz[point_offset + 2u]
    );
    if (inside_semantic_region(point, path_idx)) {
        optixReportIntersection(2.0f, 0, path_idx);
    }
}

extern "C" __global__ void __anyhit__gafime_dp_mark()
{
    const uint32_t row = optixGetPayload_0();
    const bool triangle_2d_instanced = params.geometry_mode == 2u;
    const uint32_t group_idx = current_group_index();
    const uint32_t path_base = params.group_path_offsets != nullptr ? params.group_path_offsets[group_idx] : 0u;
    const uint32_t path_idx = triangle_2d_instanced
        ? path_base + (optixGetPrimitiveIndex() >> 1u)
        : optixGetAttribute_0();
    if (path_idx < params.path_count) {
        /* Preserve the mature legacy triangle guard verbatim.  Compact
         * queries install an exact-term pointer and deliberately take the
         * second branch, so their conservative triangles are still guarded
         * by the original conjunction rather than an AABB approximation. */
        if (triangle_2d_instanced && !inside_box(optixGetWorldRayOrigin(), path_idx) &&
            params.semantic_exact_terms == nullptr) {
            optixIgnoreIntersection();
            return;
        }
        if (triangle_2d_instanced && params.semantic_exact_terms != nullptr &&
            !inside_semantic_region(optixGetWorldRayOrigin(), path_idx)) {
            optixIgnoreIntersection();
            return;
        }
        if (params.semantic_exact_terms != nullptr && params.semantic_region_ids != nullptr) {
            const uint32_t result_region = params.semantic_region_ids[path_idx];
            if (result_region >= params.path_count) {
                optixIgnoreIntersection();
                return;
            }
            if (params.semantic_direct_stats != 0u) {
                GafimeSemanticRtRegionExactStats& record = params.semantic_stats[result_region];
                const uint64_t point_offset = semantic_point_offset(row, group_idx);
                const float3 counterpart = make_float3(
                    params.semantic_counterpart_points_xyz != nullptr
                        ? params.semantic_counterpart_points_xyz[point_offset + 0u] : 0.0f,
                    params.semantic_counterpart_points_xyz != nullptr
                        ? params.semantic_counterpart_points_xyz[point_offset + 1u] : 0.0f,
                    params.semantic_counterpart_points_xyz != nullptr
                        ? params.semantic_counterpart_points_xyz[point_offset + 2u] : 0.0f
                );
                const bool paired_requested =
                    (params.semantic_statistic_mask &
                     GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED) != 0u;
                const bool labeled_requested =
                    (params.semantic_statistic_mask &
                     GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED) != 0u;
                const bool occupancy_requested =
                    (params.semantic_statistic_mask &
                     GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY) != 0u;
                const unsigned long long one = 1ull;
                if (params.semantic_view == 0u) {
                    if (occupancy_requested) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(&record.occupancy_inside), one);
                    }
                    if (labeled_requested) {
                        const uint32_t word = row >> 5u;
                        const uint32_t bit = 1u << (row & 31u);
                        if ((params.semantic_label_zero_words[word] & bit) != 0u) {
                            atomicAdd(reinterpret_cast<unsigned long long*>(&record.label_inside_0), one);
                        } else if ((params.semantic_label_one_words[word] & bit) != 0u) {
                            atomicAdd(reinterpret_cast<unsigned long long*>(&record.label_inside_1), one);
                        }
                    }
                    if (paired_requested && params.semantic_counterpart_points_xyz != nullptr) {
                        if (inside_semantic_region(counterpart, path_idx)) {
                            atomicAdd(reinterpret_cast<unsigned long long*>(&record.paired_n11), one);
                        } else {
                            atomicAdd(reinterpret_cast<unsigned long long*>(&record.paired_n10), one);
                        }
                    }
                    if (params.semantic_direct_region_ordinals != nullptr) {
                        /* Direct first-hit is admitted only for one finite,
                         * bounded, pairwise non-overlapping 2D group.  Its
                         * terminating accepted callback therefore carries one
                         * exact canonical result-region ordinal for this row.
                         * Store ordinal+1 so zero remains no membership. */
                        params.semantic_direct_region_ordinals[row] = result_region + 1u;
                    }
                } else if (paired_requested && params.semantic_counterpart_points_xyz != nullptr &&
                           !inside_semantic_region(counterpart, path_idx)) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&record.paired_n01), one);
                }
                optixTerminateRay();
                return;
            }
            if (params.membership_words != nullptr) {
                const uint64_t word_idx = static_cast<uint64_t>(result_region) *
                    params.words_per_path + (row >> 5u);
                atomicOr(&params.membership_words[word_idx], 1u << (row & 31u));
                optixIgnoreIntersection();
                return;
            }
        }
        if (params.direct_inside_counts != nullptr) {
            bool first_callback = true;
            if (params.membership_words != nullptr) {
                const uint64_t word_idx =
                    static_cast<uint64_t>(path_idx) * params.words_per_path + (row >> 5u);
                const uint32_t row_mask = 1u << (row & 31u);
                first_callback = (atomicOr(&params.membership_words[word_idx], row_mask) & row_mask) == 0u;
            }
            const float y = params.target[row];
            if (first_callback && isfinite(y)) {
                atomicAdd(&params.direct_inside_counts[path_idx], 1u);
                const double centered_y = static_cast<double>(y) - params.target_stats[1];
                atomicAdd(&params.direct_inside_sum_y[path_idx], centered_y);
            }
            if (params.direct_first_hit != 0u) {
                optixTerminateRay();
                return;
            }
        } else if (params.membership_words != nullptr) {
            const uint64_t word_idx =
                static_cast<uint64_t>(path_idx) * params.words_per_path + (row >> 5u);
            atomicOr(&params.membership_words[word_idx], 1u << (row & 31u));
        } else {
            const uint64_t out_idx = static_cast<uint64_t>(path_idx) * params.rows + row;
            params.membership[out_idx] = 1.0f;
        }
    }
    optixIgnoreIntersection();
}

#else

#include <cuda_runtime.h>

#include <cmath>

namespace gafime_cuda_v1::rt_kernel {

__global__ void validate_rt_feature_domain_kernel(
    const float* features,
    uint64_t value_count,
    uint32_t* invalid_out
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < value_count;
         index += stride) {
        const uint32_t magnitude_bits = __float_as_uint(features[index]) & 0x7fffffffu;
        const bool is_subnormal =
            (magnitude_bits & 0x7f800000u) == 0u &&
            (magnitude_bits & 0x007fffffu) != 0u;
        if (is_subnormal) {
            atomicExch(invalid_out, 1u);
            return;
        }
    }
}

__global__ void validate_semantic_region_input_domain_kernel(
    const float* columns,
    uint64_t rows,
    const uint32_t* input_slots,
    uint32_t input_slot_count,
    uint32_t* invalid_out
) {
    const uint64_t value_count = rows * static_cast<uint64_t>(input_slot_count);
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < value_count;
         index += stride) {
        const uint32_t slot_index = static_cast<uint32_t>(index / rows);
        const uint64_t row = index - static_cast<uint64_t>(slot_index) * rows;
        const float value = columns[static_cast<uint64_t>(input_slots[slot_index]) * rows + row];
        const uint32_t magnitude_bits = __float_as_uint(value) & 0x7fffffffu;
        const bool nonfinite = (magnitude_bits & 0x7f800000u) == 0x7f800000u;
        const bool subnormal =
            (magnitude_bits & 0x7f800000u) == 0u &&
            (magnitude_bits & 0x007fffffu) != 0u;
        if (nonfinite || subnormal) {
            atomicExch(invalid_out, 1u);
            return;
        }
    }
}

__global__ void scatter_semantic_region_membership_kernel(
    const float* membership,
    uint64_t rows,
    uint32_t region_count,
    const uint32_t* output_slots,
    float* columns,
    uint32_t* invalid_out
) {
    const uint64_t value_count = rows * static_cast<uint64_t>(region_count);
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < value_count;
         index += stride) {
        const uint32_t region = static_cast<uint32_t>(index / rows);
        const uint64_t row = index - static_cast<uint64_t>(region) * rows;
        const float value = membership[index];
        const uint32_t bits = __float_as_uint(value);
        if (bits != 0u && bits != 0x3f800000u) {
            atomicExch(invalid_out, 1u);
            continue;
        }
        columns[static_cast<uint64_t>(output_slots[region]) * rows + row] = value;
    }
}

__device__ inline bool semantic_region_terms_hold(
    const float* columns,
    uint64_t rows,
    const GafimeDecisionPathTerm* terms,
    uint32_t begin,
    uint32_t end,
    uint64_t row
) {
    for (uint32_t term_index = begin; term_index < end; ++term_index) {
        const GafimeDecisionPathTerm term = terms[term_index];
        const float value = columns[static_cast<uint64_t>(term.feature) * rows + row];
        const bool holds = term.sign == GAFIME_DECISION_PATH_SIGN_LE
            ? value <= term.threshold
            : value > term.threshold;
        if (!holds) return false;
    }
    return true;
}

__global__ void semantic_region_binary_label_masks_kernel(
    const uint64_t* row_indices,
    const uint8_t* values,
    uint64_t label_count,
    uint32_t* label_zero_words,
    uint32_t* label_one_words
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < label_count;
         index += stride) {
        const uint64_t row = row_indices[index];
        const uint32_t word = static_cast<uint32_t>(row >> 5u);
        const uint32_t bit = 1u << (row & 31u);
        if (values[index] == 0u) {
            atomicOr(&label_zero_words[word], bit);
        } else {
            atomicOr(&label_one_words[word], bit);
        }
    }
}

__global__ void semantic_region_membership_masks_sm_kernel(
    const float* primary_columns,
    const float* paired_columns,
    uint64_t rows,
    const GafimeDecisionPathTerm* primary_terms,
    const GafimeDecisionPathTerm* paired_terms,
    const uint32_t* region_offsets,
    uint32_t region_count,
    uint32_t words_per_region,
    uint32_t* primary_membership_words,
    uint32_t* paired_membership_words
) {
    const uint32_t region = blockIdx.x;
    const uint64_t row = static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (region >= region_count || row >= rows) return;

    const uint32_t begin = region_offsets[region];
    const uint32_t end = region_offsets[region + 1u];
    const uint64_t word_offset =
        static_cast<uint64_t>(region) * words_per_region + (row >> 5u);
    const uint32_t bit = 1u << (row & 31u);
    if (semantic_region_terms_hold(primary_columns, rows, primary_terms, begin, end, row)) {
        atomicOr(&primary_membership_words[word_offset], bit);
    }
    if (paired_columns != nullptr && paired_membership_words != nullptr &&
        semantic_region_terms_hold(paired_columns, rows, paired_terms, begin, end, row)) {
        atomicOr(&paired_membership_words[word_offset], bit);
    }
}

constexpr uint32_t kSemanticRegionSmBinCount = 256u;

__device__ inline uint32_t semantic_region_sm_bin(
    float value,
    float lo,
    float inv_span
) {
    if (isfinite(lo) && inv_span > 0.0f && isfinite(inv_span)) {
        const float scaled = (value - lo) * inv_span *
            static_cast<float>(kSemanticRegionSmBinCount);
        if (scaled <= 0.0f) return 0u;
        if (scaled >= static_cast<float>(kSemanticRegionSmBinCount)) {
            return kSemanticRegionSmBinCount - 1u;
        }
        return static_cast<uint32_t>(scaled);
    }
    const uint64_t bucket = static_cast<uint64_t>(rt_float_bucket(value));
    const uint32_t bin = static_cast<uint32_t>(
        (bucket * kSemanticRegionSmBinCount) >> 23u
    );
    return bin < kSemanticRegionSmBinCount ? bin : kSemanticRegionSmBinCount - 1u;
}

__device__ inline bool semantic_region_point_terms_hold(
    const float* point,
    const GafimeDecisionPathTerm* terms,
    uint32_t begin,
    uint32_t end
) {
    for (uint32_t term_index = begin; term_index < end; ++term_index) {
        const GafimeDecisionPathTerm term = terms[term_index];
        if (term.feature >= 3u) return false;
        const float value = point[term.feature];
        const bool holds = term.sign == GAFIME_DECISION_PATH_SIGN_LE
            ? value <= term.threshold
            : value > term.threshold;
        if (!holds) return false;
    }
    return true;
}

__global__ void semantic_region_membership_masks_binned_sm_kernel(
    const float* primary_points_xyz,
    const float* paired_points_xyz,
    uint64_t rows,
    const GafimeDecisionPathTerm* exact_terms,
    const uint32_t* exact_region_offsets,
    const uint32_t* group_path_offsets,
    const uint32_t* group_region_ids,
    const uint32_t* bin_offsets,
    const uint32_t* bin_candidates,
    const float* bin_lo,
    const float* bin_inv_span,
    uint32_t group_count,
    uint32_t point_stride,
    uint32_t point_group_stride,
    uint32_t words_per_region,
    uint32_t* primary_membership_words,
    uint32_t* paired_membership_words
) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t group = blockIdx.y;
    if (row >= rows || group >= group_count) return;

    const uint64_t point_offset =
        static_cast<uint64_t>(group) * point_group_stride + row * point_stride;
    const float* primary_point = primary_points_xyz + point_offset;
    const float* paired_point = paired_points_xyz == nullptr
        ? nullptr
        : paired_points_xyz + point_offset;
    const uint64_t bin_base = static_cast<uint64_t>(group) *
        (kSemanticRegionSmBinCount + 1u);
    const uint32_t group_begin = group_path_offsets[group];
    const uint32_t group_end = group_path_offsets[group + 1u];
    const uint32_t bit = 1u << (row & 31u);
    const uint32_t view_count = paired_point != nullptr && paired_membership_words != nullptr ? 2u : 1u;
    for (uint32_t view = 0u; view < view_count; ++view) {
        const float* point = view == 0u ? primary_point : paired_point;
        uint32_t* membership_words = view == 0u
            ? primary_membership_words
            : paired_membership_words;
        /* The paired point gets its own query-bound bin lookup.  Reusing the
         * primary candidate bin would silently lose paired-only membership
         * when paired slots cross a bin boundary. */
        const uint32_t bin = semantic_region_sm_bin(
            point[0], bin_lo[group], bin_inv_span[group]);
        const uint32_t begin_candidate = bin_offsets[bin_base + bin];
        const uint32_t end_candidate = bin_offsets[bin_base + bin + 1u];
        for (uint32_t candidate_index = begin_candidate;
             candidate_index < end_candidate;
             ++candidate_index) {
            const uint32_t physical_region = bin_candidates[candidate_index];
            if (physical_region < group_begin || physical_region >= group_end) continue;
            const uint32_t result_region = group_region_ids[physical_region];
            const uint64_t word_offset =
                static_cast<uint64_t>(result_region) * words_per_region + (row >> 5u);
            const uint32_t begin = exact_region_offsets[physical_region];
            const uint32_t end = exact_region_offsets[physical_region + 1u];
            if (semantic_region_point_terms_hold(point, exact_terms, begin, end)) {
                atomicOr(&membership_words[word_offset], bit);
            }
        }
    }
}

__global__ void reduce_semantic_region_membership_masks_kernel(
    const uint32_t* primary_membership_words,
    const uint32_t* paired_membership_words,
    const uint32_t* label_zero_words,
    const uint32_t* label_one_words,
    uint64_t rows,
    uint32_t region_count,
    uint32_t words_per_region,
    uint32_t statistic_mask,
    uint64_t label_zero_count,
    uint64_t label_one_count,
    GafimeSemanticRtRegionExactStats* stats
) {
    const uint32_t region = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (region >= region_count) return;

    uint64_t occupancy = 0u;
    uint64_t paired_n01 = 0u;
    uint64_t paired_n10 = 0u;
    uint64_t paired_n11 = 0u;
    uint64_t label_inside_0 = 0u;
    uint64_t label_inside_1 = 0u;
    const uint64_t base = static_cast<uint64_t>(region) * words_per_region;
    const bool want_paired =
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED) != 0u;
    const bool want_labels =
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED) != 0u;
    for (uint32_t word = lane; word < words_per_region; word += blockDim.x) {
        const uint32_t primary = primary_membership_words[base + word];
        occupancy += static_cast<uint64_t>(__popc(primary));
        if (want_paired) {
            const uint32_t paired = paired_membership_words[base + word];
            paired_n11 += static_cast<uint64_t>(__popc(primary & paired));
            paired_n10 += static_cast<uint64_t>(__popc(primary & ~paired));
            paired_n01 += static_cast<uint64_t>(__popc(~primary & paired));
        }
        if (want_labels) {
            label_inside_0 += static_cast<uint64_t>(
                __popc(primary & label_zero_words[word])
            );
            label_inside_1 += static_cast<uint64_t>(
                __popc(primary & label_one_words[word])
            );
        }
    }

    __shared__ uint64_t occupancy_sum[256];
    __shared__ uint64_t paired_n01_sum[256];
    __shared__ uint64_t paired_n10_sum[256];
    __shared__ uint64_t paired_n11_sum[256];
    __shared__ uint64_t label_inside_0_sum[256];
    __shared__ uint64_t label_inside_1_sum[256];
    occupancy_sum[lane] = occupancy;
    paired_n01_sum[lane] = paired_n01;
    paired_n10_sum[lane] = paired_n10;
    paired_n11_sum[lane] = paired_n11;
    label_inside_0_sum[lane] = label_inside_0;
    label_inside_1_sum[lane] = label_inside_1;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            occupancy_sum[lane] += occupancy_sum[lane + stride];
            paired_n01_sum[lane] += paired_n01_sum[lane + stride];
            paired_n10_sum[lane] += paired_n10_sum[lane + stride];
            paired_n11_sum[lane] += paired_n11_sum[lane + stride];
            label_inside_0_sum[lane] += label_inside_0_sum[lane + stride];
            label_inside_1_sum[lane] += label_inside_1_sum[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0u) {
        GafimeSemanticRtRegionExactStats record = {};
        record.row_count = rows;
        if ((statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY) != 0u) {
            record.occupancy_inside = occupancy_sum[0];
        }
        if (want_paired) {
            const uint64_t used = paired_n01_sum[0] + paired_n10_sum[0] + paired_n11_sum[0];
            record.paired_n00 = used <= rows ? rows - used : 0u;
            record.paired_n01 = paired_n01_sum[0];
            record.paired_n10 = paired_n10_sum[0];
            record.paired_n11 = paired_n11_sum[0];
        }
        if (want_labels) {
            record.label_support = label_zero_count + label_one_count;
            record.label_inside_0 = label_inside_0_sum[0];
            record.label_inside_1 = label_inside_1_sum[0];
            record.label_outside_0 = label_inside_0_sum[0] <= label_zero_count
                ? label_zero_count - label_inside_0_sum[0]
                : 0u;
            record.label_outside_1 = label_inside_1_sum[0] <= label_one_count
                ? label_one_count - label_inside_1_sum[0]
                : 0u;
        }
        stats[region] = record;
    }
}

__device__ inline float semantic_region_divide_f32(uint64_t numerator, uint64_t denominator) {
    return __fdiv_rn(static_cast<float>(numerator), static_cast<float>(denominator));
}

__device__ inline float semantic_region_gini_f32(uint64_t zero_count, uint64_t one_count) {
    const uint64_t total = zero_count + one_count;
    const float p0 = semantic_region_divide_f32(zero_count, total);
    const float p1 = semantic_region_divide_f32(one_count, total);
    const float p0_sq = __fmul_rn(p0, p0);
    const float p1_sq = __fmul_rn(p1, p1);
    return __fsub_rn(__fsub_rn(1.0f, p0_sq), p1_sq);
}

__global__ void finalize_semantic_region_stats_kernel(
    uint32_t region_count,
    uint32_t statistic_mask,
    uint32_t finalizer_mask,
    uint64_t rows,
    uint64_t label_zero_count,
    uint64_t label_one_count,
    GafimeSemanticRtRegionExactStats* stats
) {
    const uint32_t region = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (region >= region_count) return;
    GafimeSemanticRtRegionExactStats& record = stats[region];

    record.row_count = rows;
    if ((statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED) != 0u) {
        const uint64_t used = record.paired_n01 + record.paired_n10 + record.paired_n11;
        record.paired_n00 = used <= rows ? rows - used : 0u;
    }
    if ((statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED) != 0u) {
        record.label_support = label_zero_count + label_one_count;
        record.label_outside_0 = record.label_inside_0 <= label_zero_count
            ? label_zero_count - record.label_inside_0
            : 0u;
        record.label_outside_1 = record.label_inside_1 <= label_one_count
            ? label_one_count - record.label_inside_1
            : 0u;
    }

    if ((finalizer_mask & GAFIME_SEMANTIC_RT_REGION_FINALIZE_OCCUPANCY) != 0u &&
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY) != 0u) {
        if (record.row_count == 0u) {
            record.occupancy_state = GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT;
        } else {
            record.occupancy = semantic_region_divide_f32(record.occupancy_inside, record.row_count);
            record.occupancy_state = GAFIME_SEMANTIC_SCALAR_MEASURED;
        }
    }

    if ((finalizer_mask & GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_AGREEMENT) != 0u &&
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED) != 0u) {
        if (record.row_count == 0u) {
            record.paired_agreement_state = GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT;
        } else {
            record.paired_agreement = semantic_region_divide_f32(
                record.paired_n00 + record.paired_n11,
                record.row_count
            );
            record.paired_agreement_state = GAFIME_SEMANTIC_SCALAR_MEASURED;
        }
    }

    if ((finalizer_mask & GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_IOU) != 0u &&
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED) != 0u) {
        const uint64_t union_count =
            record.paired_n01 + record.paired_n10 + record.paired_n11;
        if (union_count == 0u) {
            record.paired_iou_state = GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND;
        } else {
            record.paired_iou = semantic_region_divide_f32(record.paired_n11, union_count);
            record.paired_iou_state = GAFIME_SEMANTIC_SCALAR_MEASURED;
        }
    }

    if ((finalizer_mask & GAFIME_SEMANTIC_RT_REGION_FINALIZE_LABELED_GINI_GAIN) != 0u &&
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED) != 0u) {
        const uint64_t support = record.label_support;
        const uint64_t global_zero = record.label_outside_0 + record.label_inside_0;
        const uint64_t global_one = record.label_outside_1 + record.label_inside_1;
        const uint64_t outside = record.label_outside_0 + record.label_outside_1;
        const uint64_t inside = record.label_inside_0 + record.label_inside_1;
        if (support < 2u) {
            record.labeled_gini_gain_state = GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT;
        } else if (global_zero == 0u || global_one == 0u || outside == 0u || inside == 0u) {
            record.labeled_gini_gain_state = GAFIME_SEMANTIC_SCALAR_CONSTANT_OPERAND;
        } else {
            const float parent = semantic_region_gini_f32(global_zero, global_one);
            const float outside_weight = semantic_region_divide_f32(outside, support);
            const float inside_weight = semantic_region_divide_f32(inside, support);
            const float outside_term = __fmul_rn(
                outside_weight,
                semantic_region_gini_f32(record.label_outside_0, record.label_outside_1)
            );
            const float inside_term = __fmul_rn(
                inside_weight,
                semantic_region_gini_f32(record.label_inside_0, record.label_inside_1)
            );
            record.labeled_gini_gain = __fsub_rn(
                __fsub_rn(parent, outside_term),
                inside_term
            );
            record.labeled_gini_gain_state = GAFIME_SEMANTIC_SCALAR_MEASURED;
        }
    }
}

__global__ void materialize_semantic_region_coverage_kernel(
    const uint32_t* primary_membership_words,
    uint64_t rows,
    uint32_t region_count,
    uint32_t words_per_region,
    float* output_column
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const uint32_t word = static_cast<uint32_t>(row >> 5u);
        const uint32_t bit = 1u << (row & 31u);
        uint32_t count = 0u;
        for (uint32_t region = 0u; region < region_count; ++region) {
            const uint64_t word_offset = static_cast<uint64_t>(region) * words_per_region + word;
            count += (primary_membership_words[word_offset] & bit) != 0u ? 1u : 0u;
        }
        output_column[row] = static_cast<float>(count);
    }
}

__global__ void materialize_semantic_region_weighted_sum_kernel(
    const uint32_t* primary_membership_words,
    uint64_t rows,
    uint32_t region_count,
    uint32_t words_per_region,
    const float* region_weights,
    float* output_column,
    uint32_t* nonfinite_out
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const uint32_t word = static_cast<uint32_t>(row >> 5u);
        const uint32_t bit = 1u << (row & 31u);
        float sum = 0.0f;
        for (uint32_t region = 0u; region < region_count; ++region) {
            const uint64_t word_offset = static_cast<uint64_t>(region) * words_per_region + word;
            if ((primary_membership_words[word_offset] & bit) != 0u) {
                /* This is one explicit round-to-nearest addition, not a
                 * multiply/accumulate expression.  Canonical region order
                 * supplies the only permitted accumulation order. */
                sum = __fadd_rn(sum, region_weights[region]);
            }
        }
        if (!isfinite(sum)) {
            atomicExch(nonfinite_out, 1u);
            continue;
        }
        output_column[row] = sum;
    }
}

__global__ void materialize_semantic_region_weighted_sum_ordinals_kernel(
    const uint32_t* direct_region_ordinals,
    uint64_t rows,
    uint32_t region_count,
    const float* region_weights,
    float* output_column,
    uint32_t* nonfinite_out
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const uint32_t ordinal_plus_one = direct_region_ordinals[row];
        if (ordinal_plus_one > region_count) {
            atomicExch(nonfinite_out, 1u);
            continue;
        }
        float sum = 0.0f;
        if (ordinal_plus_one != 0u) {
            sum = __fadd_rn(sum, region_weights[ordinal_plus_one - 1u]);
        }
        if (!isfinite(sum)) {
            atomicExch(nonfinite_out, 1u);
            continue;
        }
        output_column[row] = sum;
    }
}

__global__ void scatter_semantic_region_coverage_counts_kernel(
    const uint32_t* direct_region_ordinals,
    uint64_t rows,
    float* output_column
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        output_column[row] = direct_region_ordinals[row] == 0u ? 0.0f : 1.0f;
    }
}

__global__ void pack_decision_path_points_kernel(
    const float* features,
    uint64_t n_samples,
    uint32_t axis0,
    uint32_t axis1,
    uint32_t axis2,
    uint32_t dims,
    float* points_xyz
) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_samples) {
        return;
    }
    const uint32_t axes[3] = {axis0, axis1, axis2};
    for (uint32_t dim = 0; dim < 3u; ++dim) {
        const float value = dim < dims ? features[static_cast<uint64_t>(axes[dim]) * n_samples + row] : 0.0f;
        points_xyz[row * 3u + dim] = value;
    }
}

__global__ void pack_grouped_decision_path_points_kernel(
    const float* features,
    uint64_t n_samples,
    const uint32_t* group_axes,
    const uint32_t* group_dims,
    uint32_t group_count,
    uint32_t point_stride,
    float* points_xyz
) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t group_idx = blockIdx.y;
    if (row >= n_samples || group_idx >= group_count) {
        return;
    }
    const uint32_t dims = group_dims[group_idx];
    const uint32_t* axes = group_axes + static_cast<uint64_t>(group_idx) * 3u;
    const uint64_t point_base =
        static_cast<uint64_t>(group_idx) * n_samples * point_stride +
        row * point_stride;
    for (uint32_t dim = 0; dim < point_stride; ++dim) {
        const float value = dim < dims ? features[static_cast<uint64_t>(axes[dim]) * n_samples + row] : 0.0f;
        points_xyz[point_base + dim] = value;
    }
}

__global__ void decision_path_membership_kernel(
    const float* features,
    uint64_t n_samples,
    uint64_t row_offset,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership
) {
    const uint32_t path_idx = blockIdx.x;
    const uint64_t row = row_offset + static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count || row >= n_samples) {
        return;
    }

    const uint32_t begin = path_offsets[path_idx];
    const uint32_t end = path_offsets[path_idx + 1];
    bool member = true;
    bool undetermined = false;

    for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
        const GafimeDecisionPathTerm term = terms[term_idx];
        if (term.feature >= n_features) {
            member = false;
            break;
        }
        const float x = features[static_cast<uint64_t>(term.feature) * n_samples + row];
        if (isnan(x)) {
            undetermined = true;
            continue;
        }
        const bool holds =
            term.sign == GAFIME_DECISION_PATH_SIGN_LE ? x <= term.threshold : x > term.threshold;
        if (!holds) {
            member = false;
            break;
        }
    }

    membership[static_cast<uint64_t>(path_idx) * n_samples + row] =
        member ? (undetermined ? nanf("") : 1.0f) : 0.0f;
}

__global__ void decision_path_bitset_kernel(
    const float* features,
    uint64_t n_samples,
    uint64_t row_offset,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    uint32_t words_per_path,
    uint32_t* membership_words
) {
    const uint32_t path_idx = blockIdx.x;
    const uint64_t row = row_offset + static_cast<uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count || row >= n_samples) {
        return;
    }

    const uint32_t begin = path_offsets[path_idx];
    const uint32_t end = path_offsets[path_idx + 1];
    bool member = true;
    for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
        const GafimeDecisionPathTerm term = terms[term_idx];
        if (term.feature >= n_features) {
            member = false;
            break;
        }
        const float x = features[static_cast<uint64_t>(term.feature) * n_samples + row];
        if (!isfinite(x)) {
            member = false;
            break;
        }
        const bool holds =
            term.sign == GAFIME_DECISION_PATH_SIGN_LE ? x <= term.threshold : x > term.threshold;
        if (!holds) {
            member = false;
            break;
        }
    }

    if (member) {
        const uint64_t word_idx =
            static_cast<uint64_t>(path_idx) * words_per_path + (row >> 5u);
        atomicOr(&membership_words[word_idx], 1u << (row & 31u));
    }
}

__global__ void score_decision_path_bitset_kernel(
    const uint32_t* membership_words,
    const float* target,
    const double* target_stats,
    uint64_t n_samples,
    uint32_t path_count,
    uint32_t words_per_path,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint32_t path_idx = blockIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    uint64_t local_sx = 0u;
    double local_sxy = 0.0;
    const double mean_y = target_stats[1];
    const uint64_t path_offset = static_cast<uint64_t>(path_idx) * words_per_path;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            const uint32_t word = membership_words[path_offset + (row >> 5u)];
            if (((word >> (row & 31u)) & 1u) != 0u) {
                ++local_sx;
                local_sxy += static_cast<double>(y) - mean_y;
            }
        }
    }

    __shared__ uint64_t sx[256];
    __shared__ double sxy[256];
    sx[threadIdx.x] = local_sx;
    sxy[threadIdx.x] = local_sxy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sx[threadIdx.x] += sx[threadIdx.x + stride];
            sxy[threadIdx.x] += sxy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double pearson = 0.0;
        const double n = target_stats[0];
        if (n > 0.0) {
            const double count = static_cast<double>(sx[0]);
            const double sxx = fmax(count - count * count / n, 0.0);
            const double syy = fmax(target_stats[2], 0.0);
            const double denom = sqrt(fmax(sxx * syy, 0.0));
            if (denom > 0.0) {
                pearson = fmin(fmax(sxy[0] / denom, -1.0), 1.0);
            }
        }
        for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
            const uint32_t metric_id = metric_ids[metric_idx];
            float out = 0.0f;
            if (metric_id == GAFIME_METRIC_PEARSON) {
                out = static_cast<float>(pearson);
            } else if (metric_id == GAFIME_METRIC_R2) {
                out = static_cast<float>(fmin(fmax(pearson * pearson, 0.0), 1.0));
            }
            metric_values[static_cast<uint64_t>(path_idx) * metric_count + metric_idx] = out;
        }
    }
}

__global__ void decision_path_target_stats_kernel(
    const float* target,
    uint64_t n_samples,
    double* target_stats
) {
    uint64_t local_n = 0u;
    double local_sy = 0.0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            ++local_n;
            local_sy += static_cast<double>(y);
        }
    }

    __shared__ uint64_t sn[256];
    __shared__ double sy[256];
    __shared__ double mean_y;
    __shared__ double syy[256];
    sn[threadIdx.x] = local_n;
    sy[threadIdx.x] = local_sy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            sn[threadIdx.x] += sn[threadIdx.x + stride];
            sy[threadIdx.x] += sy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mean_y = sn[0] > 0u ? sy[0] / static_cast<double>(sn[0]) : 0.0;
    }
    __syncthreads();

    double local_syy = 0.0;
    for (uint64_t row = threadIdx.x; row < n_samples; row += blockDim.x) {
        const float y = target[row];
        if (isfinite(y)) {
            const double dy = static_cast<double>(y) - mean_y;
            local_syy += dy * dy;
        }
    }
    syy[threadIdx.x] = local_syy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            syy[threadIdx.x] += syy[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        target_stats[0] = static_cast<double>(sn[0]);
        target_stats[1] = mean_y;
        target_stats[2] = syy[0];
    }
}

static __forceinline__ __device__ float decision_path_binary_pearson(
    uint32_t inside_count,
    double centered_inside_sum,
    const double* target_stats
) {
    const double n = target_stats[0];
    if (n <= 0.0) {
        return 0.0f;
    }
    const double count = static_cast<double>(inside_count);
    const double sxx = fmax(count - count * count / n, 0.0);
    const double syy = fmax(target_stats[2], 0.0);
    const double denom = sqrt(fmax(sxx * syy, 0.0));
    if (denom <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(fmin(fmax(centered_inside_sum / denom, -1.0), 1.0));
}

__global__ void score_decision_path_direct_stats_kernel(
    const uint32_t* inside_counts,
    const double* inside_sum_y,
    const double* target_stats,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* metric_values
) {
    const uint32_t path_idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    const float pearson = decision_path_binary_pearson(
        inside_counts[path_idx],
        inside_sum_y[path_idx],
        target_stats
    );

    for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
        const uint32_t metric_id = metric_ids[metric_idx];
        float out = 0.0f;
        if (metric_id == GAFIME_METRIC_PEARSON) {
            out = pearson;
        } else if (metric_id == GAFIME_METRIC_R2) {
            out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
        }
        metric_values[static_cast<uint64_t>(path_idx) * metric_count + metric_idx] = out;
    }
}

__global__ void score_decision_path_direct_stats_scatter_kernel(
    const uint32_t* inside_counts,
    const double* inside_sum_y,
    const double* target_stats,
    const uint32_t* original_paths,
    uint32_t path_count,
    const uint32_t* metric_ids,
    uint32_t metric_count,
    float* final_metric_values
) {
    const uint32_t path_idx = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path_idx >= path_count) {
        return;
    }

    const float pearson = decision_path_binary_pearson(
        inside_counts[path_idx],
        inside_sum_y[path_idx],
        target_stats
    );

    const uint32_t original_path = original_paths[path_idx];
    for (uint32_t metric_idx = 0; metric_idx < metric_count; ++metric_idx) {
        const uint32_t metric_id = metric_ids[metric_idx];
        float out = 0.0f;
        if (metric_id == GAFIME_METRIC_PEARSON) {
            out = pearson;
        } else if (metric_id == GAFIME_METRIC_R2) {
            out = fminf(fmaxf(pearson * pearson, 0.0f), 1.0f);
        }
        final_metric_values[static_cast<uint64_t>(original_path) * metric_count + metric_idx] = out;
    }
}

__global__ void scatter_decision_path_score_metrics_kernel(
    const float* group_metric_values,
    const uint32_t* original_paths,
    uint32_t group_path_count,
    uint32_t metric_count,
    float* final_metric_values
) {
    const uint64_t value_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t value_count = static_cast<uint64_t>(group_path_count) * metric_count;
    if (value_idx >= value_count) {
        return;
    }
    const uint32_t local_path = static_cast<uint32_t>(value_idx / metric_count);
    const uint32_t metric_idx = static_cast<uint32_t>(value_idx - static_cast<uint64_t>(local_path) * metric_count);
    const uint32_t original_path = original_paths[local_path];
    final_metric_values[static_cast<uint64_t>(original_path) * metric_count + metric_idx] =
        group_metric_values[value_idx];
}

}  // namespace gafime_cuda_v1::rt_kernel

#endif
