#include "rt_launcher.cuh"
#include "cuda_internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "rt_kernels.cuh"
#include "../common/semantic_primitives_abi_impl.hpp"

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
#include <cuda.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "gafime_rt_optix_ptx.hpp"
#endif

namespace {

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return GAFIME_STATUS_OK;
    if (status == cudaErrorMemoryAllocation) return GAFIME_STATUS_OUT_OF_MEMORY;
    return GAFIME_STATUS_DEVICE_ERROR;
}

class ScopedCudaDevice {
public:
    explicit ScopedCudaDevice(uint32_t device_id) {
        if (device_id > static_cast<uint32_t>(INT_MAX)) {
            status_ = cudaErrorInvalidDevice;
            return;
        }
        status_ = cudaGetDevice(&previous_device_);
        if (status_ != cudaSuccess) {
            return;
        }
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

bool checked_mul_u64(uint64_t left, uint64_t right, uint64_t* out) {
    if (left != 0 && right > UINT64_MAX / left) {
        return false;
    }
    *out = left * right;
    return true;
}

bool checked_add_u64(uint64_t left, uint64_t right, uint64_t* out) {
    if (left > UINT64_MAX - right) {
        return false;
    }
    *out = left + right;
    return true;
}

bool checked_element_bytes(uint64_t count, size_t element_size, uint64_t* out) {
    if (out == nullptr) return false;
    uint64_t element_bytes = 0;
    if (element_size == 0 ||
        !checked_mul_u64(count, static_cast<uint64_t>(element_size), &element_bytes) ||
        count > static_cast<uint64_t>(SIZE_MAX / element_size)) {
        return false;
    }
    *out = element_bytes;
    return true;
}

bool checked_plan_add(uint64_t* total, uint64_t value) {
    uint64_t next = 0;
    if (total == nullptr || !checked_add_u64(*total, value, &next)) return false;
    *total = next;
    return true;
}

bool allocation_fits_size_t(uint64_t count, size_t element_size) {
    return element_size != 0 && count <= static_cast<uint64_t>(SIZE_MAX / element_size);
}

cudaError_t current_device_max_grid_y(uint32_t* max_grid_y_out) {
    if (max_grid_y_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int max_grid_y = 0;
    status = cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (max_grid_y <= 0) {
        return cudaErrorInvalidConfiguration;
    }
    *max_grid_y_out = static_cast<uint32_t>(max_grid_y);
    return cudaSuccess;
}

template <typename T>
int ensure_device_capacity(T** ptr, size_t& capacity, size_t count) {
    if (count <= capacity) {
        return GAFIME_STATUS_OK;
    }
    if (count == 0) {
        return GAFIME_STATUS_OK;
    }
    if (count > SIZE_MAX / sizeof(T)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    T* next = nullptr;
    const int status = cuda_status(cudaMalloc(
        reinterpret_cast<void**>(&next),
        count * sizeof(T)
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    cudaFree(*ptr);
    *ptr = next;
    capacity = count;
    return GAFIME_STATUS_OK;
}

int ensure_device_bytes(void** ptr, size_t& capacity, size_t bytes) {
    if (bytes <= capacity) {
        return GAFIME_STATUS_OK;
    }
    if (bytes == 0) {
        return GAFIME_STATUS_OK;
    }
    void* next = nullptr;
    const int status = cuda_status(cudaMalloc(&next, bytes));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    cudaFree(*ptr);
    *ptr = next;
    capacity = bytes;
    return GAFIME_STATUS_OK;
}

bool decision_path_sign_supported(uint32_t sign) {
    return sign == GAFIME_DECISION_PATH_SIGN_LE || sign == GAFIME_DECISION_PATH_SIGN_GT;
}

bool rt_required(const GafimeDecisionPathBatch* batch) {
    return batch != nullptr && (batch->flags & GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u;
}

int validate_decision_path_batch(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* batch
) {
    if (resident_features == nullptr || batch == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if ((batch->flags & ~GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_count == 0 || batch->path_count > GAFIME_MAX_DECISION_PATH_COUNT ||
        batch->term_count == 0 ||
        batch->terms == nullptr || batch->path_offsets == nullptr || batch->membership_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_offsets[0] != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t path_idx = 0; path_idx < batch->path_count; ++path_idx) {
        const uint32_t begin = batch->path_offsets[path_idx];
        const uint32_t end = batch->path_offsets[path_idx + 1];
        if (begin >= end || end > batch->term_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
            const GafimeDecisionPathTerm& term = batch->terms[term_idx];
            if (term.feature >= cols || !decision_path_sign_supported(term.sign)) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (batch->path_offsets[batch->path_count] != batch->term_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != 0 && batch->path_count > UINT64_MAX / rows) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t output_count = rows * static_cast<uint64_t>(batch->path_count);
    const uint64_t offset_count = static_cast<uint64_t>(batch->path_count) + 1u;
    if (!allocation_fits_size_t(batch->term_count, sizeof(GafimeDecisionPathTerm)) ||
        !allocation_fits_size_t(offset_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(output_count, sizeof(float))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_OK;
}

int validate_decision_path_score_batch(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathScoreBatch* batch,
    const GafimeResultTable* result
) {
    if (resident_features == nullptr || target == nullptr || batch == nullptr || result == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->abi_version != GAFIME_ABI_VERSION || result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if ((batch->flags & ~GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_count == 0 || batch->path_count > GAFIME_MAX_DECISION_PATH_COUNT ||
        batch->term_count == 0 || batch->metric_count == 0 ||
        batch->terms == nullptr || batch->path_offsets == nullptr || batch->metric_ids == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (result->capacity < batch->path_count || result->metric_count < batch->metric_count ||
        result->max_arity < 1u || result->combo_indices == nullptr ||
        result->metric_values == nullptr || result->ranks == nullptr ||
        result->families == nullptr || result->candidate_ids == nullptr ||
        result->row_flags == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_offsets[0] != 0u || batch->path_offsets[batch->path_count] != batch->term_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t metric_idx = 0; metric_idx < batch->metric_count; ++metric_idx) {
        const uint32_t metric = batch->metric_ids[metric_idx];
        if (metric != GAFIME_METRIC_PEARSON && metric != GAFIME_METRIC_R2) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    for (uint32_t path_idx = 0; path_idx < batch->path_count; ++path_idx) {
        const uint32_t begin = batch->path_offsets[path_idx];
        const uint32_t end = batch->path_offsets[path_idx + 1u];
        if (begin >= end || end > batch->term_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
            const GafimeDecisionPathTerm& term = batch->terms[term_idx];
            if (term.feature >= cols || !decision_path_sign_supported(term.sign)) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    const uint64_t offset_count = static_cast<uint64_t>(batch->path_count) + 1u;
    const uint64_t words_per_path = (rows + 31u) / 32u;
    uint64_t word_count = 0;
    uint64_t metric_value_count = 0;
    uint64_t result_combo_count = 0;
    uint64_t result_metric_count = 0;
    uint64_t triangle_vertex_count = 0;
    uint64_t triangle_index_count = 0;
    if (!checked_mul_u64(batch->path_count, words_per_path, &word_count) ||
        !checked_mul_u64(batch->path_count, batch->metric_count, &metric_value_count) ||
        !checked_mul_u64(batch->path_count, result->max_arity, &result_combo_count) ||
        !checked_mul_u64(batch->path_count, result->metric_count, &result_metric_count) ||
        !checked_mul_u64(batch->path_count, 4u, &triangle_vertex_count) ||
        !checked_mul_u64(batch->path_count, 2u, &triangle_index_count) ||
        !allocation_fits_size_t(batch->term_count, sizeof(GafimeDecisionPathTerm)) ||
        !allocation_fits_size_t(offset_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(batch->metric_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(word_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(metric_value_count, sizeof(float)) ||
        !allocation_fits_size_t(result_combo_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(result_metric_count, sizeof(float)) ||
        !allocation_fits_size_t(
            triangle_vertex_count,
            sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex)) ||
        !allocation_fits_size_t(
            triangle_index_count,
            sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex)) ||
        !allocation_fits_size_t(batch->path_count, sizeof(uint32_t)) ||
        !allocation_fits_size_t(batch->path_count, sizeof(uint64_t))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_OK;
}

bool rt_disabled_by_env() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT");
    return mode != nullptr && std::strcmp(mode, "off") == 0;
}

bool rt_score_first_hit_requested_env() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT_SCORE");
    return mode != nullptr && std::strcmp(mode, "firsthit") == 0;
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)

enum class RtGeometryMode : uint32_t {
    CustomAabb = 0,
    CustomAabbInstanced = 1,
    Triangle2dInstanced = 2,
};

bool rt_score_direct_stats_requested() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT_SCORE");
    return mode != nullptr && (
        std::strcmp(mode, "direct") == 0 ||
        std::strcmp(mode, "firsthit") == 0
    );
}

bool rt_score_first_hit_direct_requested() {
    return rt_score_first_hit_requested_env();
}

bool append_unique_axis(std::vector<uint32_t>& axes, uint32_t feature) {
    if (std::find(axes.begin(), axes.end(), feature) != axes.end()) {
        return true;
    }
    if (axes.size() >= 3) {
        return false;
    }
    axes.push_back(feature);
    return true;
}

uint32_t axis_index(const std::array<uint32_t, 3>& axes, uint32_t dims, uint32_t feature) {
    for (uint32_t idx = 0; idx < dims; ++idx) {
        if (axes[idx] == feature) {
            return idx;
        }
    }
    return UINT32_MAX;
}

struct RtBoxPlan {
    std::array<uint32_t, 3> axes{0, 0, 0};
    uint32_t dims = 0;
    bool all_boxes_bounded = true;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtBox> boxes;
};

int build_rt_box_plan(
    uint32_t path_count,
    uint32_t term_count,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    RtBoxPlan& plan
) {
    std::vector<uint32_t> axes;
    axes.reserve(3);
    for (uint32_t term_idx = 0; term_idx < term_count; ++term_idx) {
        const GafimeDecisionPathTerm& term = terms[term_idx];
        if (!std::isfinite(term.threshold) ||
            std::fpclassify(term.threshold) == FP_SUBNORMAL) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (!append_unique_axis(axes, term.feature)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    std::sort(axes.begin(), axes.end());
    plan.dims = static_cast<uint32_t>(axes.size());
    if (plan.dims == 0 || plan.dims > 3) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    for (uint32_t idx = 0; idx < plan.dims; ++idx) {
        plan.axes[idx] = axes[idx];
    }

    plan.boxes.assign(path_count, {});
    for (uint32_t path_idx = 0; path_idx < path_count; ++path_idx) {
        float lo[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        float hi[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        uint32_t open_lo_mask = 0;
        const uint32_t begin = path_offsets[path_idx];
        const uint32_t end = path_offsets[path_idx + 1];
        for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
            const GafimeDecisionPathTerm& term = terms[term_idx];
            const uint32_t axis = axis_index(plan.axes, plan.dims, term.feature);
            if (axis == UINT32_MAX) {
                return GAFIME_STATUS_UNSUPPORTED_BACKEND;
            }
            if (term.sign == GAFIME_DECISION_PATH_SIGN_LE) {
                hi[axis] = std::min(hi[axis], term.threshold);
            } else {
                if (term.threshold >= lo[axis]) {
                    lo[axis] = term.threshold;
                    open_lo_mask |= (1u << axis);
                }
            }
        }
        for (uint32_t axis = 0; axis < plan.dims; ++axis) {
            if (lo[axis] > hi[axis] || (lo[axis] == hi[axis] && (open_lo_mask & (1u << axis)) != 0u)) {
                lo[axis] = 0.0f;
                hi[axis] = 0.0f;
                open_lo_mask |= (1u << axis);
            }
            if (lo[axis] <= -FLT_MAX * 0.5f || hi[axis] >= FLT_MAX * 0.5f) {
                plan.all_boxes_bounded = false;
            }
        }
        plan.boxes[path_idx] = {
            lo[0],
            lo[1],
            lo[2],
            hi[0],
            hi[1],
            hi[2],
            open_lo_mask,
            plan.dims,
        };
    }
    return GAFIME_STATUS_OK;
}

int build_rt_box_plan(const GafimeDecisionPathBatch* paths, RtBoxPlan& plan) {
    return build_rt_box_plan(
        paths->path_count,
        paths->term_count,
        paths->terms,
        paths->path_offsets,
        plan
    );
}

int build_rt_box_plan(const GafimeDecisionPathScoreBatch* paths, RtBoxPlan& plan) {
    return build_rt_box_plan(
        paths->path_count,
        paths->term_count,
        paths->terms,
        paths->path_offsets,
        plan
    );
}

RtGeometryMode choose_rt_geometry_mode(const RtBoxPlan&) {
    return RtGeometryMode::CustomAabb;
}

bool rt_triangle_axis_is_safe(float lo, float hi) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) {
        return false;
    }
    const double span = static_cast<double>(hi) - static_cast<double>(lo);
    const double scale = std::max({
        1.0,
        std::abs(static_cast<double>(lo)),
        std::abs(static_cast<double>(hi)),
    });
    return span >= std::ldexp(scale, -12);
}

bool rt_box_plan_triangle2d_is_safe(const RtBoxPlan& plan) {
    if (plan.dims != 2u || !plan.all_boxes_bounded || plan.boxes.empty()) {
        return false;
    }
    return std::all_of(plan.boxes.begin(), plan.boxes.end(), [](const auto& box) {
        return rt_triangle_axis_is_safe(box.lo_x, box.hi_x) &&
            rt_triangle_axis_is_safe(box.lo_y, box.hi_y);
    });
}

bool rt_ranges_overlap_open_closed(float a_lo, float a_hi, float b_lo, float b_hi) {
    return std::max(a_lo, b_lo) < std::min(a_hi, b_hi);
}

bool rt_box_plan_non_overlapping_2d(const RtBoxPlan& plan) {
    if (plan.dims != 2u || !plan.all_boxes_bounded) {
        return false;
    }
    std::vector<uint32_t> order(plan.boxes.size());
    for (uint32_t idx = 0; idx < order.size(); ++idx) {
        order[idx] = idx;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
        const auto& a = plan.boxes[left];
        const auto& b = plan.boxes[right];
        if (a.lo_x != b.lo_x) {
            return a.lo_x < b.lo_x;
        }
        if (a.hi_x != b.hi_x) {
            return a.hi_x < b.hi_x;
        }
        return left < right;
    });

    std::vector<uint32_t> active;
    active.reserve(128u);
    for (uint32_t box_idx : order) {
        const auto& box = plan.boxes[box_idx];
        active.erase(
            std::remove_if(active.begin(), active.end(), [&](uint32_t active_idx) {
                return plan.boxes[active_idx].hi_x <= box.lo_x;
            }),
            active.end()
        );
        for (uint32_t active_idx : active) {
            const auto& other = plan.boxes[active_idx];
            if (rt_ranges_overlap_open_closed(other.lo_y, other.hi_y, box.lo_y, box.hi_y)) {
                return false;
            }
        }
        active.push_back(box_idx);
    }
    return true;
}

struct RtScoreGroup {
    std::vector<uint32_t> original_paths;
    std::vector<GafimeDecisionPathTerm> terms;
    std::vector<uint32_t> offsets{0u};
    std::vector<uint32_t> axes;
};

bool collect_path_axes(
    const GafimeDecisionPathScoreBatch* paths,
    uint32_t path_idx,
    std::vector<uint32_t>& axes
) {
    axes.clear();
    const uint32_t begin = paths->path_offsets[path_idx];
    const uint32_t end = paths->path_offsets[path_idx + 1u];
    for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
        const GafimeDecisionPathTerm& term = paths->terms[term_idx];
        if (!std::isfinite(term.threshold) ||
            std::fpclassify(term.threshold) == FP_SUBNORMAL ||
            !append_unique_axis(axes, term.feature)) {
            return false;
        }
    }
    std::sort(axes.begin(), axes.end());
    return !axes.empty();
}

bool merge_rt_axes(
    const std::vector<uint32_t>& current,
    const std::vector<uint32_t>& incoming,
    std::vector<uint32_t>& merged
) {
    merged = current;
    for (const uint32_t axis : incoming) {
        if (!append_unique_axis(merged, axis)) {
            return false;
        }
    }
    std::sort(merged.begin(), merged.end());
    return true;
}

void append_path_to_rt_score_group(
    const GafimeDecisionPathScoreBatch* paths,
    uint32_t path_idx,
    const std::vector<uint32_t>& merged_axes,
    RtScoreGroup& group
) {
    group.axes = merged_axes;
    group.original_paths.push_back(path_idx);
    const uint32_t begin = paths->path_offsets[path_idx];
    const uint32_t end = paths->path_offsets[path_idx + 1u];
    group.terms.insert(group.terms.end(), paths->terms + begin, paths->terms + end);
    group.offsets.push_back(static_cast<uint32_t>(group.terms.size()));
}

int build_rt_score_groups(
    const GafimeDecisionPathScoreBatch* paths,
    std::vector<RtScoreGroup>& groups
) {
    groups.clear();
    groups.reserve(paths->path_count);
    std::vector<uint32_t> path_axes;
    std::vector<uint32_t> merged_axes;
    const bool prefer_direct_pair_groups = rt_score_direct_stats_requested();
    for (uint32_t path_idx = 0; path_idx < paths->path_count; ++path_idx) {
        if (!collect_path_axes(paths, path_idx, path_axes)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        bool placed = false;
        for (RtScoreGroup& group : groups) {
            if (prefer_direct_pair_groups &&
                (group.axes.size() == 2u || path_axes.size() == 2u)) {
                if (group.axes.size() == 2u && path_axes.size() == 2u &&
                    group.axes == path_axes) {
                    append_path_to_rt_score_group(paths, path_idx, group.axes, group);
                    placed = true;
                    break;
                }
                continue;
            }
            if (merge_rt_axes(group.axes, path_axes, merged_axes)) {
                append_path_to_rt_score_group(paths, path_idx, merged_axes, group);
                placed = true;
                break;
            }
        }
        if (!placed) {
            RtScoreGroup group;
            append_path_to_rt_score_group(paths, path_idx, path_axes, group);
            groups.push_back(std::move(group));
        }
    }
    return groups.empty() ? GAFIME_STATUS_UNSUPPORTED_BACKEND : GAFIME_STATUS_OK;
}

OptixAabb make_rt_conservative_aabb(
    const gafime_cuda_v1::rt_kernel::GafimeRtBox& box
) {
    const float lo_x = static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(box.lo_x));
    const float lo_y = static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(box.lo_y));
    const float hi_x = static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(box.hi_x));
    const float hi_y = static_cast<float>(gafime_cuda_v1::rt_kernel::rt_float_bucket(box.hi_y));
    return {lo_x - 1.0f, lo_y - 1.0f, -0.5f, hi_x + 1.0f, hi_y + 1.0f, 0.5f};
}

void build_rt_aabbs(
    const RtBoxPlan& plan,
    std::vector<OptixAabb>& aabbs
) {
    aabbs.clear();
    aabbs.reserve(plan.boxes.size());
    for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
        aabbs.push_back(make_rt_conservative_aabb(box));
    }
}

float expand_rt_triangle_bound(float value, bool upper) {
    float out = value;
    const float direction = upper ? FLT_MAX : -FLT_MAX;
    for (uint32_t step = 0; step < 8u; ++step) {
        out = std::nextafter(out, direction);
    }
    return out;
}

void build_rt_triangles(
    const RtBoxPlan& plan,
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex>& vertices,
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex>& indices
) {
    vertices.clear();
    indices.clear();
    vertices.reserve(plan.boxes.size() * 4u);
    indices.reserve(plan.boxes.size() * 2u);
    for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
        const uint32_t base = static_cast<uint32_t>(vertices.size());
        const float lo_x = expand_rt_triangle_bound(box.lo_x, false);
        const float lo_y = expand_rt_triangle_bound(box.lo_y, false);
        const float hi_x = expand_rt_triangle_bound(box.hi_x, true);
        const float hi_y = expand_rt_triangle_bound(box.hi_y, true);
        vertices.push_back({lo_x, lo_y, 0.0f});
        vertices.push_back({hi_x, lo_y, 0.0f});
        vertices.push_back({hi_x, hi_y, 0.0f});
        vertices.push_back({lo_x, hi_y, 0.0f});
        indices.push_back({base, base + 1u, base + 2u});
        indices.push_back({base, base + 2u, base + 3u});
    }
}

uint64_t rt_hash_mix(uint64_t hash, uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash;
}

uint64_t rt_hash_float(uint64_t hash, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return rt_hash_mix(hash, bits);
}

uint64_t rt_u32_vector_signature(const std::vector<uint32_t>& values) {
    uint64_t hash = 0xcbf29ce484222325ull;
    hash = rt_hash_mix(hash, static_cast<uint64_t>(values.size()));
    for (const uint32_t value : values) {
        hash = rt_hash_mix(hash, value);
    }
    return hash;
}

uint64_t rt_score_batch_signature(const GafimeDecisionPathScoreBatch* paths) {
    uint64_t hash = 0xcbf29ce484222325ull;
    hash = rt_hash_mix(hash, gafime_cuda_v1::rt_kernel::kRtFloatEncodingVersion);
    hash = rt_hash_mix(hash, paths->abi_version);
    hash = rt_hash_mix(hash, paths->path_count);
    hash = rt_hash_mix(hash, paths->term_count);
    hash = rt_hash_mix(hash, paths->flags);
    for (uint32_t path_idx = 0; path_idx <= paths->path_count; ++path_idx) {
        hash = rt_hash_mix(hash, paths->path_offsets[path_idx]);
    }
    for (uint32_t term_idx = 0; term_idx < paths->term_count; ++term_idx) {
        const GafimeDecisionPathTerm& term = paths->terms[term_idx];
        hash = rt_hash_mix(hash, term.feature);
        hash = rt_hash_mix(hash, term.sign);
        hash = rt_hash_float(hash, term.threshold);
    }
    return hash;
}

uint64_t rt_plan_signature(const RtBoxPlan& plan, RtGeometryMode geometry_mode) {
    uint64_t hash = 0xcbf29ce484222325ull;
    hash = rt_hash_mix(hash, gafime_cuda_v1::rt_kernel::kRtFloatEncodingVersion);
    hash = rt_hash_mix(hash, static_cast<uint32_t>(geometry_mode));
    hash = rt_hash_mix(hash, plan.dims);
    for (uint32_t axis = 0; axis < 3u; ++axis) {
        hash = rt_hash_mix(hash, plan.axes[axis]);
    }
    hash = rt_hash_mix(hash, static_cast<uint64_t>(plan.boxes.size()));
    for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
        hash = rt_hash_float(hash, box.lo_x);
        hash = rt_hash_float(hash, box.lo_y);
        hash = rt_hash_float(hash, box.lo_z);
        hash = rt_hash_float(hash, box.hi_x);
        hash = rt_hash_float(hash, box.hi_y);
        hash = rt_hash_float(hash, box.hi_z);
        hash = rt_hash_mix(hash, box.open_lo_mask);
        hash = rt_hash_mix(hash, box.dims);
    }
    return hash;
}

uint64_t rt_instanced_group_signature(
    const std::vector<RtBoxPlan>& group_plans,
    uint32_t path_count,
    RtGeometryMode geometry_mode
) {
    uint64_t hash = 0xcbf29ce484222325ull;
    hash = rt_hash_mix(hash, gafime_cuda_v1::rt_kernel::kRtFloatEncodingVersion);
    hash = rt_hash_mix(hash, static_cast<uint32_t>(geometry_mode));
    hash = rt_hash_mix(hash, path_count);
    hash = rt_hash_mix(hash, static_cast<uint64_t>(group_plans.size()));
    for (const RtBoxPlan& plan : group_plans) {
        hash = rt_hash_mix(hash, plan.dims);
        for (uint32_t axis = 0; axis < 3u; ++axis) {
            hash = rt_hash_mix(hash, plan.axes[axis]);
        }
        hash = rt_hash_mix(hash, static_cast<uint64_t>(plan.boxes.size()));
        for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
            hash = rt_hash_float(hash, box.lo_x);
            hash = rt_hash_float(hash, box.lo_y);
            hash = rt_hash_float(hash, box.lo_z);
            hash = rt_hash_float(hash, box.hi_x);
            hash = rt_hash_float(hash, box.hi_y);
            hash = rt_hash_float(hash, box.hi_z);
            hash = rt_hash_mix(hash, box.open_lo_mask);
            hash = rt_hash_mix(hash, box.dims);
        }
    }
    return hash;
}

struct RtGroupedScorePlan {
    std::vector<RtScoreGroup> groups;
    std::vector<RtBoxPlan> group_plans;
    std::vector<uint32_t> group_path_offsets;
    std::vector<uint32_t> group_axes;
    std::vector<uint32_t> group_dims;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtBox> flat_boxes;
    std::vector<uint32_t> group_original_path_offsets;
    std::vector<uint32_t> flattened_original_paths;
    uint64_t original_paths_signature = 0;
    uint64_t instanced_geometry_signature = 0;
    bool all_instanced_triangle2d = false;
    bool all_groups_non_overlapping_2d = false;
};

int build_rt_grouped_score_plan(
    const GafimeDecisionPathScoreBatch* paths,
    RtGroupedScorePlan& plan
) {
    plan = {};
    int status = build_rt_score_groups(paths, plan.groups);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (plan.groups.empty()) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    plan.group_original_path_offsets.reserve(plan.groups.size());
    plan.flattened_original_paths.reserve(paths->path_count);
    for (const RtScoreGroup& group : plan.groups) {
        plan.group_original_path_offsets.push_back(static_cast<uint32_t>(plan.flattened_original_paths.size()));
        plan.flattened_original_paths.insert(
            plan.flattened_original_paths.end(),
            group.original_paths.begin(),
            group.original_paths.end()
        );
    }
    if (plan.flattened_original_paths.size() != paths->path_count) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    plan.original_paths_signature = rt_u32_vector_signature(plan.flattened_original_paths);

    plan.group_plans.resize(plan.groups.size());
    plan.group_path_offsets.assign(plan.groups.size() + 1u, 0u);
    plan.group_axes.assign(plan.groups.size() * 3u, 0u);
    plan.group_dims.assign(plan.groups.size(), 0u);
    plan.flat_boxes.clear();
    plan.all_instanced_triangle2d = true;
    plan.all_groups_non_overlapping_2d = rt_score_first_hit_direct_requested();

    uint32_t flat_path_count = 0u;
    for (size_t group_idx = 0; group_idx < plan.groups.size(); ++group_idx) {
        const RtScoreGroup& group = plan.groups[group_idx];
        if (group.original_paths.empty()) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        GafimeDecisionPathScoreBatch group_batch = {};
        group_batch.abi_version = GAFIME_ABI_VERSION;
        group_batch.path_count = static_cast<uint32_t>(group.original_paths.size());
        group_batch.term_count = static_cast<uint32_t>(group.terms.size());
        group_batch.flags = paths->flags;
        group_batch.terms = group.terms.data();
        group_batch.path_offsets = group.offsets.data();
        group_batch.metric_ids = paths->metric_ids;
        group_batch.metric_count = paths->metric_count;

        RtBoxPlan& group_plan = plan.group_plans[group_idx];
        status = build_rt_box_plan(&group_batch, group_plan);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        if (!rt_box_plan_triangle2d_is_safe(group_plan)) {
            plan.all_instanced_triangle2d = false;
        }
        if (plan.all_groups_non_overlapping_2d && !rt_box_plan_non_overlapping_2d(group_plan)) {
            plan.all_groups_non_overlapping_2d = false;
        }
        plan.group_path_offsets[group_idx] = flat_path_count;
        plan.group_dims[group_idx] = group_plan.dims;
        for (uint32_t axis_idx = 0u; axis_idx < 3u; ++axis_idx) {
            plan.group_axes[group_idx * 3u + axis_idx] = group_plan.axes[axis_idx];
        }
        flat_path_count += group_batch.path_count;
        plan.flat_boxes.insert(plan.flat_boxes.end(), group_plan.boxes.begin(), group_plan.boxes.end());
    }
    plan.group_path_offsets[plan.groups.size()] = flat_path_count;
    if (flat_path_count != paths->path_count || plan.flat_boxes.size() != paths->path_count) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    const RtGeometryMode geometry_mode = plan.all_instanced_triangle2d
        ? RtGeometryMode::Triangle2dInstanced
        : RtGeometryMode::CustomAabbInstanced;
    plan.instanced_geometry_signature = rt_instanced_group_signature(
        plan.group_plans,
        paths->path_count,
        geometry_mode
    );
    return GAFIME_STATUS_OK;
}

size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

#endif

int execute_decision_path_membership_sm(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* paths
) {
    const uint64_t output_count = rows * static_cast<uint64_t>(paths->path_count);
    const size_t term_bytes = static_cast<size_t>(paths->term_count) * sizeof(GafimeDecisionPathTerm);
    const size_t offset_bytes = (static_cast<size_t>(paths->path_count) + 1u) * sizeof(uint32_t);
    const size_t output_bytes = static_cast<size_t>(output_count) * sizeof(float);

    GafimeDecisionPathTerm* terms_device = nullptr;
    uint32_t* offsets_device = nullptr;
    float* membership_device = nullptr;

    int status = cuda_status(cudaMalloc(&terms_device, term_bytes));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&offsets_device, offset_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&membership_device, output_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(terms_device, paths->terms, term_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(offsets_device, paths->path_offsets, offset_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(gafime_cuda_v1::launch_decision_path_membership(
            resident_features,
            rows,
            cols,
            terms_device,
            offsets_device,
            paths->path_count,
            membership_device,
            0
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaDeviceSynchronize());
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(paths->membership_host, membership_device, output_bytes, cudaMemcpyDeviceToHost));
    }

    cudaFree(membership_device);
    cudaFree(offsets_device);
    cudaFree(terms_device);
    return status;
}

int write_decision_path_score_rows_host(
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result,
    const std::vector<float>& metric_values
) {
    for (uint64_t row = 0; row < paths->path_count; ++row) {
        for (uint32_t slot = 0; slot < result->max_arity; ++slot) {
            result->combo_indices[row * result->max_arity + slot] =
                slot == 0u ? static_cast<uint32_t>(row) : UINT32_MAX;
        }
        for (uint32_t metric_idx = 0; metric_idx < result->metric_count; ++metric_idx) {
            const float value = metric_idx < paths->metric_count
                ? metric_values[row * paths->metric_count + metric_idx]
                : 0.0f;
            result->metric_values[row * result->metric_count + metric_idx] = value;
        }
        result->ranks[row] = static_cast<uint32_t>(row);
        result->families[row] = GAFIME_FAMILY_DECISION_PATH;
        result->candidate_ids[row] = row;
        result->row_flags[row] = 0;
    }
    result->row_count = paths->path_count;
    return GAFIME_STATUS_OK;
}

int write_decision_path_score_metadata_host(
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
) {
    for (uint64_t row = 0; row < paths->path_count; ++row) {
        for (uint32_t slot = 0; slot < result->max_arity; ++slot) {
            result->combo_indices[row * result->max_arity + slot] =
                slot == 0u ? static_cast<uint32_t>(row) : UINT32_MAX;
        }
        result->ranks[row] = static_cast<uint32_t>(row);
        result->families[row] = GAFIME_FAMILY_DECISION_PATH;
        result->candidate_ids[row] = row;
        result->row_flags[row] = 0;
    }
    result->row_count = paths->path_count;
    return GAFIME_STATUS_OK;
}

int execute_decision_path_score_sm(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
) {
    const uint32_t words_per_path = static_cast<uint32_t>((rows + 31u) / 32u);
    const uint64_t word_count = static_cast<uint64_t>(paths->path_count) * words_per_path;
    const uint64_t metric_value_count = static_cast<uint64_t>(paths->path_count) * paths->metric_count;
    const size_t term_bytes = static_cast<size_t>(paths->term_count) * sizeof(GafimeDecisionPathTerm);
    const size_t offset_bytes = (static_cast<size_t>(paths->path_count) + 1u) * sizeof(uint32_t);
    const size_t mask_bytes = static_cast<size_t>(word_count) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(paths->metric_count) * sizeof(uint32_t);
    const size_t metric_value_bytes = static_cast<size_t>(metric_value_count) * sizeof(float);

    GafimeDecisionPathTerm* terms_device = nullptr;
    uint32_t* offsets_device = nullptr;
    uint32_t* metric_ids_device = nullptr;
    uint32_t* mask_device = nullptr;
    double* target_stats_device = nullptr;
    float* metric_values_device = nullptr;

    int status = cuda_status(cudaMalloc(&terms_device, term_bytes));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&offsets_device, offset_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&metric_ids_device, metric_id_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&mask_device, mask_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(mask_device, 0, mask_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&metric_values_device, metric_value_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&target_stats_device, 3u * sizeof(double)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(terms_device, paths->terms, term_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(offsets_device, paths->path_offsets, offset_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(metric_ids_device, paths->metric_ids, metric_id_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        uint32_t max_grid_y = 0;
        status = cuda_status(current_device_max_grid_y(&max_grid_y));
        const uint64_t tile_count = gafime_cuda_v1::detail::decision_path_row_tile_count(
            rows,
            max_grid_y
        );
        for (uint64_t tile_idx = 0; status == GAFIME_STATUS_OK && tile_idx < tile_count; ++tile_idx) {
            const gafime_cuda_v1::detail::DecisionPathRowTile tile =
                gafime_cuda_v1::detail::decision_path_row_tile(rows, max_grid_y, tile_idx);
            const dim3 grid(paths->path_count, tile.block_count);
            gafime_cuda_v1::rt_kernel::decision_path_bitset_kernel<<<
                grid,
                gafime_cuda_v1::detail::kDecisionPathThreads
            >>>(
                resident_features,
                rows,
                tile.row_offset,
                cols,
                terms_device,
                offsets_device,
                paths->path_count,
                words_per_path,
                mask_device
            );
            status = cuda_status(cudaGetLastError());
        }
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        gafime_cuda_v1::rt_kernel::decision_path_target_stats_kernel<<<1, threads>>>(
            target,
            rows,
            target_stats_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        gafime_cuda_v1::rt_kernel::score_decision_path_bitset_kernel<<<paths->path_count, threads>>>(
            mask_device,
            target,
            target_stats_device,
            rows,
            paths->path_count,
            words_per_path,
            metric_ids_device,
            paths->metric_count,
            metric_values_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaDeviceSynchronize());
    }
    std::vector<float> metric_values(static_cast<size_t>(metric_value_count), 0.0f);
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(metric_values.data(), metric_values_device, metric_value_bytes, cudaMemcpyDeviceToHost));
    }
    if (status == GAFIME_STATUS_OK) {
        status = write_decision_path_score_rows_host(paths, result, metric_values);
    }

    cudaFree(target_stats_device);
    cudaFree(metric_values_device);
    cudaFree(mask_device);
    cudaFree(metric_ids_device);
    cudaFree(offsets_device);
    cudaFree(terms_device);
    return status;
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)

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

struct EmptySbtData {};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using EmptyRecord = SbtRecord<EmptySbtData>;

int optix_status(OptixResult status) {
    return status == OPTIX_SUCCESS ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

struct RtOptixProgram {
    uint32_t device_id = UINT32_MAX;
    uint32_t rtcore_version = 0u;
    RtGeometryMode geometry_mode = RtGeometryMode::CustomAabb;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup program_groups[3]{};
    OptixPipeline pipeline = nullptr;
    EmptyRecord* raygen_record = nullptr;
    EmptyRecord* miss_record = nullptr;
    EmptyRecord* hitgroup_record = nullptr;
    OptixShaderBindingTable sbt{};
    float* points_device = nullptr;
    gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes_device = nullptr;
    float* membership_device = nullptr;
    uint32_t* membership_words_device = nullptr;
    uint32_t* direct_inside_counts_device = nullptr;
    double* direct_inside_sum_y_device = nullptr;
    double* direct_target_stats_device = nullptr;
    uint32_t* metric_ids_device = nullptr;
    float* score_values_device = nullptr;
    OptixAabb* aabbs_device = nullptr;
    gafime_cuda_v1::rt_kernel::GafimeRtTriVertex* vertices_device = nullptr;
    gafime_cuda_v1::rt_kernel::GafimeRtTriIndex* indices_device = nullptr;
    void* gas_temp_device = nullptr;
    void* gas_output_device = nullptr;
    OptixInstance* instances_device = nullptr;
    uint32_t* group_path_offsets_device = nullptr;
    uint32_t* group_axes_device = nullptr;
    uint32_t* group_dims_device = nullptr;
    float* grouped_final_metric_values_device = nullptr;
    uint32_t* grouped_original_paths_device = nullptr;
    void* ias_temp_device = nullptr;
    void* ias_output_device = nullptr;
    GafimeRtParams* params_device = nullptr;
    cudaStream_t stream = nullptr;
    size_t points_capacity = 0;
    size_t box_capacity = 0;
    size_t membership_capacity = 0;
    size_t membership_word_capacity = 0;
    size_t direct_inside_count_capacity = 0;
    size_t direct_inside_sum_y_capacity = 0;
    size_t direct_target_stats_capacity = 0;
    size_t metric_id_capacity = 0;
    size_t score_value_capacity = 0;
    size_t aabb_capacity = 0;
    size_t vertex_capacity = 0;
    size_t index_capacity = 0;
    size_t params_capacity = 0;
    size_t gas_temp_capacity = 0;
    size_t gas_output_capacity = 0;
    size_t instance_capacity = 0;
    size_t group_path_offset_capacity = 0;
    size_t group_axis_capacity = 0;
    size_t group_dim_capacity = 0;
    size_t grouped_final_metric_value_capacity = 0;
    size_t grouped_original_path_capacity = 0;
    size_t ias_temp_capacity = 0;
    size_t ias_output_capacity = 0;
    OptixTraversableHandle gas_handle = 0;
    uint64_t gas_signature = 0;
    bool gas_valid = false;
    bool packed_points_valid = false;
    const float* packed_points_features = nullptr;
    uint64_t packed_points_rows = 0;
    uint64_t packed_points_generation = 0;
    uint64_t packed_points_signature = 0;
    uint32_t packed_points_group_count = 0;
    bool grouped_original_paths_valid = false;
    uint64_t grouped_original_paths_signature = 0;
    size_t grouped_original_paths_count = 0;
    bool grouped_score_plan_valid = false;
    uint64_t grouped_score_plan_signature = 0;
    RtGroupedScorePlan grouped_score_plan;
    bool target_stats_valid = false;
    const float* target_stats_target = nullptr;
    uint64_t target_stats_rows = 0;
    uint64_t target_stats_generation = 0;

    RtOptixProgram() = default;
    RtOptixProgram(const RtOptixProgram&) = delete;
    RtOptixProgram& operator=(const RtOptixProgram&) = delete;
    ~RtOptixProgram() = default;

    void reset() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        cudaFree(params_device);
        cudaFree(gas_output_device);
        cudaFree(gas_temp_device);
        cudaFree(ias_output_device);
        cudaFree(ias_temp_device);
        cudaFree(group_dims_device);
        cudaFree(group_axes_device);
        cudaFree(grouped_original_paths_device);
        cudaFree(grouped_final_metric_values_device);
        cudaFree(group_path_offsets_device);
        cudaFree(instances_device);
        cudaFree(indices_device);
        cudaFree(vertices_device);
        cudaFree(aabbs_device);
        cudaFree(score_values_device);
        cudaFree(metric_ids_device);
        cudaFree(direct_target_stats_device);
        cudaFree(direct_inside_sum_y_device);
        cudaFree(direct_inside_counts_device);
        cudaFree(membership_words_device);
        cudaFree(membership_device);
        cudaFree(boxes_device);
        cudaFree(points_device);
        params_device = nullptr;
        gas_output_device = nullptr;
        gas_temp_device = nullptr;
        ias_output_device = nullptr;
        ias_temp_device = nullptr;
        group_dims_device = nullptr;
        group_axes_device = nullptr;
        grouped_original_paths_device = nullptr;
        grouped_final_metric_values_device = nullptr;
        group_path_offsets_device = nullptr;
        instances_device = nullptr;
        indices_device = nullptr;
        vertices_device = nullptr;
        aabbs_device = nullptr;
        score_values_device = nullptr;
        metric_ids_device = nullptr;
        direct_target_stats_device = nullptr;
        direct_inside_sum_y_device = nullptr;
        direct_inside_counts_device = nullptr;
        membership_words_device = nullptr;
        membership_device = nullptr;
        boxes_device = nullptr;
        points_device = nullptr;
        points_capacity = 0;
        box_capacity = 0;
        membership_capacity = 0;
        membership_word_capacity = 0;
        metric_id_capacity = 0;
        score_value_capacity = 0;
        direct_inside_count_capacity = 0;
        direct_inside_sum_y_capacity = 0;
        direct_target_stats_capacity = 0;
        aabb_capacity = 0;
        vertex_capacity = 0;
        index_capacity = 0;
        params_capacity = 0;
        gas_temp_capacity = 0;
        gas_output_capacity = 0;
        instance_capacity = 0;
        group_path_offset_capacity = 0;
        group_axis_capacity = 0;
        group_dim_capacity = 0;
        grouped_final_metric_value_capacity = 0;
        grouped_original_path_capacity = 0;
        ias_temp_capacity = 0;
        ias_output_capacity = 0;
        gas_handle = 0;
        gas_signature = 0;
        gas_valid = false;
        packed_points_valid = false;
        packed_points_features = nullptr;
        packed_points_rows = 0;
        packed_points_generation = 0;
        packed_points_signature = 0;
        packed_points_group_count = 0;
        grouped_original_paths_valid = false;
        grouped_original_paths_signature = 0;
        grouped_original_paths_count = 0;
        grouped_score_plan_valid = false;
        grouped_score_plan_signature = 0;
        grouped_score_plan = {};
        target_stats_valid = false;
        target_stats_target = nullptr;
        target_stats_rows = 0;
        target_stats_generation = 0;
        cudaFree(hitgroup_record);
        cudaFree(miss_record);
        cudaFree(raygen_record);
        hitgroup_record = nullptr;
        miss_record = nullptr;
        raygen_record = nullptr;
        if (pipeline != nullptr) {
            optixPipelineDestroy(pipeline);
            pipeline = nullptr;
        }
        for (OptixProgramGroup& program_group : program_groups) {
            if (program_group != nullptr) {
                optixProgramGroupDestroy(program_group);
                program_group = nullptr;
            }
        }
        if (module != nullptr) {
            optixModuleDestroy(module);
            module = nullptr;
        }
        if (context != nullptr) {
            optixDeviceContextDestroy(context);
            context = nullptr;
        }
        sbt = {};
        device_id = UINT32_MAX;
        rtcore_version = 0u;
        geometry_mode = RtGeometryMode::CustomAabb;
    }

    bool ready(uint32_t wanted_device_id, RtGeometryMode wanted_geometry_mode) const {
        return context != nullptr && pipeline != nullptr && rtcore_version != 0u &&
            device_id == wanted_device_id &&
            geometry_mode == wanted_geometry_mode;
    }

    bool sbt_ready() const {
        return raygen_record != nullptr && miss_record != nullptr && hitgroup_record != nullptr &&
            sbt.raygenRecord != 0u && sbt.missRecordBase != 0u &&
            sbt.hitgroupRecordBase != 0u;
    }
};

size_t rt_geometry_mode_index(RtGeometryMode mode) {
    switch (mode) {
        case RtGeometryMode::Triangle2dInstanced:
            return 2u;
        case RtGeometryMode::CustomAabbInstanced:
            return 1u;
        case RtGeometryMode::CustomAabb:
        default:
            return 0u;
    }
}

struct RtDeviceState {
    explicit RtDeviceState(uint32_t state_device_id) : device_id(state_device_id) {}

    RtDeviceState(const RtDeviceState&) = delete;
    RtDeviceState& operator=(const RtDeviceState&) = delete;

    ~RtDeviceState() {
        ScopedCudaDevice device(device_id);
        if (device.status() == cudaSuccess) {
            reset();
        }
    }

    RtOptixProgram& program(RtGeometryMode mode) {
        return programs[rt_geometry_mode_index(mode)];
    }

    void reset() {
        for (RtOptixProgram& program_state : programs) {
            program_state.reset();
        }
        cudaFree(feature_domain_invalid_device);
        feature_domain_invalid_device = nullptr;
        feature_domain_valid = false;
        feature_domain_features = nullptr;
        feature_domain_rows = 0;
        feature_domain_cols = 0;
        feature_domain_generation = 0;
        feature_domain_representable = false;
    }

    uint32_t device_id;
    std::mutex execution_mutex;
    std::atomic<bool> retired{false};
    std::array<RtOptixProgram, 3> programs;
    uint32_t* feature_domain_invalid_device = nullptr;
    bool feature_domain_valid = false;
    const float* feature_domain_features = nullptr;
    uint64_t feature_domain_rows = 0;
    uint32_t feature_domain_cols = 0;
    uint64_t feature_domain_generation = 0;
    bool feature_domain_representable = false;
};

gafime_cuda_v1::detail::DeviceStateMap<RtDeviceState>& rt_device_states() {
    // CUDA/OptiX may tear down their process globals before C++ static
    // destructors in this DSO. Native state is therefore released explicitly
    // by the lifecycle ABI; keeping only the registry object alive avoids
    // calling OptiX destroy functions after libnvoptix has begun shutdown.
    static auto* states = new gafime_cuda_v1::detail::DeviceStateMap<RtDeviceState>();
    return *states;
}

std::shared_ptr<RtDeviceState> acquire_rt_device_state(uint32_t device_id) {
    return rt_device_states().get_or_create(
        device_id,
        [](uint32_t id) { return std::make_shared<RtDeviceState>(id); }
    );
}

struct RtDeviceStateLease {
    std::shared_ptr<RtDeviceState> state;
    std::unique_lock<std::mutex> execution_lock;
};

RtDeviceStateLease acquire_rt_device_state_lease(uint32_t device_id) {
    for (;;) {
        std::shared_ptr<RtDeviceState> state = acquire_rt_device_state(device_id);
        std::unique_lock<std::mutex> execution_lock(state->execution_mutex);
        if (!state->retired.load(std::memory_order_acquire)) {
            return {std::move(state), std::move(execution_lock)};
        }
    }
}

int validate_rt_feature_domain(
    RtDeviceState& state,
    const float* features,
    uint64_t rows,
    uint32_t cols,
    uint64_t feature_generation,
    bool* representable_out
) {
    if (features == nullptr || representable_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (state.feature_domain_valid &&
        state.feature_domain_features == features &&
        state.feature_domain_rows == rows &&
        state.feature_domain_cols == cols &&
        state.feature_domain_generation == feature_generation) {
        *representable_out = state.feature_domain_representable;
        return GAFIME_STATUS_OK;
    }

    uint64_t value_count = 0;
    if (!checked_mul_u64(rows, cols, &value_count)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (state.feature_domain_invalid_device == nullptr) {
        const int allocation_status = cuda_status(cudaMalloc(
            reinterpret_cast<void**>(&state.feature_domain_invalid_device),
            sizeof(uint32_t)
        ));
        if (allocation_status != GAFIME_STATUS_OK) {
            return allocation_status;
        }
    }
    int status = cuda_status(cudaMemset(
        state.feature_domain_invalid_device,
        0,
        sizeof(uint32_t)
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (value_count != 0u) {
        constexpr uint32_t threads = 256u;
        const uint64_t requested_blocks = (value_count - 1u) / threads + 1u;
        const uint32_t blocks = static_cast<uint32_t>(
            std::min<uint64_t>(requested_blocks, 65'535u)
        );
        gafime_cuda_v1::rt_kernel::validate_rt_feature_domain_kernel<<<blocks, threads>>>(
            features,
            value_count,
            state.feature_domain_invalid_device
        );
        status = cuda_status(cudaGetLastError());
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    uint32_t invalid = 0u;
    status = cuda_status(cudaMemcpy(
        &invalid,
        state.feature_domain_invalid_device,
        sizeof(invalid),
        cudaMemcpyDeviceToHost
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    state.feature_domain_valid = true;
    state.feature_domain_features = features;
    state.feature_domain_rows = rows;
    state.feature_domain_cols = cols;
    state.feature_domain_generation = feature_generation;
    state.feature_domain_representable = invalid == 0u;
    *representable_out = state.feature_domain_representable;
    return GAFIME_STATUS_OK;
}

int release_rt_device_state(uint32_t device_id) {
    return rt_device_states().release(device_id, [device_id](RtDeviceState& state) -> int {
        std::lock_guard<std::mutex> execution_guard(state.execution_mutex);
        ScopedCudaDevice device(device_id);
        if (device.status() != cudaSuccess) {
            return cuda_status(device.status());
        }
        state.retired.store(true, std::memory_order_release);
        state.reset();
        return GAFIME_STATUS_OK;
    });
}

void invalidate_instanced_execution_caches(RtOptixProgram& program) {
    program.gas_valid = false;
    program.gas_handle = 0;
    program.packed_points_valid = false;
    program.grouped_original_paths_valid = false;
    program.target_stats_valid = false;
}

/* Pipeline/module/context setup is opaque vendor state and intentionally
 * outside the compact-query allocation budget.  The SBT records are CUDA
 * allocations owned by the query, so planning can defer them until the
 * complete explicit allocation plan has passed admission. */
int ensure_optix_sbt(RtOptixProgram* program) {
    if (program == nullptr || program->pipeline == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (program->sbt_ready()) return GAFIME_STATUS_OK;

    EmptyRecord raygen_record = {};
    EmptyRecord miss_record = {};
    EmptyRecord hitgroup_record = {};
    int status = optix_status(optixSbtRecordPackHeader(program->program_groups[0], &raygen_record));
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixSbtRecordPackHeader(program->program_groups[1], &miss_record));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixSbtRecordPackHeader(program->program_groups[2], &hitgroup_record));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program->raygen_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program->miss_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program->hitgroup_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            program->raygen_record, &raygen_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            program->miss_record, &miss_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            program->hitgroup_record, &hitgroup_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) {
        program->reset();
        return status;
    }

    program->sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(program->raygen_record);
    program->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(program->miss_record);
    program->sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    program->sbt.missRecordCount = 1u;
    program->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(program->hitgroup_record);
    program->sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    program->sbt.hitgroupRecordCount = 1u;
    return GAFIME_STATUS_OK;
}

int ensure_optix_program(
    RtDeviceState& state,
    RtGeometryMode geometry_mode,
    bool require_sbt = true
) {
    RtOptixProgram& program = state.program(geometry_mode);
    const uint32_t device_id = state.device_id;
    if (program.ready(device_id, geometry_mode)) {
        return require_sbt ? ensure_optix_sbt(&program) : GAFIME_STATUS_OK;
    }
    program.reset();

    if (cudaFree(nullptr) != cudaSuccess) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    static std::once_flag optix_init_once;
    static OptixResult optix_init_status = OPTIX_ERROR_INTERNAL_ERROR;
    std::call_once(optix_init_once, [] { optix_init_status = optixInit(); });
    if (optix_init_status != OPTIX_SUCCESS) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    CUcontext cu_ctx = nullptr;
    if (cuCtxGetCurrent(&cu_ctx) != CUDA_SUCCESS) {
        cu_ctx = nullptr;
    }

    OptixDeviceContextOptions context_options = {};
    int status = optix_status(optixDeviceContextCreate(cu_ctx, &context_options, &program.context));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    unsigned int rtcore_version = 0u;
    status = optix_status(optixDeviceContextGetProperty(
        program.context,
        OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
        &rtcore_version,
        sizeof(rtcore_version)
    ));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }
    if (rtcore_version == 0u) {
        program.reset();
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    program.rtcore_version = rtcore_version;

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    const bool instanced_mode = geometry_mode != RtGeometryMode::CustomAabb;
    const bool triangle_mode = geometry_mode == RtGeometryMode::Triangle2dInstanced;
    pipeline_options.traversableGraphFlags = instanced_mode
        ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 1;
    pipeline_options.numAttributeValues = triangle_mode ? 0u : 1u;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = triangle_mode
        ? OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE
        : OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[4096];
    size_t log_size = sizeof(log);
    status = optix_status(optixModuleCreate(
        program.context,
        &module_options,
        &pipeline_options,
        gafime_cuda_v1::kRtOptixPtx,
        gafime_cuda_v1::kRtOptixPtxSize,
        log,
        &log_size,
        &program.module
    ));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_descs[3] = {};
    pg_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_descs[0].raygen.module = program.module;
    pg_descs[0].raygen.entryFunctionName = "__raygen__gafime_dp";
    pg_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_descs[1].miss.module = program.module;
    pg_descs[1].miss.entryFunctionName = "__miss__gafime_dp";
    pg_descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (!triangle_mode) {
        pg_descs[2].hitgroup.moduleIS = program.module;
        pg_descs[2].hitgroup.entryFunctionNameIS = "__intersection__gafime_dp_box";
    }
    pg_descs[2].hitgroup.moduleAH = program.module;
    pg_descs[2].hitgroup.entryFunctionNameAH = "__anyhit__gafime_dp_mark";

    log_size = sizeof(log);
    status = optix_status(optixProgramGroupCreate(
        program.context,
        pg_descs,
        3,
        &pg_options,
        log,
        &log_size,
        program.program_groups
    ));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    status = optix_status(optixPipelineCreate(
        program.context,
        &pipeline_options,
        &link_options,
        program.program_groups,
        3,
        log,
        &log_size,
        &program.pipeline
    ));
    if (status == GAFIME_STATUS_OK) {
        const uint32_t max_traversable_depth = instanced_mode ? 2u : 1u;
        status = optix_status(optixPipelineSetStackSize(program.pipeline, 0, 0, 0, max_traversable_depth));
    }
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    program.device_id = device_id;
    program.geometry_mode = geometry_mode;
    return require_sbt ? ensure_optix_sbt(&program) : GAFIME_STATUS_OK;
}

/* The local semantic experiment does not extend the normative semantic ABI
 * table.  It translates only already-validated physical frozen-region terms
 * into the existing exact custom-AABB machinery, then scatters membership
 * directly into fresh slots of the same bank. */
struct SemanticRtBatchShape {
    uint32_t region_count = 0u;
    uint64_t term_count = 0u;
    std::array<uint32_t, GAFIME_CUDA_RT_SEMANTIC_MAX_AXES> axes{0u, 0u, 0u};
    uint32_t axis_count = 0u;
};

struct SemanticRtRegionPlan {
    std::vector<GafimeDecisionPathTerm> terms;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> output_slots;
    RtBoxPlan boxes;
    std::vector<OptixAabb> aabbs;
};

class ScopedRtDeviceAllocation {
public:
    ScopedRtDeviceAllocation() = default;
    ScopedRtDeviceAllocation(const ScopedRtDeviceAllocation&) = delete;
    ScopedRtDeviceAllocation& operator=(const ScopedRtDeviceAllocation&) = delete;

    ~ScopedRtDeviceAllocation() {
        if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
    }

    int allocate(uint64_t bytes) {
        if (bytes == 0u) return GAFIME_STATUS_OK;
        if (bytes > static_cast<uint64_t>(SIZE_MAX)) return GAFIME_STATUS_OUT_OF_MEMORY;
        void* next = nullptr;
        const int status = cuda_status(cudaMalloc(&next, static_cast<size_t>(bytes)));
        if (status != GAFIME_STATUS_OK) return status;
        if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
        ptr_ = next;
        return GAFIME_STATUS_OK;
    }

    template <typename T>
    T* as() const {
        return static_cast<T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
};

int inspect_semantic_rt_batch_shape(
    const gafime_cuda_v1::detail::CudaSemanticBankView& bank,
    const GafimeSemanticProgramBatch* batch,
    SemanticRtBatchShape* shape_out
) {
    if (batch == nullptr || shape_out == nullptr || bank.initialized_slots == nullptr ||
        bank.rows == 0u || bank.rows > GAFIME_CUDA_RT_SEMANTIC_MAX_ROWS ||
        batch->node_count == 0u || batch->node_count > GAFIME_CUDA_RT_SEMANTIC_MAX_REGIONS) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    SemanticRtBatchShape shape{};
    shape.region_count = batch->node_count;
    std::array<uint32_t, GAFIME_CUDA_RT_SEMANTIC_MAX_REGIONS> output_slots{};
    for (uint32_t node_index = 0; node_index < batch->node_count; ++node_index) {
        output_slots[node_index] = batch->nodes[node_index].output_slot;
    }
    for (uint32_t node_index = 0; node_index < batch->node_count; ++node_index) {
        const GafimeSemanticProgramNode& node = batch->nodes[node_index];
        if (node.opcode != GAFIME_SEMANTIC_PROGRAM_FROZEN_REGION_CONJUNCTION ||
            node.region_term_count == 0u ||
            node.region_term_count > gafime_semantic_abi::kSemanticMaxRegionTerms) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        uint64_t next_term_count = 0;
        if (!checked_add_u64(shape.term_count, node.region_term_count, &next_term_count)) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        shape.term_count = next_term_count;
        for (uint32_t term_offset = 0; term_offset < node.region_term_count; ++term_offset) {
            const GafimeSemanticFrozenRegionTerm& source =
                batch->region_terms.ptr[node.region_term_offset + term_offset];
            const uint32_t threshold_bits = static_cast<uint32_t>(source.threshold_bits);
            float threshold = 0.0f;
            std::memcpy(&threshold, &threshold_bits, sizeof(threshold));
            if (!std::isfinite(threshold) || std::fpclassify(threshold) == FP_SUBNORMAL) {
                return GAFIME_STATUS_UNSUPPORTED_BACKEND;
            }
            // Generic materialization permits a topological later node to
            // consume an earlier output.  This local entry deliberately
            // launches every region in parallel, so it accepts only an
            // input-closed region run; Rust splits dependencies into ordered
            // calls before reaching this narrow lowering.
            for (uint32_t output_index = 0; output_index < batch->node_count; ++output_index) {
                if (source.input_slot == output_slots[output_index]) {
                    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
                }
            }
            bool seen = false;
            for (uint32_t axis_index = 0; axis_index < shape.axis_count; ++axis_index) {
                seen = seen || shape.axes[axis_index] == source.input_slot;
            }
            if (!seen) {
                if (shape.axis_count == GAFIME_CUDA_RT_SEMANTIC_MAX_AXES) {
                    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
                }
                shape.axes[shape.axis_count++] = source.input_slot;
            }
        }
    }
    if (shape.axis_count == 0u) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    *shape_out = shape;
    return GAFIME_STATUS_OK;
}

bool semantic_rt_host_plan_bytes(
    const gafime_cuda_v1::detail::CudaSemanticBankView& bank,
    const SemanticRtBatchShape& shape,
    uint64_t* host_peak_out
) {
    if (host_peak_out == nullptr) return false;
    uint64_t term_bytes = 0;
    uint64_t offset_bytes = 0;
    uint64_t output_slot_bytes = 0;
    uint64_t box_bytes = 0;
    uint64_t aabb_bytes = 0;
    uint64_t axis_bytes = 0;
    if (!checked_element_bytes(shape.term_count, sizeof(GafimeDecisionPathTerm), &term_bytes) ||
        !checked_element_bytes(static_cast<uint64_t>(shape.region_count) + 1u, sizeof(uint32_t), &offset_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(uint32_t), &output_slot_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox), &box_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(OptixAabb), &aabb_bytes) ||
        !checked_element_bytes(shape.axis_count, sizeof(uint32_t), &axis_bytes)) {
        return false;
    }
    uint64_t plan_bytes = 0;
    if (!checked_plan_add(&plan_bytes, term_bytes) ||
        !checked_plan_add(&plan_bytes, offset_bytes) ||
        !checked_plan_add(&plan_bytes, output_slot_bytes) ||
        !checked_plan_add(&plan_bytes, box_bytes) ||
        !checked_plan_add(&plan_bytes, aabb_bytes) ||
        !checked_plan_add(&plan_bytes, axis_bytes)) {
        return false;
    }
    // The common physical validator makes this one initialized-slot copy
    // before this plan is allocated, so peaks do not overlap.  Charge the
    // larger exact element payload rather than hiding that check's allocation.
    *host_peak_out = std::max(plan_bytes, static_cast<uint64_t>(bank.slot_capacity));
    return true;
}

bool semantic_rt_device_plan_bytes(
    const gafime_cuda_v1::detail::CudaSemanticBankView& bank,
    const SemanticRtBatchShape& shape,
    const OptixAccelBufferSizes& gas_sizes,
    uint64_t* device_peak_out
) {
    if (device_peak_out == nullptr) return false;
    const uint64_t gas_temp_bytes = static_cast<uint64_t>(gas_sizes.tempSizeInBytes);
    const uint64_t gas_output_bytes = static_cast<uint64_t>(gas_sizes.outputSizeInBytes);
    if (static_cast<size_t>(gas_temp_bytes) != gas_sizes.tempSizeInBytes ||
        static_cast<size_t>(gas_output_bytes) != gas_sizes.outputSizeInBytes) {
        return false;
    }
    uint64_t point_count = 0;
    uint64_t membership_count = 0;
    uint64_t point_bytes = 0;
    uint64_t box_bytes = 0;
    uint64_t membership_bytes = 0;
    uint64_t aabb_bytes = 0;
    uint64_t params_bytes = 0;
    uint64_t sbt_bytes = 0;
    uint64_t axis_bytes = 0;
    uint64_t output_slot_bytes = 0;
    uint64_t invalid_bytes = 0;
    if (!checked_mul_u64(bank.rows, 3u, &point_count) ||
        !checked_mul_u64(bank.rows, shape.region_count, &membership_count) ||
        !checked_element_bytes(point_count, sizeof(float), &point_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox), &box_bytes) ||
        !checked_element_bytes(membership_count, sizeof(float), &membership_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(OptixAabb), &aabb_bytes) ||
        !checked_element_bytes(1u, sizeof(GafimeRtParams), &params_bytes) ||
        !checked_element_bytes(3u, sizeof(EmptyRecord), &sbt_bytes) ||
        !checked_element_bytes(shape.axis_count, sizeof(uint32_t), &axis_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(uint32_t), &output_slot_bytes) ||
        !checked_element_bytes(1u, sizeof(uint32_t), &invalid_bytes)) {
        return false;
    }
    uint64_t total = 0;
    if (!checked_plan_add(&total, point_bytes) ||
        !checked_plan_add(&total, box_bytes) ||
        !checked_plan_add(&total, membership_bytes) ||
        !checked_plan_add(&total, aabb_bytes) ||
        !checked_plan_add(&total, params_bytes) ||
        !checked_plan_add(&total, sbt_bytes) ||
        !checked_plan_add(&total, axis_bytes) ||
        !checked_plan_add(&total, output_slot_bytes) ||
        !checked_plan_add(&total, invalid_bytes) ||
        !checked_plan_add(&total, gas_temp_bytes) ||
        !checked_plan_add(&total, gas_output_bytes)) {
        return false;
    }
    *device_peak_out = total;
    return true;
}

int build_semantic_rt_region_plan(
    const GafimeSemanticProgramBatch* batch,
    const SemanticRtBatchShape& shape,
    SemanticRtRegionPlan* plan_out
) {
    if (batch == nullptr || plan_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    SemanticRtRegionPlan plan{};
    if (shape.term_count > static_cast<uint64_t>(SIZE_MAX) ||
        shape.region_count == 0u) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    plan.terms.reserve(static_cast<size_t>(shape.term_count));
    plan.offsets.reserve(static_cast<size_t>(shape.region_count) + 1u);
    plan.output_slots.reserve(shape.region_count);
    plan.offsets.push_back(0u);
    for (uint32_t node_index = 0; node_index < shape.region_count; ++node_index) {
        const GafimeSemanticProgramNode& node = batch->nodes[node_index];
        for (uint32_t term_offset = 0; term_offset < node.region_term_count; ++term_offset) {
            const GafimeSemanticFrozenRegionTerm& source =
                batch->region_terms.ptr[node.region_term_offset + term_offset];
            float threshold = 0.0f;
            const uint32_t threshold_bits = static_cast<uint32_t>(source.threshold_bits);
            std::memcpy(&threshold, &threshold_bits, sizeof(threshold));
            plan.terms.push_back({
                source.input_slot,
                source.relation == GAFIME_SEMANTIC_REGION_LESS_EQUAL
                    ? GAFIME_DECISION_PATH_SIGN_LE
                    : GAFIME_DECISION_PATH_SIGN_GT,
                threshold,
                0u,
                {0u, 0u},
            });
        }
        if (plan.terms.size() > static_cast<size_t>(UINT32_MAX)) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        plan.offsets.push_back(static_cast<uint32_t>(plan.terms.size()));
        plan.output_slots.push_back(node.output_slot);
    }
    const int status = build_rt_box_plan(
        shape.region_count,
        static_cast<uint32_t>(plan.terms.size()),
        plan.terms.data(),
        plan.offsets.data(),
        plan.boxes
    );
    if (status != GAFIME_STATUS_OK) return status;
    build_rt_aabbs(plan.boxes, plan.aabbs);
    if (plan.aabbs.size() != shape.region_count || plan.boxes.dims != shape.axis_count) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    *plan_out = std::move(plan);
    return GAFIME_STATUS_OK;
}
int execute_semantic_region_materialize_rt_optix(
    GafimeGpuSemanticBank bank_handle,
    const GafimeSemanticProgramBatch* batch,
    uint64_t max_temporary_bytes,
    uint64_t* peak_bytes_out
) {
    if (peak_bytes_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *peak_bytes_out = 0u;
    if (batch == nullptr || !gafime_gpu_abi::naturally_aligned(batch) ||
        !gafime_semantic_abi::abi_compatible(
            batch->abi_version,
            batch->struct_size,
            gafime_semantic_abi::kProgramBatchV13StablePrefixSize)) {
        return batch != nullptr && gafime_gpu_abi::naturally_aligned(batch)
            ? GAFIME_STATUS_ABI_MISMATCH
            : GAFIME_STATUS_INVALID_ARGUMENT;
    }

    gafime_cuda_v1::detail::CudaSemanticBankView bank{};
    int status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(bank_handle, &bank);
    if (status != GAFIME_STATUS_OK) return status;
    if (!gafime_gpu_abi::route_fields_equal(batch->route, bank.route)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    // The shared validator deliberately copies the initialized-slot mask to
    // validate generic intra-batch dependencies.  `ensure_optix_program`
    // then allocates three fixed-size SBT records before the variable-size AS
    // query.  Admit the larger non-overlapping prequery requirement before
    // either allocation; an early rejection reports that minimum rather than
    // pretending it is the later full OptiX plan peak.
    uint64_t fixed_sbt_bytes = 0u;
    if (!checked_element_bytes(3u, sizeof(EmptyRecord), &fixed_sbt_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t prequery_minimum = std::max(
        static_cast<uint64_t>(bank.slot_capacity), fixed_sbt_bytes);
    if (prequery_minimum > max_temporary_bytes) {
        *peak_bytes_out = prequery_minimum;
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }

    // Use the shared physical validator before touching any v1.3 descriptor
    // array.  It preserves the standard slot/topology/fresh-output invariant
    // and rejects old semantic-minor consumers before descriptor-stride use.
    status = gafime_semantic_abi::validate_program_batch(
        batch,
        GAFIME_PRECISION_FP32,
        bank.source_slots,
        bank.slot_capacity,
        *bank.initialized_slots,
        gafime_semantic_abi::kSemanticMaxRegionTerms
    );
    if (status != GAFIME_STATUS_OK) return status;

    SemanticRtBatchShape shape{};
    status = inspect_semantic_rt_batch_shape(bank, batch, &shape);
    if (status != GAFIME_STATUS_OK) return status;

    uint64_t host_peak = 0u;
    if (!semantic_rt_host_plan_bytes(bank, shape, &host_peak)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }

    ScopedCudaDevice device(bank.device_id);
    status = cuda_status(device.status());
    if (status != GAFIME_STATUS_OK) return status;
    // A stack-owned state deliberately avoids the legacy per-device cache:
    // this first proof measures only cold, call-local resources and cannot
    // retain bank-derived geometry beyond this synchronous entrypoint.
    RtDeviceState state(bank.device_id);
    // This helper creates exactly three fixed-size SBT records.  Those explicit
    // CUDA buffers are included in `semantic_rt_device_plan_bytes`; only the
    // opaque driver/context/pipeline allocations remain outside the local
    // explicit-buffer budget.  No variable-size AS/workspace allocation occurs
    // until after the exact query and max_temporary_bytes admission below.
    status = ensure_optix_program(state, RtGeometryMode::CustomAabb);
    if (status != GAFIME_STATUS_OK) return status;
    RtOptixProgram& program = state.program(RtGeometryMode::CustomAabb);

    // Memory-size discovery consumes only build metadata, never AABB contents.
    // The zero CUdeviceptr is therefore valid for this query and lets an
    // over-budget request fail before any variable-size device allocation.
    CUdeviceptr aabb_buffer = 0;
    uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives = shape.region_count;
    build_input.customPrimitiveArray.flags = geometry_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1u;
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes gas_sizes = {};
    status = optix_status(optixAccelComputeMemoryUsage(
        program.context,
        &accel_options,
        &build_input,
        1u,
        &gas_sizes
    ));
    if (status != GAFIME_STATUS_OK) return status;

    uint64_t device_peak = 0u;
    uint64_t peak = 0u;
    if (!semantic_rt_device_plan_bytes(bank, shape, gas_sizes, &device_peak) ||
        !checked_add_u64(host_peak, device_peak, &peak)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    *peak_bytes_out = peak;
    if (peak > max_temporary_bytes) return GAFIME_STATUS_OUT_OF_MEMORY;

    SemanticRtRegionPlan plan{};
    status = build_semantic_rt_region_plan(batch, shape, &plan);
    if (status != GAFIME_STATUS_OK) {
        *peak_bytes_out = 0u;
        return status;
    }

    uint64_t point_count = 0u;
    uint64_t membership_count = 0u;
    if (!checked_mul_u64(bank.rows, 3u, &point_count) ||
        !checked_mul_u64(bank.rows, shape.region_count, &membership_count) ||
        point_count > static_cast<uint64_t>(SIZE_MAX) ||
        membership_count > static_cast<uint64_t>(SIZE_MAX)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = ensure_device_capacity(
        &program.points_device, program.points_capacity, static_cast<size_t>(point_count));
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(
            &program.boxes_device, program.box_capacity, static_cast<size_t>(shape.region_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(
            &program.membership_device, program.membership_capacity,
            static_cast<size_t>(membership_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.aabbs_device, program.aabb_capacity, plan.aabbs.size());
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.params_device, program.params_capacity, size_t{1u});
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(
            &program.gas_temp_device, program.gas_temp_capacity, gas_sizes.tempSizeInBytes);
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(
            &program.gas_output_device, program.gas_output_capacity, gas_sizes.outputSizeInBytes);
    }
    if (status == GAFIME_STATUS_OK && program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
    }
    if (status != GAFIME_STATUS_OK) return status;

    ScopedRtDeviceAllocation axes_device;
    ScopedRtDeviceAllocation output_slots_device;
    ScopedRtDeviceAllocation invalid_device;
    uint64_t axis_bytes = 0u;
    uint64_t output_slot_bytes = 0u;
    if (!checked_element_bytes(shape.axis_count, sizeof(uint32_t), &axis_bytes) ||
        !checked_element_bytes(shape.region_count, sizeof(uint32_t), &output_slot_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    status = axes_device.allocate(axis_bytes);
    if (status == GAFIME_STATUS_OK) status = output_slots_device.allocate(output_slot_bytes);
    if (status == GAFIME_STATUS_OK) status = invalid_device.allocate(sizeof(uint32_t));
    if (status != GAFIME_STATUS_OK) return status;

    status = cuda_status(cudaMemcpyAsync(
        axes_device.as<uint32_t>(), plan.boxes.axes.data(), static_cast<size_t>(axis_bytes),
        cudaMemcpyHostToDevice, program.stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(
            output_slots_device.as<uint32_t>(), plan.output_slots.data(),
            static_cast<size_t>(output_slot_bytes), cudaMemcpyHostToDevice, program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemsetAsync(
            invalid_device.as<uint32_t>(), 0, sizeof(uint32_t), program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t kThreads = 256u;
        const uint64_t value_count = bank.rows * static_cast<uint64_t>(shape.axis_count);
        const uint32_t blocks = static_cast<uint32_t>((value_count + kThreads - 1u) / kThreads);
        gafime_cuda_v1::rt_kernel::validate_semantic_region_input_domain_kernel<<<
            blocks, kThreads, 0, program.stream
        >>>(
            bank.columns,
            bank.rows,
            axes_device.as<uint32_t>(),
            shape.axis_count,
            invalid_device.as<uint32_t>()
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    uint32_t invalid = 0u;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            &invalid, invalid_device.as<uint32_t>(), sizeof(invalid), cudaMemcpyDeviceToHost));
    }
    if (status != GAFIME_STATUS_OK) return status;
    if (invalid != 0u) {
        *peak_bytes_out = 0u;
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    status = cuda_status(cudaMemcpyAsync(
        program.boxes_device,
        plan.boxes.boxes.data(),
        plan.boxes.boxes.size() * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox),
        cudaMemcpyHostToDevice,
        program.stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(
            program.aabbs_device,
            plan.aabbs.data(),
            plan.aabbs.size() * sizeof(OptixAabb),
            cudaMemcpyHostToDevice,
            program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemsetAsync(
            program.membership_device,
            0,
            static_cast<size_t>(membership_count) * sizeof(float),
            program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t kThreads = 256u;
        const uint32_t blocks = static_cast<uint32_t>((bank.rows + kThreads - 1u) / kThreads);
        gafime_cuda_v1::rt_kernel::pack_decision_path_points_kernel<<<
            blocks, kThreads, 0, program.stream
        >>>(
            bank.columns,
            bank.rows,
            plan.boxes.axes[0],
            plan.boxes.axes[1],
            plan.boxes.axes[2],
            plan.boxes.dims,
            program.points_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status != GAFIME_STATUS_OK) return status;

    aabb_buffer = reinterpret_cast<CUdeviceptr>(program.aabbs_device);
    status = optix_status(optixAccelBuild(
        program.context,
        program.stream,
        &accel_options,
        &build_input,
        1u,
        reinterpret_cast<CUdeviceptr>(program.gas_temp_device),
        gas_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(program.gas_output_device),
        gas_sizes.outputSizeInBytes,
        &program.gas_handle,
        nullptr,
        0u));
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    if (status != GAFIME_STATUS_OK) return status;

    GafimeRtParams params = {};
    params.handle = program.gas_handle;
    params.points_xyz = program.points_device;
    params.boxes = program.boxes_device;
    params.membership = program.membership_device;
    params.rows = static_cast<uint32_t>(bank.rows);
    params.path_count = shape.region_count;
    params.geometry_mode = static_cast<uint32_t>(RtGeometryMode::CustomAabb);
    params.group_count = 1u;
    params.point_stride = 3u;
    params.direct_first_hit = 0u;
    status = cuda_status(cudaMemcpyAsync(
        program.params_device, &params, sizeof(params), cudaMemcpyHostToDevice, program.stream));
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixLaunch(
            program.pipeline,
            program.stream,
            reinterpret_cast<CUdeviceptr>(program.params_device),
            sizeof(GafimeRtParams),
            &program.sbt,
            static_cast<uint32_t>(bank.rows),
            1u,
            1u));
    }
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    if (status != GAFIME_STATUS_OK) return status;

    status = cuda_status(cudaMemsetAsync(
        invalid_device.as<uint32_t>(), 0, sizeof(uint32_t), program.stream));
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t kThreads = 256u;
        const uint32_t blocks = static_cast<uint32_t>((membership_count + kThreads - 1u) / kThreads);
        gafime_cuda_v1::rt_kernel::scatter_semantic_region_membership_kernel<<<
            blocks, kThreads, 0, program.stream
        >>>(
            program.membership_device,
            bank.rows,
            shape.region_count,
            output_slots_device.as<uint32_t>(),
            bank.columns,
            invalid_device.as<uint32_t>()
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            &invalid, invalid_device.as<uint32_t>(), sizeof(invalid), cudaMemcpyDeviceToHost));
    }
    if (status != GAFIME_STATUS_OK) return status;
    if (invalid != 0u) return GAFIME_STATUS_DEVICE_ERROR;

    return gafime_cuda_v1::detail::commit_cuda_semantic_bank_outputs(
        bank_handle, plan.output_slots.data(), shape.region_count);
}

int execute_decision_path_membership_optix(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint64_t feature_generation,
    const GafimeDecisionPathBatch* paths
) {
    static_cast<void>(arch_class);
    if (rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    RtDeviceStateLease state_lease = acquire_rt_device_state_lease(device_id);
    RtDeviceState& state = *state_lease.state;
    bool features_are_representable = false;
    int status = validate_rt_feature_domain(
        state,
        resident_features,
        rows,
        cols,
        feature_generation,
        &features_are_representable
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (!features_are_representable) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    RtBoxPlan plan;
    status = build_rt_box_plan(paths, plan);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const RtGeometryMode geometry_mode = choose_rt_geometry_mode(plan);
    status = ensure_optix_program(state, geometry_mode);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    RtOptixProgram& program = state.program(geometry_mode);

    const uint64_t output_count = rows * static_cast<uint64_t>(paths->path_count);
    const size_t box_bytes = static_cast<size_t>(paths->path_count) * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox);
    const size_t output_bytes = static_cast<size_t>(output_count) * sizeof(float);
    const size_t point_count = static_cast<size_t>(rows) * 3u;

    status = ensure_device_capacity(&program.points_device, program.points_capacity, point_count);
    if (status == GAFIME_STATUS_OK) {
        if (static_cast<size_t>(paths->path_count) > program.box_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.boxes_device, program.box_capacity, static_cast<size_t>(paths->path_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.membership_device, program.membership_capacity, static_cast<size_t>(output_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.params_device, program.params_capacity, static_cast<size_t>(1u));
    }
    if (status == GAFIME_STATUS_OK && program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
    }
    const uint64_t geometry_signature = rt_plan_signature(plan, geometry_mode);
    const bool rebuild_gas = !program.gas_valid || program.gas_signature != geometry_signature;
    if (rebuild_gas) {
        program.gas_valid = false;
        program.gas_handle = 0;
    }
    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        status = cuda_status(cudaMemcpy(program.boxes_device, plan.boxes.data(), box_bytes, cudaMemcpyHostToDevice));
    }
    std::vector<OptixAabb> aabbs;
    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        build_rt_aabbs(plan, aabbs);
        if (aabbs.size() > program.aabb_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.aabbs_device, program.aabb_capacity, aabbs.size());
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(
                program.aabbs_device,
                aabbs.data(),
                aabbs.size() * sizeof(OptixAabb),
                cudaMemcpyHostToDevice
            ));
        }
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemsetAsync(program.membership_device, 0, output_bytes, program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        const uint32_t row_blocks = static_cast<uint32_t>((rows + threads - 1) / threads);
        gafime_cuda_v1::rt_kernel::pack_decision_path_points_kernel<<<row_blocks, threads, 0, program.stream>>>(
            resident_features,
            rows,
            plan.axes[0],
            plan.axes[1],
            plan.axes[2],
            plan.dims,
            program.points_device
        );
        status = cuda_status(cudaGetLastError());
    }

    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(program.aabbs_device);
        uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = paths->path_count;
        build_input.customPrimitiveArray.flags = geometry_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes gas_sizes = {};
        status = optix_status(optixAccelComputeMemoryUsage(
            program.context,
            &accel_options,
            &build_input,
            1,
            &gas_sizes
        ));
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.gas_temp_device, program.gas_temp_capacity, gas_sizes.tempSizeInBytes);
        }
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.gas_output_device, program.gas_output_capacity, gas_sizes.outputSizeInBytes);
        }
        if (status == GAFIME_STATUS_OK) {
            status = optix_status(optixAccelBuild(
                program.context,
                program.stream,
                &accel_options,
                &build_input,
                1,
                reinterpret_cast<CUdeviceptr>(program.gas_temp_device),
                gas_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(program.gas_output_device),
                gas_sizes.outputSizeInBytes,
                &program.gas_handle,
                nullptr,
                0
            ));
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaStreamSynchronize(program.stream));
        }
        if (status == GAFIME_STATUS_OK) {
            program.gas_signature = geometry_signature;
            program.gas_valid = true;
        } else {
            program.gas_valid = false;
        }
    }

    GafimeRtParams params = {};
    params.handle = program.gas_handle;
    params.points_xyz = program.points_device;
    params.boxes = program.boxes_device;
    params.target = nullptr;
    params.membership = program.membership_device;
    params.membership_words = nullptr;
    params.direct_inside_counts = nullptr;
    params.direct_inside_sum_y = nullptr;
    params.rows = static_cast<uint32_t>(rows);
    params.path_count = paths->path_count;
    params.geometry_mode = static_cast<uint32_t>(geometry_mode);
    params.words_per_path = 0;
    params.point_stride = 3u;
    params.direct_first_hit = 0u;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(program.params_device, &params, sizeof(params), cudaMemcpyHostToDevice, program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixLaunch(
            program.pipeline,
            program.stream,
            reinterpret_cast<CUdeviceptr>(program.params_device),
            sizeof(GafimeRtParams),
            &program.sbt,
            static_cast<uint32_t>(rows),
            1,
            1
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(paths->membership_host, program.membership_device, output_bytes, cudaMemcpyDeviceToHost));
    }
    if (status != GAFIME_STATUS_OK) {
        program.gas_valid = false;
        program.gas_handle = 0;
    }

    return status;
}

int execute_decision_path_score_optix_planned(
    RtDeviceState& state,
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t device_id,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result,
    const RtBoxPlan& plan,
    const double* precomputed_target_stats_device = nullptr,
    std::vector<float>* metric_values_out = nullptr,
    const uint32_t* scatter_original_paths_device = nullptr,
    float* scatter_metric_values_device = nullptr
) {
    const bool scatter_metrics = scatter_original_paths_device != nullptr || scatter_metric_values_device != nullptr;
    if ((scatter_original_paths_device == nullptr) != (scatter_metric_values_device == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (scatter_metrics && metric_values_out != nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = GAFIME_STATUS_OK;
    const RtGeometryMode geometry_mode = choose_rt_geometry_mode(plan);
    status = ensure_optix_program(state, geometry_mode);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    RtOptixProgram& program = state.program(geometry_mode);
    const bool direct_stats = rt_score_direct_stats_requested();
    const bool direct_first_hit = rt_score_first_hit_direct_requested();
    if (direct_first_hit && !rt_box_plan_non_overlapping_2d(plan)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    // The non-overlap proof above plus terminate-on-first-hit permits at most
    // one accepted callback per ray, so first-hit needs no duplicate bitset.
    const bool needs_duplicate_guard =
        gafime_cuda_v1::detail::decision_path_score_needs_duplicate_guard(direct_first_hit);

    const uint32_t words_per_path = static_cast<uint32_t>((rows + 31u) / 32u);
    const uint64_t word_count = static_cast<uint64_t>(paths->path_count) * words_per_path;
    const uint64_t metric_value_count = static_cast<uint64_t>(paths->path_count) * paths->metric_count;
    const size_t box_bytes = static_cast<size_t>(paths->path_count) * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox);
    const size_t mask_bytes = static_cast<size_t>(word_count) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(paths->metric_count) * sizeof(uint32_t);
    const size_t metric_value_bytes = static_cast<size_t>(metric_value_count) * sizeof(float);
    const size_t direct_target_stats_count = 3u;
    const size_t point_count = static_cast<size_t>(rows) * 3u;

    status = ensure_device_capacity(&program.points_device, program.points_capacity, point_count);
    if (status == GAFIME_STATUS_OK) {
        if (static_cast<size_t>(paths->path_count) > program.box_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.boxes_device, program.box_capacity, static_cast<size_t>(paths->path_count));
    }
    if (status == GAFIME_STATUS_OK && needs_duplicate_guard) {
        status = ensure_device_capacity(&program.membership_words_device, program.membership_word_capacity, static_cast<size_t>(word_count));
    }
    if (status == GAFIME_STATUS_OK && direct_stats) {
        status = ensure_device_capacity(
            &program.direct_inside_counts_device,
            program.direct_inside_count_capacity,
            static_cast<size_t>(paths->path_count)
        );
    }
    if (status == GAFIME_STATUS_OK && direct_stats) {
        status = ensure_device_capacity(
            &program.direct_inside_sum_y_device,
            program.direct_inside_sum_y_capacity,
            static_cast<size_t>(paths->path_count)
        );
    }
    const double* target_stats_device = precomputed_target_stats_device;
    if (status == GAFIME_STATUS_OK && target_stats_device == nullptr) {
        status = ensure_device_capacity(
            &program.direct_target_stats_device,
            program.direct_target_stats_capacity,
            direct_target_stats_count
        );
        if (status == GAFIME_STATUS_OK) {
            target_stats_device = program.direct_target_stats_device;
        }
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.metric_ids_device, program.metric_id_capacity, static_cast<size_t>(paths->metric_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.score_values_device, program.score_value_capacity, static_cast<size_t>(metric_value_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.params_device, program.params_capacity, static_cast<size_t>(1u));
    }
    if (status == GAFIME_STATUS_OK && program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
    }
    const uint64_t geometry_signature = rt_plan_signature(plan, geometry_mode);
    const bool rebuild_gas = !program.gas_valid || program.gas_signature != geometry_signature;
    if (rebuild_gas) {
        program.gas_valid = false;
        program.gas_handle = 0;
    }
    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        status = cuda_status(cudaMemcpy(program.boxes_device, plan.boxes.data(), box_bytes, cudaMemcpyHostToDevice));
    }
    std::vector<OptixAabb> aabbs;
    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        build_rt_aabbs(plan, aabbs);
        if (aabbs.size() > program.aabb_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.aabbs_device, program.aabb_capacity, aabbs.size());
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(
                program.aabbs_device,
                aabbs.data(),
                aabbs.size() * sizeof(OptixAabb),
                cudaMemcpyHostToDevice
            ));
        }
    }
    if (status == GAFIME_STATUS_OK && needs_duplicate_guard) {
        status = cuda_status(cudaMemsetAsync(program.membership_words_device, 0, mask_bytes, program.stream));
    }
    if (status == GAFIME_STATUS_OK && direct_stats) {
        status = cuda_status(cudaMemsetAsync(
            program.direct_inside_counts_device,
            0,
            static_cast<size_t>(paths->path_count) * sizeof(uint32_t),
            program.stream
        ));
    }
    if (status == GAFIME_STATUS_OK && direct_stats) {
        status = cuda_status(cudaMemsetAsync(
            program.direct_inside_sum_y_device,
            0,
            static_cast<size_t>(paths->path_count) * sizeof(double),
            program.stream
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(
            program.metric_ids_device,
            paths->metric_ids,
            metric_id_bytes,
            cudaMemcpyHostToDevice,
            program.stream
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        const uint32_t row_blocks = static_cast<uint32_t>((rows + threads - 1u) / threads);
        gafime_cuda_v1::rt_kernel::pack_decision_path_points_kernel<<<row_blocks, threads, 0, program.stream>>>(
            resident_features,
            rows,
            plan.axes[0],
            plan.axes[1],
            plan.axes[2],
            plan.dims,
            program.points_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK && precomputed_target_stats_device == nullptr) {
        constexpr uint32_t threads = 256;
        gafime_cuda_v1::rt_kernel::decision_path_target_stats_kernel<<<1, threads, 0, program.stream>>>(
            target,
            rows,
            program.direct_target_stats_device
        );
        status = cuda_status(cudaGetLastError());
    }

    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(program.aabbs_device);
        uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = paths->path_count;
        build_input.customPrimitiveArray.flags = geometry_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes gas_sizes = {};
        status = optix_status(optixAccelComputeMemoryUsage(
            program.context,
            &accel_options,
            &build_input,
            1,
            &gas_sizes
        ));
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.gas_temp_device, program.gas_temp_capacity, gas_sizes.tempSizeInBytes);
        }
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.gas_output_device, program.gas_output_capacity, gas_sizes.outputSizeInBytes);
        }
        if (status == GAFIME_STATUS_OK) {
            status = optix_status(optixAccelBuild(
                program.context,
                program.stream,
                &accel_options,
                &build_input,
                1,
                reinterpret_cast<CUdeviceptr>(program.gas_temp_device),
                gas_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(program.gas_output_device),
                gas_sizes.outputSizeInBytes,
                &program.gas_handle,
                nullptr,
                0
            ));
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaStreamSynchronize(program.stream));
        }
        if (status == GAFIME_STATUS_OK) {
            program.gas_signature = geometry_signature;
            program.gas_valid = true;
        } else {
            program.gas_valid = false;
        }
    }

    GafimeRtParams params = {};
    params.handle = program.gas_handle;
    params.points_xyz = program.points_device;
    params.boxes = program.boxes_device;
    params.target = direct_stats ? target : nullptr;
    params.target_stats = target_stats_device;
    params.membership = nullptr;
    params.membership_words = needs_duplicate_guard ? program.membership_words_device : nullptr;
    params.direct_inside_counts = direct_stats ? program.direct_inside_counts_device : nullptr;
    params.direct_inside_sum_y = direct_stats ? program.direct_inside_sum_y_device : nullptr;
    params.rows = static_cast<uint32_t>(rows);
    params.path_count = paths->path_count;
    params.geometry_mode = static_cast<uint32_t>(geometry_mode);
    params.words_per_path = words_per_path;
    params.point_stride = 3u;
    params.direct_first_hit = direct_first_hit ? 1u : 0u;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(program.params_device, &params, sizeof(params), cudaMemcpyHostToDevice, program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixLaunch(
            program.pipeline,
            program.stream,
            reinterpret_cast<CUdeviceptr>(program.params_device),
            sizeof(GafimeRtParams),
            &program.sbt,
            static_cast<uint32_t>(rows),
            1,
            1
        ));
    }
    if (status == GAFIME_STATUS_OK && direct_stats) {
        constexpr uint32_t threads = 256;
        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(paths->path_count) + threads - 1u) / threads
        );
        gafime_cuda_v1::rt_kernel::score_decision_path_direct_stats_kernel<<<blocks, threads, 0, program.stream>>>(
            program.direct_inside_counts_device,
            program.direct_inside_sum_y_device,
            target_stats_device,
            paths->path_count,
            program.metric_ids_device,
            paths->metric_count,
            program.score_values_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK && !direct_stats) {
        constexpr uint32_t threads = 256;
        gafime_cuda_v1::rt_kernel::score_decision_path_bitset_kernel<<<paths->path_count, threads, 0, program.stream>>>(
            program.membership_words_device,
            target,
            target_stats_device,
            rows,
            paths->path_count,
            words_per_path,
            program.metric_ids_device,
            paths->metric_count,
            program.score_values_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK && scatter_metrics) {
        constexpr uint32_t threads = 256;
        const uint32_t blocks = static_cast<uint32_t>((metric_value_count + threads - 1u) / threads);
        gafime_cuda_v1::rt_kernel::scatter_decision_path_score_metrics_kernel<<<blocks, threads, 0, program.stream>>>(
            program.score_values_device,
            scatter_original_paths_device,
            paths->path_count,
            paths->metric_count,
            scatter_metric_values_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(program.stream));
    }
    if (status == GAFIME_STATUS_OK && scatter_metrics) {
        return GAFIME_STATUS_OK;
    }

    std::vector<float> local_metric_values;
    std::vector<float>& metric_values = metric_values_out == nullptr ? local_metric_values : *metric_values_out;
    metric_values.assign(static_cast<size_t>(metric_value_count), 0.0f);
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(metric_values.data(), program.score_values_device, metric_value_bytes, cudaMemcpyDeviceToHost));
    }
    if (status == GAFIME_STATUS_OK && metric_values_out != nullptr) {
        return GAFIME_STATUS_OK;
    }
    if (status == GAFIME_STATUS_OK && result == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (status == GAFIME_STATUS_OK) {
        status = write_decision_path_score_rows_host(paths, result, metric_values);
    }
    if (status != GAFIME_STATUS_OK) {
        program.gas_valid = false;
        program.gas_handle = 0;
    }
    return status;
}

int execute_decision_path_score_optix_grouped_instanced(
    RtDeviceState& state,
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t device_id,
    const GafimeDecisionPathScoreBatch* paths,
    const RtGroupedScorePlan& grouped_plan,
    uint64_t feature_generation,
    const double* precomputed_target_stats_device,
    const uint32_t* flattened_original_paths_device,
    float* final_metric_values_device
) {
    const std::vector<RtScoreGroup>& groups = grouped_plan.groups;
    const bool direct_first_hit = rt_score_first_hit_direct_requested();
    if (!rt_score_direct_stats_requested() || groups.size() <= 1u || rows > UINT32_MAX / 3u) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (direct_first_hit && !grouped_plan.all_groups_non_overlapping_2d) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    uint32_t max_grid_y = 0u;
    int status = cuda_status(current_device_max_grid_y(&max_grid_y));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (!gafime_cuda_v1::detail::decision_path_group_count_fits_grid(
            groups.size(),
            max_grid_y)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    const RtGeometryMode geometry_mode = grouped_plan.all_instanced_triangle2d
        ? RtGeometryMode::Triangle2dInstanced
        : RtGeometryMode::CustomAabbInstanced;
    const bool triangle_mode = geometry_mode == RtGeometryMode::Triangle2dInstanced;
    status = ensure_optix_program(state, geometry_mode);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    RtOptixProgram& program = state.program(geometry_mode);
    if (program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    std::vector<OptixAabb> aabbs;
    std::vector<uint32_t> aabb_offsets(groups.size(), 0u);
    std::vector<uint32_t> aabb_counts(groups.size(), 0u);
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> vertices;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> indices;
    std::vector<uint32_t> vertex_offsets(groups.size(), 0u);
    std::vector<uint32_t> vertex_counts(groups.size(), 0u);
    std::vector<uint32_t> index_offsets(groups.size(), 0u);
    std::vector<uint32_t> index_counts(groups.size(), 0u);

    constexpr uint32_t grouped_point_stride = 3u;
    const size_t point_count = static_cast<size_t>(rows) * groups.size() * grouped_point_stride;
    const size_t direct_stats_count = static_cast<size_t>(paths->path_count);
    const uint32_t words_per_path = static_cast<uint32_t>((rows + 31u) / 32u);
    // First-hit reaches this path only after the non-overlap proof, and OptiX
    // terminates after the first accepted intersection, so no duplicate bitset
    // is needed for that mode.
    const bool needs_duplicate_guard =
        gafime_cuda_v1::detail::decision_path_score_needs_duplicate_guard(direct_first_hit);
    const size_t membership_word_count = needs_duplicate_guard
        ? direct_stats_count * words_per_path
        : 0u;
    const size_t membership_word_bytes = membership_word_count * sizeof(uint32_t);
    const uint64_t geometry_signature = grouped_plan.instanced_geometry_signature;
    bool rebuild_geometry = !program.gas_valid || program.gas_signature != geometry_signature;
    if (rebuild_geometry) {
        program.gas_valid = false;
        program.gas_handle = 0;
    }
    const bool reuse_packed_points =
        program.packed_points_valid &&
        feature_generation != 0u &&
        point_count <= program.points_capacity &&
        program.packed_points_features == resident_features &&
        program.packed_points_rows == rows &&
        program.packed_points_generation == feature_generation &&
        program.packed_points_signature == geometry_signature &&
        program.packed_points_group_count == static_cast<uint32_t>(groups.size());
    if (rebuild_geometry) {
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            if (triangle_mode) {
                std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> group_vertices;
                std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> group_indices;
                build_rt_triangles(
                    grouped_plan.group_plans[group_idx],
                    group_vertices,
                    group_indices
                );
                vertex_offsets[group_idx] = static_cast<uint32_t>(vertices.size());
                vertex_counts[group_idx] = static_cast<uint32_t>(group_vertices.size());
                index_offsets[group_idx] = static_cast<uint32_t>(indices.size());
                index_counts[group_idx] = static_cast<uint32_t>(group_indices.size());
                vertices.insert(vertices.end(), group_vertices.begin(), group_vertices.end());
                indices.insert(indices.end(), group_indices.begin(), group_indices.end());
            } else {
                std::vector<OptixAabb> group_aabbs;
                build_rt_aabbs(grouped_plan.group_plans[group_idx], group_aabbs);
                aabb_offsets[group_idx] = static_cast<uint32_t>(aabbs.size());
                aabb_counts[group_idx] = static_cast<uint32_t>(group_aabbs.size());
                aabbs.insert(aabbs.end(), group_aabbs.begin(), group_aabbs.end());
            }
        }
    }
    if (point_count > program.points_capacity) {
        program.packed_points_valid = false;
    }
    status = ensure_device_capacity(&program.points_device, program.points_capacity, point_count);
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.boxes_device, program.box_capacity, grouped_plan.flat_boxes.size());
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry && !triangle_mode) {
        status = ensure_device_capacity(&program.aabbs_device, program.aabb_capacity, aabbs.size());
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry && triangle_mode) {
        status = ensure_device_capacity(
            &program.vertices_device,
            program.vertex_capacity,
            vertices.size()
        );
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry && triangle_mode) {
        status = ensure_device_capacity(
            &program.indices_device,
            program.index_capacity,
            indices.size()
        );
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.instances_device, program.instance_capacity, groups.size());
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(
            &program.group_path_offsets_device,
            program.group_path_offset_capacity,
            grouped_plan.group_path_offsets.size()
        );
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.group_axes_device, program.group_axis_capacity, grouped_plan.group_axes.size());
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.group_dims_device, program.group_dim_capacity, grouped_plan.group_dims.size());
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.direct_inside_counts_device, program.direct_inside_count_capacity, direct_stats_count);
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.direct_inside_sum_y_device, program.direct_inside_sum_y_capacity, direct_stats_count);
    }
    if (status == GAFIME_STATUS_OK && needs_duplicate_guard) {
        status = ensure_device_capacity(
            &program.membership_words_device,
            program.membership_word_capacity,
            membership_word_count
        );
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.metric_ids_device, program.metric_id_capacity, static_cast<size_t>(paths->metric_count));
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.params_device, program.params_capacity, static_cast<size_t>(1u));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = cuda_status(cudaMemcpy(
            program.boxes_device,
            grouped_plan.flat_boxes.data(),
            grouped_plan.flat_boxes.size() * sizeof(grouped_plan.flat_boxes[0]),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry && !triangle_mode) {
        status = cuda_status(cudaMemcpy(
            program.aabbs_device,
            aabbs.data(),
            aabbs.size() * sizeof(aabbs[0]),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry && triangle_mode) {
        status = cuda_status(cudaMemcpy(
            program.vertices_device,
            vertices.data(),
            vertices.size() * sizeof(vertices[0]),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry && triangle_mode) {
        status = cuda_status(cudaMemcpy(
            program.indices_device,
            indices.data(),
            indices.size() * sizeof(indices[0]),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = cuda_status(cudaMemcpy(
            program.group_path_offsets_device,
            grouped_plan.group_path_offsets.data(),
            grouped_plan.group_path_offsets.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = cuda_status(cudaMemcpy(
            program.group_axes_device,
            grouped_plan.group_axes.data(),
            grouped_plan.group_axes.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = cuda_status(cudaMemcpy(
            program.group_dims_device,
            grouped_plan.group_dims.data(),
            grouped_plan.group_dims.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
    }
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        std::vector<OptixTraversableHandle> group_handles(groups.size(), 0);
        std::vector<size_t> gas_output_offsets(groups.size(), 0u);
        std::vector<OptixAccelBufferSizes> gas_sizes(groups.size());
        uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        size_t max_temp_bytes = 0u;
        size_t total_output_bytes = 0u;
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            CUdeviceptr aabb_buffer = triangle_mode ? 0u : reinterpret_cast<CUdeviceptr>(
                program.aabbs_device + aabb_offsets[group_idx]
            );
            CUdeviceptr vertex_buffer = triangle_mode ? reinterpret_cast<CUdeviceptr>(
                program.vertices_device + vertex_offsets[group_idx]
            ) : 0u;
            const CUdeviceptr index_buffer = triangle_mode ? reinterpret_cast<CUdeviceptr>(
                program.indices_device + index_offsets[group_idx]
            ) : 0u;
            OptixBuildInput build_input = {};
            if (triangle_mode) {
                build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                build_input.triangleArray.vertexBuffers = &vertex_buffer;
                build_input.triangleArray.numVertices = vertex_counts[group_idx];
                build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                build_input.triangleArray.vertexStrideInBytes =
                    sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
                build_input.triangleArray.indexBuffer = index_buffer;
                build_input.triangleArray.numIndexTriplets = index_counts[group_idx];
                build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                build_input.triangleArray.indexStrideInBytes =
                    sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
                build_input.triangleArray.flags = geometry_flags;
                build_input.triangleArray.numSbtRecords = 1;
            } else {
                build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
                build_input.customPrimitiveArray.numPrimitives = aabb_counts[group_idx];
                build_input.customPrimitiveArray.flags = geometry_flags;
                build_input.customPrimitiveArray.numSbtRecords = 1;
            }

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = triangle_mode
                ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                : OPTIX_BUILD_FLAG_NONE;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
            status = optix_status(optixAccelComputeMemoryUsage(
                program.context,
                &accel_options,
                &build_input,
                1,
                &gas_sizes[group_idx]
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
            gas_output_offsets[group_idx] = align_up_size(total_output_bytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
            total_output_bytes = gas_output_offsets[group_idx] + gas_sizes[group_idx].outputSizeInBytes;
            max_temp_bytes = std::max(max_temp_bytes, static_cast<size_t>(gas_sizes[group_idx].tempSizeInBytes));
        }
        status = ensure_device_bytes(&program.gas_temp_device, program.gas_temp_capacity, max_temp_bytes);
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.gas_output_device, program.gas_output_capacity, total_output_bytes);
        }
        if (status != GAFIME_STATUS_OK) {
            return status;
        }

        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            CUdeviceptr aabb_buffer = triangle_mode ? 0u : reinterpret_cast<CUdeviceptr>(
                program.aabbs_device + aabb_offsets[group_idx]
            );
            CUdeviceptr vertex_buffer = triangle_mode ? reinterpret_cast<CUdeviceptr>(
                program.vertices_device + vertex_offsets[group_idx]
            ) : 0u;
            const CUdeviceptr index_buffer = triangle_mode ? reinterpret_cast<CUdeviceptr>(
                program.indices_device + index_offsets[group_idx]
            ) : 0u;
            OptixBuildInput build_input = {};
            if (triangle_mode) {
                build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                build_input.triangleArray.vertexBuffers = &vertex_buffer;
                build_input.triangleArray.numVertices = vertex_counts[group_idx];
                build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                build_input.triangleArray.vertexStrideInBytes =
                    sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
                build_input.triangleArray.indexBuffer = index_buffer;
                build_input.triangleArray.numIndexTriplets = index_counts[group_idx];
                build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                build_input.triangleArray.indexStrideInBytes =
                    sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
                build_input.triangleArray.flags = geometry_flags;
                build_input.triangleArray.numSbtRecords = 1;
            } else {
                build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
                build_input.customPrimitiveArray.numPrimitives = aabb_counts[group_idx];
                build_input.customPrimitiveArray.flags = geometry_flags;
                build_input.customPrimitiveArray.numSbtRecords = 1;
            }

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = triangle_mode
                ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                : OPTIX_BUILD_FLAG_NONE;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
            status = optix_status(optixAccelBuild(
                program.context,
                program.stream,
                &accel_options,
                &build_input,
                1,
                reinterpret_cast<CUdeviceptr>(program.gas_temp_device),
                gas_sizes[group_idx].tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(static_cast<char*>(program.gas_output_device) + gas_output_offsets[group_idx]),
                gas_sizes[group_idx].outputSizeInBytes,
                &group_handles[group_idx],
                nullptr,
                0
            ));
            if (status != GAFIME_STATUS_OK) {
                return status;
            }
        }

        std::vector<OptixInstance> instances(groups.size());
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            OptixInstance instance = {};
            const float z = static_cast<float>(group_idx) * 4.0f;
            const float transform[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, z,
            };
            std::memcpy(instance.transform, transform, sizeof(transform));
            instance.instanceId = static_cast<uint32_t>(group_idx);
            instance.visibilityMask = 1u;
            instance.sbtOffset = 0u;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = group_handles[group_idx];
            instances[group_idx] = instance;
        }
        status = cuda_status(cudaMemcpy(program.instances_device, instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }

        OptixBuildInput ias_input = {};
        ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        ias_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(program.instances_device);
        ias_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());
        OptixAccelBuildOptions ias_options = {};
        ias_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        ias_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes ias_sizes = {};
        status = optix_status(optixAccelComputeMemoryUsage(
            program.context,
            &ias_options,
            &ias_input,
            1,
            &ias_sizes
        ));
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.ias_temp_device, program.ias_temp_capacity, ias_sizes.tempSizeInBytes);
        }
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_bytes(&program.ias_output_device, program.ias_output_capacity, ias_sizes.outputSizeInBytes);
        }
        OptixTraversableHandle ias_handle = 0;
        if (status == GAFIME_STATUS_OK) {
            status = optix_status(optixAccelBuild(
                program.context,
                program.stream,
                &ias_options,
                &ias_input,
                1,
                reinterpret_cast<CUdeviceptr>(program.ias_temp_device),
                ias_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(program.ias_output_device),
                ias_sizes.outputSizeInBytes,
                &ias_handle,
                nullptr,
                0
            ));
        }
        if (status != GAFIME_STATUS_OK) {
            program.gas_valid = false;
            return status;
        }
        program.gas_handle = ias_handle;
        program.gas_signature = geometry_signature;
        program.gas_valid = true;
    }

    if (needs_duplicate_guard) {
        status = cuda_status(cudaMemsetAsync(
            program.membership_words_device,
            0,
            membership_word_bytes,
            program.stream
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemsetAsync(program.direct_inside_counts_device, 0, direct_stats_count * sizeof(uint32_t), program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemsetAsync(program.direct_inside_sum_y_device, 0, direct_stats_count * sizeof(double), program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(
            program.metric_ids_device,
            paths->metric_ids,
            static_cast<size_t>(paths->metric_count) * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            program.stream
        ));
    }
    if (status == GAFIME_STATUS_OK && !reuse_packed_points) {
        constexpr uint32_t threads = 256;
        const uint32_t row_blocks = static_cast<uint32_t>((rows + threads - 1u) / threads);
        dim3 grid(row_blocks, static_cast<uint32_t>(groups.size()));
        gafime_cuda_v1::rt_kernel::pack_grouped_decision_path_points_kernel<<<grid, threads, 0, program.stream>>>(
            resident_features,
            rows,
            program.group_axes_device,
            program.group_dims_device,
            static_cast<uint32_t>(groups.size()),
            grouped_point_stride,
            program.points_device
        );
        status = cuda_status(cudaGetLastError());
        if (status == GAFIME_STATUS_OK) {
            program.packed_points_valid = true;
            program.packed_points_features = resident_features;
            program.packed_points_rows = rows;
            program.packed_points_generation = feature_generation;
            program.packed_points_signature = geometry_signature;
            program.packed_points_group_count = static_cast<uint32_t>(groups.size());
        } else {
            program.packed_points_valid = false;
        }
    }

    GafimeRtParams params = {};
    params.handle = program.gas_handle;
    params.points_xyz = program.points_device;
    params.boxes = program.boxes_device;
    params.target = target;
    params.target_stats = precomputed_target_stats_device;
    params.membership_words = needs_duplicate_guard ? program.membership_words_device : nullptr;
    params.direct_inside_counts = program.direct_inside_counts_device;
    params.direct_inside_sum_y = program.direct_inside_sum_y_device;
    params.rows = static_cast<uint32_t>(rows);
    params.path_count = paths->path_count;
    params.geometry_mode = static_cast<uint32_t>(geometry_mode);
    params.group_path_offsets = program.group_path_offsets_device;
    params.group_count = static_cast<uint32_t>(groups.size());
    params.words_per_path = words_per_path;
    params.point_group_stride = static_cast<uint32_t>(rows * grouped_point_stride);
    params.point_stride = grouped_point_stride;
    params.direct_first_hit = direct_first_hit ? 1u : 0u;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpyAsync(program.params_device, &params, sizeof(params), cudaMemcpyHostToDevice, program.stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixLaunch(
            program.pipeline,
            program.stream,
            reinterpret_cast<CUdeviceptr>(program.params_device),
            sizeof(GafimeRtParams),
            &program.sbt,
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(groups.size()),
            1
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(paths->path_count) + threads - 1u) / threads
        );
        gafime_cuda_v1::rt_kernel::score_decision_path_direct_stats_scatter_kernel<<<blocks, threads, 0, program.stream>>>(
            program.direct_inside_counts_device,
            program.direct_inside_sum_y_device,
            precomputed_target_stats_device,
            flattened_original_paths_device,
            paths->path_count,
            program.metric_ids_device,
            paths->metric_count,
            final_metric_values_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status != GAFIME_STATUS_OK) {
        invalidate_instanced_execution_caches(program);
    }
    return status;
}

int execute_decision_path_score_optix_grouped(
    RtDeviceState& state,
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t device_id,
    uint64_t feature_generation,
    uint64_t target_generation,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
) {
    const bool direct_stats = rt_score_direct_stats_requested();
    int status = GAFIME_STATUS_OK;
    RtOptixProgram* direct_program = nullptr;
    if (direct_stats) {
        status = ensure_optix_program(state, RtGeometryMode::CustomAabbInstanced);
        if (status == GAFIME_STATUS_OK) {
            direct_program = &state.program(RtGeometryMode::CustomAabbInstanced);
        } else if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
            return status;
        }
    }

    uint64_t grouped_plan_signature = rt_score_batch_signature(paths);
    grouped_plan_signature = rt_hash_mix(
        grouped_plan_signature,
        rt_score_first_hit_direct_requested() ? 0xf17a5177u : 0u
    );
    RtGroupedScorePlan local_grouped_plan;
    const RtGroupedScorePlan* grouped_plan = nullptr;
    if (direct_program != nullptr &&
        direct_program->grouped_score_plan_valid &&
        direct_program->grouped_score_plan_signature == grouped_plan_signature) {
        grouped_plan = &direct_program->grouped_score_plan;
    } else {
        status = build_rt_grouped_score_plan(paths, local_grouped_plan);
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        if (direct_program != nullptr) {
            direct_program->grouped_score_plan = std::move(local_grouped_plan);
            direct_program->grouped_score_plan_valid = true;
            direct_program->grouped_score_plan_signature = grouped_plan_signature;
            grouped_plan = &direct_program->grouped_score_plan;
        } else {
            grouped_plan = &local_grouped_plan;
        }
    }
    const std::vector<RtScoreGroup>& groups = grouped_plan->groups;
    if (groups.size() <= 1u) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    const uint64_t final_metric_value_count = static_cast<uint64_t>(paths->path_count) * paths->metric_count;
    const size_t final_metric_value_bytes = static_cast<size_t>(final_metric_value_count) * sizeof(float);
    const std::vector<uint32_t>& group_original_path_offsets = grouped_plan->group_original_path_offsets;
    const std::vector<uint32_t>& flattened_original_paths = grouped_plan->flattened_original_paths;
    const uint64_t original_paths_signature = grouped_plan->original_paths_signature;

    if (direct_stats && direct_program != nullptr) {
        RtOptixProgram* execution_program = direct_program;
        if (grouped_plan->all_instanced_triangle2d) {
            status = ensure_optix_program(state, RtGeometryMode::Triangle2dInstanced);
            if (status == GAFIME_STATUS_OK) {
                execution_program = &state.program(RtGeometryMode::Triangle2dInstanced);
            }
        }
        if (status == GAFIME_STATUS_OK) {
            if (execution_program->stream == nullptr) {
                status = cuda_status(cudaStreamCreate(&execution_program->stream));
            }
            if (status == GAFIME_STATUS_OK) {
                status = ensure_device_capacity(
                    &execution_program->direct_target_stats_device,
                    execution_program->direct_target_stats_capacity,
                    static_cast<size_t>(3u)
                );
            }
            const bool reuse_target_stats =
                status == GAFIME_STATUS_OK &&
                execution_program->target_stats_valid &&
                target_generation != 0u &&
                execution_program->target_stats_target == target &&
                execution_program->target_stats_rows == rows &&
                execution_program->target_stats_generation == target_generation;
            const bool reuse_original_paths =
                status == GAFIME_STATUS_OK &&
                execution_program->grouped_original_paths_valid &&
                flattened_original_paths.size() <= execution_program->grouped_original_path_capacity &&
                execution_program->grouped_original_paths_signature == original_paths_signature &&
                execution_program->grouped_original_paths_count == flattened_original_paths.size();
            if (status == GAFIME_STATUS_OK) {
                status = ensure_device_capacity(
                    &execution_program->grouped_final_metric_values_device,
                    execution_program->grouped_final_metric_value_capacity,
                    static_cast<size_t>(final_metric_value_count)
                );
            }
            if (status == GAFIME_STATUS_OK) {
                status = ensure_device_capacity(
                    &execution_program->grouped_original_paths_device,
                    execution_program->grouped_original_path_capacity,
                    flattened_original_paths.size()
                );
            }
            if (status == GAFIME_STATUS_OK && !reuse_original_paths) {
                status = cuda_status(cudaMemcpyAsync(
                    execution_program->grouped_original_paths_device,
                    flattened_original_paths.data(),
                    flattened_original_paths.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    execution_program->stream
                ));
                if (status == GAFIME_STATUS_OK) {
                    execution_program->grouped_original_paths_valid = true;
                    execution_program->grouped_original_paths_signature = original_paths_signature;
                    execution_program->grouped_original_paths_count = flattened_original_paths.size();
                } else {
                    execution_program->grouped_original_paths_valid = false;
                }
            }
            if (status == GAFIME_STATUS_OK && !reuse_target_stats) {
                constexpr uint32_t threads = 256;
                gafime_cuda_v1::rt_kernel::decision_path_target_stats_kernel<<<1, threads, 0, execution_program->stream>>>(
                    target,
                    rows,
                    execution_program->direct_target_stats_device
                );
                status = cuda_status(cudaGetLastError());
                if (status == GAFIME_STATUS_OK) {
                    execution_program->target_stats_valid = true;
                    execution_program->target_stats_target = target;
                    execution_program->target_stats_rows = rows;
                    execution_program->target_stats_generation = target_generation;
                } else {
                    execution_program->target_stats_valid = false;
                }
            }
            if (status == GAFIME_STATUS_OK) {
                status = execute_decision_path_score_optix_grouped_instanced(
                    state,
                    resident_features,
                    target,
                    rows,
                    device_id,
                    paths,
                    *grouped_plan,
                    feature_generation,
                    execution_program->direct_target_stats_device,
                    execution_program->grouped_original_paths_device,
                    execution_program->grouped_final_metric_values_device
                );
            }
            if (status == GAFIME_STATUS_OK) {
                if (result->metric_count == paths->metric_count) {
                    status = cuda_status(cudaMemcpyAsync(
                        result->metric_values,
                        execution_program->grouped_final_metric_values_device,
                        final_metric_value_bytes,
                        cudaMemcpyDeviceToHost,
                        execution_program->stream
                    ));
                    if (status == GAFIME_STATUS_OK) {
                        status = cuda_status(cudaStreamSynchronize(execution_program->stream));
                    }
                    if (status != GAFIME_STATUS_OK) {
                        invalidate_instanced_execution_caches(*execution_program);
                        return status;
                    }
                    return write_decision_path_score_metadata_host(paths, result);
                }
                std::vector<float> final_metric_values(
                    static_cast<size_t>(final_metric_value_count),
                    0.0f
                );
                status = cuda_status(cudaMemcpyAsync(
                    final_metric_values.data(),
                    execution_program->grouped_final_metric_values_device,
                    final_metric_value_bytes,
                    cudaMemcpyDeviceToHost,
                    execution_program->stream
                ));
                if (status == GAFIME_STATUS_OK) {
                    status = cuda_status(cudaStreamSynchronize(execution_program->stream));
                }
                if (status != GAFIME_STATUS_OK) {
                    invalidate_instanced_execution_caches(*execution_program);
                    return status;
                }
                return write_decision_path_score_rows_host(paths, result, final_metric_values);
            }
        }
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
            invalidate_instanced_execution_caches(*execution_program);
            return status;
        }
    }

    struct ScopedTargetStats {
        double* ptr = nullptr;
        ~ScopedTargetStats() {
            cudaFree(ptr);
        }
    } shared_target_stats;

    struct ScopedGroupedScoreBuffers {
        float* final_metric_values_device = nullptr;
        uint32_t* original_paths_device = nullptr;
        size_t final_metric_value_capacity = 0;
        size_t original_path_capacity = 0;

        ~ScopedGroupedScoreBuffers() {
            cudaFree(original_paths_device);
            cudaFree(final_metric_values_device);
        }
    } grouped_score_buffers;

    if (direct_stats) {
        status = cuda_status(cudaMalloc(reinterpret_cast<void**>(&shared_target_stats.ptr), 3u * sizeof(double)));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
        constexpr uint32_t threads = 256;
        gafime_cuda_v1::rt_kernel::decision_path_target_stats_kernel<<<1, threads>>>(
            target,
            rows,
            shared_target_stats.ptr
        );
        status = cuda_status(cudaGetLastError());
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaDeviceSynchronize());
        }
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    status = ensure_device_capacity(
        &grouped_score_buffers.final_metric_values_device,
        grouped_score_buffers.final_metric_value_capacity,
        static_cast<size_t>(final_metric_value_count)
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_device_capacity(
        &grouped_score_buffers.original_paths_device,
        grouped_score_buffers.original_path_capacity,
        flattened_original_paths.size()
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = cuda_status(cudaMemcpy(
        grouped_score_buffers.original_paths_device,
        flattened_original_paths.data(),
        flattened_original_paths.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
        const RtScoreGroup& group = groups[group_idx];
        GafimeDecisionPathScoreBatch group_batch = {};
        group_batch.abi_version = GAFIME_ABI_VERSION;
        group_batch.path_count = static_cast<uint32_t>(group.original_paths.size());
        group_batch.term_count = static_cast<uint32_t>(group.terms.size());
        group_batch.flags = paths->flags;
        group_batch.terms = group.terms.data();
        group_batch.path_offsets = group.offsets.data();
        group_batch.metric_ids = paths->metric_ids;
        group_batch.metric_count = paths->metric_count;

        status = execute_decision_path_score_optix_planned(
            state,
            resident_features,
            target,
            rows,
            device_id,
            &group_batch,
            nullptr,
            grouped_plan->group_plans[group_idx],
            shared_target_stats.ptr,
            nullptr,
            grouped_score_buffers.original_paths_device + group_original_path_offsets[group_idx],
            grouped_score_buffers.final_metric_values_device
        );
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }
    std::vector<float> final_metric_values(
        static_cast<size_t>(final_metric_value_count),
        0.0f
    );
    status = cuda_status(cudaMemcpy(
        final_metric_values.data(),
        grouped_score_buffers.final_metric_values_device,
        final_metric_value_bytes,
        cudaMemcpyDeviceToHost
    ));
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    return write_decision_path_score_rows_host(paths, result, final_metric_values);
}

int execute_decision_path_score_optix(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint64_t feature_generation,
    uint64_t target_generation,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
) {
    static_cast<void>(arch_class);
    if (rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    RtDeviceStateLease state_lease = acquire_rt_device_state_lease(device_id);
    RtDeviceState& state = *state_lease.state;
    bool features_are_representable = false;
    int status = validate_rt_feature_domain(
        state,
        resident_features,
        rows,
        cols,
        feature_generation,
        &features_are_representable
    );
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    if (!features_are_representable) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    if (rt_score_direct_stats_requested()) {
        const int grouped_status = execute_decision_path_score_optix_grouped(
            state,
            resident_features,
            target,
            rows,
            device_id,
            feature_generation,
            target_generation,
            paths,
            result
        );
        if (grouped_status == GAFIME_STATUS_OK) {
            return grouped_status;
        }
        if (grouped_status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
            return grouped_status;
        }
    }

    RtBoxPlan plan;
    status = build_rt_box_plan(paths, plan);
    if (status == GAFIME_STATUS_OK) {
        return execute_decision_path_score_optix_planned(
            state,
            resident_features,
            target,
            rows,
            device_id,
            paths,
            result,
            plan
        );
    }
    if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
        return status;
    }
    return execute_decision_path_score_optix_grouped(
        state,
        resident_features,
        target,
        rows,
        device_id,
        feature_generation,
        target_generation,
        paths,
        result
    );
}

#else

int execute_decision_path_membership_optix(
    const float*,
    uint64_t,
    uint32_t,
    uint32_t,
    uint64_t,
    uint64_t,
    const GafimeDecisionPathBatch*
) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

int execute_decision_path_score_optix(
    const float*,
    const float*,
    uint64_t,
    uint32_t,
    uint32_t,
    uint64_t,
    uint64_t,
    uint64_t,
    const GafimeDecisionPathScoreBatch*,
    GafimeResultTable*
) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

#endif

constexpr uint64_t kSemanticRtRegionQueryMagic = 0x4741465345515254ull;  // "GAFSEQRT"
constexpr uint32_t kSemanticRtRegionSmBinCount = 256u;

struct SemanticRtAxis {
    uint32_t primary_slot = 0u;
    uint32_t paired_slot = 0u;
};

bool semantic_rt_axis_equal(const SemanticRtAxis& left, const SemanticRtAxis& right) {
    return left.primary_slot == right.primary_slot && left.paired_slot == right.paired_slot;
}

bool semantic_rt_append_axis(std::vector<SemanticRtAxis>& axes, SemanticRtAxis axis) {
    for (const SemanticRtAxis& existing : axes) {
        if (semantic_rt_axis_equal(existing, axis)) return true;
    }
    if (axes.size() >= GAFIME_CUDA_RT_SEMANTIC_MAX_AXES) return false;
    axes.push_back(axis);
    return true;
}

void semantic_rt_sort_axes(std::vector<SemanticRtAxis>& axes) {
    std::sort(axes.begin(), axes.end(), [](const SemanticRtAxis& left, const SemanticRtAxis& right) {
        if (left.primary_slot != right.primary_slot) return left.primary_slot < right.primary_slot;
        return left.paired_slot < right.paired_slot;
    });
}

uint32_t semantic_rt_axis_index(
    const std::vector<SemanticRtAxis>& axes,
    SemanticRtAxis wanted
) {
    for (uint32_t index = 0u; index < axes.size(); ++index) {
        if (semantic_rt_axis_equal(axes[index], wanted)) return index;
    }
    return UINT32_MAX;
}

struct SemanticRtRegionGroupBuild {
    std::vector<uint32_t> result_regions;
    std::vector<SemanticRtAxis> axes;
};

struct SemanticRtRegionGroup {
    std::vector<uint32_t> result_regions;
    std::vector<SemanticRtAxis> axes;
    std::vector<GafimeDecisionPathTerm> exact_terms;
    std::vector<uint32_t> exact_offsets;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtBox> boxes;
    bool all_boxes_bounded = true;
};

struct CompactSemanticRtRegionPlan {
    std::vector<SemanticRtRegionGroup> groups;
    std::vector<uint32_t> group_path_offsets;
    std::vector<uint32_t> group_region_ids;
    std::vector<GafimeDecisionPathTerm> flat_exact_terms;
    std::vector<uint32_t> flat_exact_offsets;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtBox> flat_boxes;
    std::vector<uint32_t> group_primary_axes;
    std::vector<uint32_t> group_paired_axes;
    std::vector<uint32_t> group_dims;
    std::vector<uint32_t> bin_offsets;
    std::vector<uint32_t> bin_candidates;
    std::vector<float> bin_lo;
    std::vector<float> bin_inv_span;
    /* Measured while the host planner's temporary bins/build groups coexist
     * with the retained snapshot.  The query admission charges this peak,
     * rather than pretending the temporary host geometry was free. */
    uint64_t host_build_peak_bytes = 0u;
    bool first_hit_eligible = false;
    bool triangle_eligible = false;
};

template <typename T>
bool semantic_rt_vector_capacity_bytes(const std::vector<T>& values, uint64_t* bytes_out) {
    return bytes_out != nullptr && checked_element_bytes(
        static_cast<uint64_t>(values.capacity()), sizeof(T), bytes_out);
}

bool semantic_rt_group_host_bytes(const SemanticRtRegionGroup& group, uint64_t* total) {
    uint64_t bytes = 0u;
    return semantic_rt_vector_capacity_bytes(group.result_regions, &bytes) &&
        checked_plan_add(total, bytes) &&
        semantic_rt_vector_capacity_bytes(group.axes, &bytes) &&
        checked_plan_add(total, bytes) &&
        semantic_rt_vector_capacity_bytes(group.exact_terms, &bytes) &&
        checked_plan_add(total, bytes) &&
        semantic_rt_vector_capacity_bytes(group.exact_offsets, &bytes) &&
        checked_plan_add(total, bytes) &&
        semantic_rt_vector_capacity_bytes(group.boxes, &bytes) &&
        checked_plan_add(total, bytes);
}

bool semantic_rt_group_build_host_bytes(
    const SemanticRtRegionGroupBuild& group,
    uint64_t* total
) {
    uint64_t bytes = 0u;
    return semantic_rt_vector_capacity_bytes(group.result_regions, &bytes) &&
        checked_plan_add(total, bytes) &&
        semantic_rt_vector_capacity_bytes(group.axes, &bytes) &&
        checked_plan_add(total, bytes);
}

bool compact_semantic_rt_plan_host_bytes(
    const CompactSemanticRtRegionPlan& plan,
    uint64_t* total_out
) {
    if (total_out == nullptr) return false;
    uint64_t total = sizeof(CompactSemanticRtRegionPlan);
    uint64_t bytes = 0u;
    const bool base_ok =
        semantic_rt_vector_capacity_bytes(plan.groups, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.group_path_offsets, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.group_region_ids, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.flat_exact_terms, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.flat_exact_offsets, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.flat_boxes, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.group_primary_axes, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.group_paired_axes, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.group_dims, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.bin_offsets, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.bin_candidates, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.bin_lo, &bytes) && checked_plan_add(&total, bytes) &&
        semantic_rt_vector_capacity_bytes(plan.bin_inv_span, &bytes) && checked_plan_add(&total, bytes);
    if (!base_ok) return false;
    for (const SemanticRtRegionGroup& group : plan.groups) {
        if (!semantic_rt_group_host_bytes(group, &total)) return false;
    }
    *total_out = total;
    return true;
}

bool semantic_rt_group_builds_host_bytes(
    const std::vector<SemanticRtRegionGroupBuild>& builds,
    uint64_t* total_out
) {
    if (total_out == nullptr) return false;
    uint64_t total = sizeof(builds);
    uint64_t bytes = 0u;
    if (!semantic_rt_vector_capacity_bytes(builds, &bytes) || !checked_plan_add(&total, bytes)) {
        return false;
    }
    for (const SemanticRtRegionGroupBuild& build : builds) {
        if (!semantic_rt_group_build_host_bytes(build, &total)) return false;
    }
    *total_out = total;
    return true;
}

bool semantic_rt_group_merge_axes(
    const std::vector<SemanticRtAxis>& current,
    const std::vector<SemanticRtAxis>& incoming,
    std::vector<SemanticRtAxis>* merged_out
) {
    if (merged_out == nullptr) return false;
    std::vector<SemanticRtAxis> merged = current;
    for (const SemanticRtAxis& axis : incoming) {
        if (!semantic_rt_append_axis(merged, axis)) return false;
    }
    semantic_rt_sort_axes(merged);
    *merged_out = std::move(merged);
    return true;
}

bool semantic_rt_triangle_axis_safe(float lo, float hi) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) return false;
    const double span = static_cast<double>(hi) - static_cast<double>(lo);
    const double scale = std::max({
        1.0,
        std::abs(static_cast<double>(lo)),
        std::abs(static_cast<double>(hi)),
    });
    return span >= std::ldexp(scale, -12);
}

bool semantic_rt_ranges_overlap_open_closed(float a_lo, float a_hi, float b_lo, float b_hi) {
    return std::max(a_lo, b_lo) < std::min(a_hi, b_hi);
}

bool semantic_rt_boxes_non_overlapping_2d(const SemanticRtRegionGroup& group) {
    if (group.axes.size() != 2u || !group.all_boxes_bounded) return false;
    std::vector<uint32_t> order(group.boxes.size());
    for (uint32_t index = 0u; index < order.size(); ++index) order[index] = index;
    std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
        const auto& a = group.boxes[left];
        const auto& b = group.boxes[right];
        if (a.lo_x != b.lo_x) return a.lo_x < b.lo_x;
        if (a.hi_x != b.hi_x) return a.hi_x < b.hi_x;
        return left < right;
    });
    std::vector<uint32_t> active;
    active.reserve(group.boxes.size());
    for (const uint32_t box_index : order) {
        const auto& box = group.boxes[box_index];
        active.erase(
            std::remove_if(active.begin(), active.end(), [&](uint32_t active_index) {
                return group.boxes[active_index].hi_x <= box.lo_x;
            }),
            active.end()
        );
        for (const uint32_t active_index : active) {
            const auto& other = group.boxes[active_index];
            if (semantic_rt_ranges_overlap_open_closed(
                    other.lo_y, other.hi_y, box.lo_y, box.hi_y)) {
                return false;
            }
        }
        active.push_back(box_index);
    }
    return true;
}

bool semantic_rt_group_triangle_safe(const SemanticRtRegionGroup& group) {
    if (group.axes.size() != 2u || !group.all_boxes_bounded || group.boxes.empty()) {
        return false;
    }
    return std::all_of(group.boxes.begin(), group.boxes.end(), [](const auto& box) {
        return semantic_rt_triangle_axis_safe(box.lo_x, box.hi_x) &&
            semantic_rt_triangle_axis_safe(box.lo_y, box.hi_y);
    });
}

uint32_t semantic_rt_sm_bin(float value) {
    const uint64_t bucket = static_cast<uint64_t>(
        gafime_cuda_v1::rt_kernel::rt_float_bucket(value)
    );
    const uint32_t bin = static_cast<uint32_t>(
        (bucket * kSemanticRtRegionSmBinCount) >> 23u
    );
    return bin < kSemanticRtRegionSmBinCount ? bin : kSemanticRtRegionSmBinCount - 1u;
}

uint32_t semantic_rt_sm_bin_query_bound(float value, float lo, float inv_span) {
    if (std::isfinite(lo) && std::isfinite(inv_span) && inv_span > 0.0f) {
        const float scaled = (value - lo) * inv_span *
            static_cast<float>(kSemanticRtRegionSmBinCount);
        if (scaled <= 0.0f) return 0u;
        if (scaled >= static_cast<float>(kSemanticRtRegionSmBinCount)) {
            return kSemanticRtRegionSmBinCount - 1u;
        }
        return static_cast<uint32_t>(scaled);
    }
    return semantic_rt_sm_bin(value);
}

int build_semantic_rt_group(
    const std::vector<GafimeDecisionPathTerm>& primary_terms,
    const std::vector<GafimeDecisionPathTerm>& paired_terms,
    const uint32_t* region_offsets,
    const SemanticRtRegionGroupBuild& source,
    SemanticRtRegionGroup* group_out
) {
    if (region_offsets == nullptr || group_out == nullptr || source.result_regions.empty() ||
        source.axes.empty() || source.axes.size() > GAFIME_CUDA_RT_SEMANTIC_MAX_AXES) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    SemanticRtRegionGroup group{};
    group.result_regions = source.result_regions;
    group.axes = source.axes;
    semantic_rt_sort_axes(group.axes);
    uint64_t exact_term_count = 0u;
    for (const uint32_t result_region : group.result_regions) {
        const uint32_t begin = region_offsets[result_region];
        const uint32_t end = region_offsets[result_region + 1u];
        if (end < begin || !checked_add_u64(exact_term_count, end - begin, &exact_term_count) ||
            exact_term_count > static_cast<uint64_t>(SIZE_MAX)) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
    }
    group.exact_terms.reserve(static_cast<size_t>(exact_term_count));
    group.exact_offsets.reserve(group.result_regions.size() + 1u);
    group.exact_offsets.push_back(0u);
    group.boxes.reserve(group.result_regions.size());
    for (const uint32_t result_region : group.result_regions) {
        float lo[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        float hi[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        uint32_t open_lo_mask = 0u;
        const uint32_t begin = region_offsets[result_region];
        const uint32_t end = region_offsets[result_region + 1u];
        for (uint32_t term_index = begin; term_index < end; ++term_index) {
            const GafimeDecisionPathTerm& primary = primary_terms[term_index];
            const GafimeDecisionPathTerm& paired = paired_terms[term_index];
            const uint32_t axis = semantic_rt_axis_index(
                group.axes, {primary.feature, paired.feature});
            if (axis == UINT32_MAX) return GAFIME_STATUS_DEVICE_ERROR;
            GafimeDecisionPathTerm exact = primary;
            exact.feature = axis;
            group.exact_terms.push_back(exact);
            if (primary.sign == GAFIME_DECISION_PATH_SIGN_LE) {
                hi[axis] = std::min(hi[axis], primary.threshold);
            } else if (primary.sign == GAFIME_DECISION_PATH_SIGN_GT) {
                if (primary.threshold >= lo[axis]) {
                    lo[axis] = primary.threshold;
                    open_lo_mask |= 1u << axis;
                }
            } else {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
        for (uint32_t axis = 0u; axis < group.axes.size(); ++axis) {
            if (lo[axis] > hi[axis] ||
                (lo[axis] == hi[axis] && (open_lo_mask & (1u << axis)) != 0u)) {
                /* An externally supplied contradictory region remains a valid
                 * physical all-zero query, never an accidental match. */
                lo[axis] = 0.0f;
                hi[axis] = 0.0f;
                open_lo_mask |= 1u << axis;
            }
            if (lo[axis] <= -FLT_MAX * 0.5f || hi[axis] >= FLT_MAX * 0.5f) {
                group.all_boxes_bounded = false;
            }
        }
        if (group.exact_terms.size() > static_cast<size_t>(UINT32_MAX)) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        group.exact_offsets.push_back(static_cast<uint32_t>(group.exact_terms.size()));
        group.boxes.push_back({
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], open_lo_mask,
            static_cast<uint32_t>(group.axes.size()),
        });
    }
    *group_out = std::move(group);
    return GAFIME_STATUS_OK;
}

int build_compact_semantic_rt_region_plan(
    const GafimeSemanticRtRegionQueryDesc* desc,
    const std::vector<GafimeDecisionPathTerm>& primary_terms,
    const std::vector<GafimeDecisionPathTerm>& paired_terms,
    CompactSemanticRtRegionPlan* plan_out
) {
    if (desc == nullptr || plan_out == nullptr || desc->region_offsets == nullptr ||
        desc->partition_offsets == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    std::vector<SemanticRtRegionGroupBuild> builds;
    builds.reserve(desc->region_count);
    for (uint32_t partition = 0u; partition < desc->partition_count; ++partition) {
        const uint32_t partition_begin = desc->partition_offsets[partition];
        const uint32_t partition_end = desc->partition_offsets[partition + 1u];
        const size_t partition_group_begin = builds.size();
        for (uint32_t region = partition_begin; region < partition_end; ++region) {
            std::vector<SemanticRtAxis> region_axes;
            const uint32_t begin = desc->region_offsets[region];
            const uint32_t end = desc->region_offsets[region + 1u];
            for (uint32_t term_index = begin; term_index < end; ++term_index) {
                if (!semantic_rt_append_axis(
                        region_axes,
                        {primary_terms[term_index].feature, paired_terms[term_index].feature})) {
                    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
                }
            }
            semantic_rt_sort_axes(region_axes);
            bool placed = false;
            std::vector<SemanticRtAxis> merged_axes;
            for (size_t group_index = partition_group_begin; group_index < builds.size(); ++group_index) {
                if (semantic_rt_group_merge_axes(builds[group_index].axes, region_axes, &merged_axes)) {
                    builds[group_index].axes = merged_axes;
                    builds[group_index].result_regions.push_back(region);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                SemanticRtRegionGroupBuild group{};
                group.axes = std::move(region_axes);
                group.result_regions.push_back(region);
                builds.push_back(std::move(group));
            }
        }
    }
    if (builds.empty() || builds.size() > GAFIME_CUDA_RT_REGION_QUERY_MAX_GROUPS) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    CompactSemanticRtRegionPlan plan{};
    plan.groups.reserve(builds.size());
    plan.group_path_offsets.reserve(builds.size() + 1u);
    plan.group_region_ids.reserve(desc->region_count);
    plan.flat_exact_terms.reserve(primary_terms.size());
    plan.flat_exact_offsets.reserve(desc->region_count + 1u);
    plan.flat_boxes.reserve(desc->region_count);
    plan.group_primary_axes.reserve(builds.size() * 3u);
    plan.group_paired_axes.reserve(builds.size() * 3u);
    plan.group_dims.reserve(builds.size());
    plan.flat_exact_offsets.push_back(0u);
    plan.first_hit_eligible = true;
    plan.triangle_eligible = true;
    for (const SemanticRtRegionGroupBuild& build : builds) {
        SemanticRtRegionGroup group{};
        const int status = build_semantic_rt_group(
            primary_terms, paired_terms, desc->region_offsets, build, &group);
        if (status != GAFIME_STATUS_OK) return status;
        plan.group_path_offsets.push_back(static_cast<uint32_t>(plan.flat_boxes.size()));
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            plan.group_primary_axes.push_back(
                axis < group.axes.size() ? group.axes[axis].primary_slot : 0u);
            plan.group_paired_axes.push_back(
                axis < group.axes.size() ? group.axes[axis].paired_slot : 0u);
        }
        plan.group_dims.push_back(static_cast<uint32_t>(group.axes.size()));
        for (uint32_t local_region = 0u; local_region < group.result_regions.size(); ++local_region) {
            const uint32_t begin = group.exact_offsets[local_region];
            const uint32_t end = group.exact_offsets[local_region + 1u];
            plan.flat_exact_terms.insert(
                plan.flat_exact_terms.end(),
                group.exact_terms.begin() + begin,
                group.exact_terms.begin() + end
            );
            if (plan.flat_exact_terms.size() > static_cast<size_t>(UINT32_MAX)) {
                return GAFIME_STATUS_OUT_OF_MEMORY;
            }
            plan.flat_exact_offsets.push_back(
                static_cast<uint32_t>(plan.flat_exact_terms.size()));
            plan.flat_boxes.push_back(group.boxes[local_region]);
            plan.group_region_ids.push_back(group.result_regions[local_region]);
        }
        plan.first_hit_eligible = plan.first_hit_eligible &&
            semantic_rt_boxes_non_overlapping_2d(group);
        plan.triangle_eligible = plan.triangle_eligible && semantic_rt_group_triangle_safe(group);
        plan.groups.push_back(std::move(group));
    }
    plan.group_path_offsets.push_back(static_cast<uint32_t>(plan.flat_boxes.size()));
    if (plan.flat_boxes.size() != desc->region_count ||
        plan.group_region_ids.size() != desc->region_count ||
        plan.flat_exact_offsets.size() != static_cast<size_t>(desc->region_count) + 1u) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

    plan.bin_offsets.resize(plan.groups.size() * (kSemanticRtRegionSmBinCount + 1u));
    plan.bin_lo.resize(plan.groups.size(), 0.0f);
    plan.bin_inv_span.resize(plan.groups.size(), 0.0f);
    std::vector<std::vector<uint32_t>> bins(
        plan.groups.size() * kSemanticRtRegionSmBinCount
    );
    for (uint32_t group = 0u; group < plan.groups.size(); ++group) {
        const uint32_t begin = plan.group_path_offsets[group];
        const uint32_t end = plan.group_path_offsets[group + 1u];
        float group_lo = FLT_MAX;
        float group_hi = -FLT_MAX;
        bool query_bound = begin < end;
        for (uint32_t physical_region = begin; physical_region < end; ++physical_region) {
            const auto& box = plan.flat_boxes[physical_region];
            if (box.lo_x <= -FLT_MAX * 0.5f || box.hi_x >= FLT_MAX * 0.5f ||
                !std::isfinite(box.lo_x) || !std::isfinite(box.hi_x)) {
                query_bound = false;
                break;
            }
            group_lo = std::min(group_lo, box.lo_x);
            group_hi = std::max(group_hi, box.hi_x);
        }
        float inv_span = 0.0f;
        if (query_bound && group_lo < group_hi) {
            const float span = group_hi - group_lo;
            if (std::isfinite(span) && span > 0.0f) inv_span = 1.0f / span;
        }
        plan.bin_lo[group] = inv_span > 0.0f ? group_lo : 0.0f;
        plan.bin_inv_span[group] = inv_span;
        for (uint32_t physical_region = begin; physical_region < end; ++physical_region) {
            const auto& box = plan.flat_boxes[physical_region];
            const uint32_t lo_bin = semantic_rt_sm_bin_query_bound(
                box.lo_x, plan.bin_lo[group], plan.bin_inv_span[group]);
            const uint32_t hi_bin = semantic_rt_sm_bin_query_bound(
                box.hi_x, plan.bin_lo[group], plan.bin_inv_span[group]);
            /* One-bin conservative expansion absorbs host/device rounding at
             * bin edges.  The original exact term guard remains definitive. */
            const uint32_t first = std::min(lo_bin, hi_bin) == 0u
                ? 0u : std::min(lo_bin, hi_bin) - 1u;
            const uint32_t last = std::max(lo_bin, hi_bin) + 1u >= kSemanticRtRegionSmBinCount
                ? kSemanticRtRegionSmBinCount - 1u : std::max(lo_bin, hi_bin) + 1u;
            for (uint32_t bin = first; bin <= last; ++bin) {
                bins[static_cast<size_t>(group) * kSemanticRtRegionSmBinCount + bin].push_back(
                    physical_region);
            }
        }
    }
    for (uint32_t group = 0u; group < plan.groups.size(); ++group) {
        const size_t offset_base = static_cast<size_t>(group) *
            (kSemanticRtRegionSmBinCount + 1u);
        for (uint32_t bin = 0u; bin < kSemanticRtRegionSmBinCount; ++bin) {
            plan.bin_offsets[offset_base + bin] =
                static_cast<uint32_t>(plan.bin_candidates.size());
            const auto& candidates = bins[static_cast<size_t>(group) *
                kSemanticRtRegionSmBinCount + bin];
            plan.bin_candidates.insert(
                plan.bin_candidates.end(), candidates.begin(), candidates.end());
        }
        plan.bin_offsets[offset_base + kSemanticRtRegionSmBinCount] =
            static_cast<uint32_t>(plan.bin_candidates.size());
    }
    uint64_t retained_host_bytes = 0u;
    uint64_t build_host_bytes = 0u;
    uint64_t bins_host_bytes = sizeof(bins);
    uint64_t bytes = 0u;
    if (!compact_semantic_rt_plan_host_bytes(plan, &retained_host_bytes) ||
        !semantic_rt_group_builds_host_bytes(builds, &build_host_bytes) ||
        !semantic_rt_vector_capacity_bytes(bins, &bytes) ||
        !checked_plan_add(&bins_host_bytes, bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    for (const std::vector<uint32_t>& bin : bins) {
        if (!semantic_rt_vector_capacity_bytes(bin, &bytes) ||
            !checked_plan_add(&bins_host_bytes, bytes)) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
    }
    uint64_t concurrent_host_bytes = 0u;
    if (!checked_add_u64(retained_host_bytes, build_host_bytes, &concurrent_host_bytes) ||
        !checked_add_u64(concurrent_host_bytes, bins_host_bytes, &concurrent_host_bytes) ||
        /* Vector growth may briefly retain an old allocation beside the new
         * capacity.  Charge twice this measured live set, plus the two tiny
         * per-region axis vectors used while groups are formed. */
        !checked_mul_u64(concurrent_host_bytes, 2u, &plan.host_build_peak_bytes) ||
        !checked_add_u64(
            plan.host_build_peak_bytes,
            2u * (sizeof(std::vector<SemanticRtAxis>) + 3u * sizeof(SemanticRtAxis)),
            &plan.host_build_peak_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    *plan_out = std::move(plan);
    return GAFIME_STATUS_OK;
}

/* This owner deliberately has no device-global cache.  A query snapshots the
 * physical descriptor into its own compact geometry/point buffers, and Rust
 * retains the bank handles for the entire query lifetime.  That makes bank
 * pointer reuse irrelevant and keeps label rebinding out of cache identity. */
template <size_t N>
bool compact_semantic_rt_reserved_zero(const uint64_t (&reserved)[N]) {
    for (const uint64_t value : reserved) {
        if (value != 0u) return false;
    }
    return true;
}

bool compact_semantic_rt_abi_compatible(uint32_t abi_version, uint32_t struct_size, size_t size) {
    return abi_version == GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION &&
        struct_size >= size;
}

/* A caller may expose only the aligned ABI/version prefix while probing an
 * invalid call.  Read that prefix, but never touch the writable table fields
 * until the complete known table layout has been negotiated. */
int validate_compact_semantic_rt_stats_table_header(
    const GafimeSemanticRtRegionStatsTable* table
) {
    if (table == nullptr || !gafime_gpu_abi::naturally_aligned(table)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    return compact_semantic_rt_abi_compatible(
               table->abi_version, table->struct_size, sizeof(*table))
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_ABI_MISMATCH;
}

void clear_compact_semantic_rt_stats_table(GafimeSemanticRtRegionStatsTable* table) {
    table->requested_statistic_mask = 0u;
    table->finalized_mask = 0u;
    table->count = 0u;
}

class CompactSemanticRtTemporaryAllocation {
public:
    CompactSemanticRtTemporaryAllocation() = default;
    CompactSemanticRtTemporaryAllocation(const CompactSemanticRtTemporaryAllocation&) = delete;
    CompactSemanticRtTemporaryAllocation& operator=(const CompactSemanticRtTemporaryAllocation&) = delete;

    ~CompactSemanticRtTemporaryAllocation() {
        if (ptr_ != nullptr) static_cast<void>(cudaFree(ptr_));
    }

    int allocate(uint64_t bytes) {
        if (bytes == 0u) return GAFIME_STATUS_OK;
        if (bytes > static_cast<uint64_t>(SIZE_MAX)) return GAFIME_STATUS_OUT_OF_MEMORY;
        return cuda_status(cudaMalloc(&ptr_, static_cast<size_t>(bytes)));
    }

    template <typename T>
    T* as(uint64_t byte_offset = 0u) const {
        return reinterpret_cast<T*>(static_cast<uint8_t*>(ptr_) + byte_offset);
    }

private:
    void* ptr_ = nullptr;
};

struct CompactSemanticRtRegionQuery {
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    explicit CompactSemanticRtRegionQuery(uint32_t id)
        : device_id(id), optix_state(std::make_unique<RtDeviceState>(id)) {}
#else
    explicit CompactSemanticRtRegionQuery(uint32_t id) : device_id(id) {}
#endif

    CompactSemanticRtRegionQuery(const CompactSemanticRtRegionQuery&) = delete;
    CompactSemanticRtRegionQuery& operator=(const CompactSemanticRtRegionQuery&) = delete;

    /* Geometry construction can allocate host vectors after generic CUDA
     * buffers exist.  Unwinding through the owning unique_ptr must release
     * those raw buffers on the query device instead of relying on a later
     * explicit free that will never receive a handle. */
    ~CompactSemanticRtRegionQuery() noexcept {
        ScopedCudaDevice device(device_id);
        if (device.status() == cudaSuccess) {
            static_cast<void>(reset());
        }
    }

    uint64_t magic = kSemanticRtRegionQueryMagic;
    std::mutex mutex;
    GafimeGpuSemanticBank primary_bank = nullptr;
    GafimeGpuSemanticBank paired_bank = nullptr;
    float* primary_columns = nullptr;
    float* paired_columns = nullptr;
    uint32_t device_id = UINT32_MAX;
    uint64_t rows = 0u;
    uint32_t region_count = 0u;
    uint32_t group_count = 0u;
    uint32_t words_per_region = 0u;
    bool has_paired = false;
    bool force_sm = false;
    bool force_sm_exhaustive = false;
    bool require_rt = false;
    bool geometry_available = false;
    bool direct_first_hit = false;
    bool membership_valid = false;
    uint64_t persistent_bytes = 0u;
    CompactSemanticRtRegionPlan plan;

    GafimeDecisionPathTerm* source_primary_terms_device = nullptr;
    GafimeDecisionPathTerm* source_paired_terms_device = nullptr;
    uint32_t* source_offsets_device = nullptr;
    GafimeDecisionPathTerm* exact_terms_device = nullptr;
    uint32_t* exact_offsets_device = nullptr;
    uint32_t* group_path_offsets_device = nullptr;
    uint32_t* group_region_ids_device = nullptr;
    uint32_t* group_primary_axes_device = nullptr;
    uint32_t* group_paired_axes_device = nullptr;
    uint32_t* group_dims_device = nullptr;
    uint32_t* bin_offsets_device = nullptr;
    uint32_t* bin_candidates_device = nullptr;
    float* bin_lo_device = nullptr;
    float* bin_inv_span_device = nullptr;
    float* primary_points_device = nullptr;
    float* paired_points_device = nullptr;
    uint32_t* primary_membership_words_device = nullptr;
    uint32_t* paired_membership_words_device = nullptr;
    uint32_t* label_zero_words_device = nullptr;
    uint32_t* label_one_words_device = nullptr;
    /* In the proof-gated direct-first-hit route this stores one canonical
     * result-region ordinal plus one per row.  The ordinary coverage endpoint
     * maps it back to 0/1, while weighted materialization uses the ordinal to
     * select the matching frozen weight without reconstructing an R by N mask. */
    uint32_t* direct_region_ordinals_device = nullptr;
    GafimeSemanticRtRegionExactStats* stats_device = nullptr;

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    std::unique_ptr<RtDeviceState> optix_state;
    RtGeometryMode optix_geometry_mode = RtGeometryMode::CustomAabb;
    bool optix_geometry_ready = false;
#endif

    int reset() noexcept {
        int status = GAFIME_STATUS_OK;
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
        if (optix_state != nullptr) {
            optix_state->reset();
            optix_state.reset();
        }
        optix_geometry_ready = false;
        optix_geometry_mode = RtGeometryMode::CustomAabb;
#endif
        auto release = [&](void* ptr) {
            if (ptr == nullptr) return;
            const int candidate = cuda_status(cudaFree(ptr));
            if (status == GAFIME_STATUS_OK && candidate != GAFIME_STATUS_OK) status = candidate;
        };
        release(stats_device);
        release(direct_region_ordinals_device);
        release(label_one_words_device);
        release(label_zero_words_device);
        release(paired_membership_words_device);
        release(primary_membership_words_device);
        release(paired_points_device);
        release(primary_points_device);
        release(bin_candidates_device);
        release(bin_offsets_device);
        release(bin_inv_span_device);
        release(bin_lo_device);
        release(group_dims_device);
        release(group_paired_axes_device);
        release(group_primary_axes_device);
        release(group_region_ids_device);
        release(group_path_offsets_device);
        release(exact_offsets_device);
        release(exact_terms_device);
        release(source_offsets_device);
        release(source_paired_terms_device);
        release(source_primary_terms_device);
        stats_device = nullptr;
        direct_region_ordinals_device = nullptr;
        label_one_words_device = nullptr;
        label_zero_words_device = nullptr;
        paired_membership_words_device = nullptr;
        primary_membership_words_device = nullptr;
        paired_points_device = nullptr;
        primary_points_device = nullptr;
        bin_candidates_device = nullptr;
        bin_offsets_device = nullptr;
        bin_inv_span_device = nullptr;
        bin_lo_device = nullptr;
        group_dims_device = nullptr;
        group_paired_axes_device = nullptr;
        group_primary_axes_device = nullptr;
        group_region_ids_device = nullptr;
        group_path_offsets_device = nullptr;
        exact_offsets_device = nullptr;
        exact_terms_device = nullptr;
        source_offsets_device = nullptr;
        source_paired_terms_device = nullptr;
        source_primary_terms_device = nullptr;
        membership_valid = false;
        geometry_available = false;
        return status;
    }
};

CompactSemanticRtRegionQuery* compact_semantic_rt_query_from_handle(
    GafimeGpuSemanticRegionQuery handle
) {
    auto* query = static_cast<CompactSemanticRtRegionQuery*>(handle);
    return query != nullptr && query->magic == kSemanticRtRegionQueryMagic ? query : nullptr;
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
/* This is deliberately a plan, not a cache.  It is formed from the immutable
 * query snapshot before any query-owned CUDA allocation occurs.  OptiX
 * context/module/pipeline state is vendor-owned and excluded by the ABI;
 * SBT, parameter, GAS/IAS, and retained build-workspace buffers are not. */
struct CompactSemanticRtOptixAllocationPlan {
    RtGeometryMode geometry_mode = RtGeometryMode::CustomAabb;
    bool instanced = false;
    bool triangle = false;
    std::vector<OptixAccelBufferSizes> gas_sizes;
    std::vector<size_t> gas_output_offsets;
    OptixAccelBufferSizes ias_sizes{};
    size_t gas_temp_bytes = 0u;
    size_t gas_output_bytes = 0u;
    uint64_t host_construction_bytes = 0u;
    uint64_t explicit_bytes = 0u;
};

bool compact_semantic_rt_plan_add_count(
    uint64_t* total,
    uint64_t count,
    size_t element_size
) {
    uint64_t bytes = 0u;
    return checked_element_bytes(count, element_size, &bytes) && checked_plan_add(total, bytes);
}

bool compact_semantic_rt_plan_add_bytes(uint64_t* total, size_t bytes) {
    return total != nullptr && checked_plan_add(total, static_cast<uint64_t>(bytes));
}

int plan_compact_semantic_rt_optix_allocations(
    RtDeviceState& planning_state,
    uint32_t device_id,
    uint64_t rows,
    uint32_t region_count,
    const CompactSemanticRtRegionPlan& semantic_plan,
    CompactSemanticRtOptixAllocationPlan* plan_out
) {
    if (plan_out == nullptr || semantic_plan.groups.empty() ||
        semantic_plan.flat_boxes.size() != region_count || rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    CompactSemanticRtOptixAllocationPlan plan{};
    const uint32_t group_count_u32 = static_cast<uint32_t>(semantic_plan.groups.size());
    plan.geometry_mode = group_count_u32 == 1u && !semantic_plan.triangle_eligible
        ? RtGeometryMode::CustomAabb
        : semantic_plan.triangle_eligible
            ? RtGeometryMode::Triangle2dInstanced
            : RtGeometryMode::CustomAabbInstanced;
    plan.instanced = plan.geometry_mode != RtGeometryMode::CustomAabb;
    plan.triangle = plan.geometry_mode == RtGeometryMode::Triangle2dInstanced;

    /* Defer SBT CUDA allocation until this complete explicit plan has passed
     * the caller budget.  ComputeMemoryUsage consumes only the build shape;
     * an aligned non-null sizing address is never dereferenced by that query
     * and keeps this allocation-sizing pass free of query buffers. */
    int status = ensure_optix_program(planning_state, plan.geometry_mode, false);
    if (status != GAFIME_STATUS_OK) return status;
    RtOptixProgram& program = planning_state.program(plan.geometry_mode);
    if (!program.ready(device_id, plan.geometry_mode)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    const size_t group_count = semantic_plan.groups.size();
    plan.gas_sizes.resize(group_count);
    plan.gas_output_offsets.resize(group_count, 0u);
    uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    const CUdeviceptr sizing_address = static_cast<CUdeviceptr>(OPTIX_INSTANCE_BYTE_ALIGNMENT);
    size_t total_gas_output = 0u;
    size_t max_gas_temp = 0u;
    for (size_t group_index = 0u; group_index < group_count; ++group_index) {
        const SemanticRtRegionGroup& group = semantic_plan.groups[group_index];
        if (group.result_regions.empty() || group.boxes.size() != group.result_regions.size()) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        /* OptiX retains the host pointer to the buffer-address field for the
         * duration of this sizing call, so keep this field alive outside the
         * branch that describes the build input. */
        CUdeviceptr sizing_buffer = sizing_address;
        OptixBuildInput input = {};
        if (plan.triangle) {
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            input.triangleArray.vertexBuffers = &sizing_buffer;
            input.triangleArray.numVertices = static_cast<uint32_t>(group.boxes.size() * 4u);
            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes =
                sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
            input.triangleArray.indexBuffer = sizing_address;
            input.triangleArray.numIndexTriplets = static_cast<uint32_t>(group.boxes.size() * 2u);
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes =
                sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
            input.triangleArray.flags = geometry_flags;
            input.triangleArray.numSbtRecords = 1u;
        } else {
            input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            input.customPrimitiveArray.aabbBuffers = &sizing_buffer;
            input.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(group.boxes.size());
            input.customPrimitiveArray.flags = geometry_flags;
            input.customPrimitiveArray.numSbtRecords = 1u;
        }
        OptixAccelBuildOptions options = {};
        options.buildFlags = plan.triangle ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_NONE;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        status = optix_status(optixAccelComputeMemoryUsage(
            program.context, &options, &input, 1u, &plan.gas_sizes[group_index]));
        if (status != GAFIME_STATUS_OK) return status;
        plan.gas_output_offsets[group_index] = align_up_size(
            total_gas_output, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        if (plan.gas_output_offsets[group_index] > SIZE_MAX -
                plan.gas_sizes[group_index].outputSizeInBytes) {
            return GAFIME_STATUS_OUT_OF_MEMORY;
        }
        total_gas_output = plan.gas_output_offsets[group_index] +
            plan.gas_sizes[group_index].outputSizeInBytes;
        max_gas_temp = std::max(max_gas_temp,
            static_cast<size_t>(plan.gas_sizes[group_index].tempSizeInBytes));
    }
    plan.gas_temp_bytes = max_gas_temp;
    plan.gas_output_bytes = total_gas_output;
    if (plan.instanced) {
        OptixBuildInput ias_input = {};
        ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        ias_input.instanceArray.instances = sizing_address;
        ias_input.instanceArray.numInstances = static_cast<uint32_t>(group_count);
        OptixAccelBuildOptions ias_options = {};
        ias_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        ias_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        status = optix_status(optixAccelComputeMemoryUsage(
            program.context, &ias_options, &ias_input, 1u, &plan.ias_sizes));
        if (status != GAFIME_STATUS_OK) return status;
    }

    uint64_t explicit_bytes = 0u;
    const uint64_t regions = region_count;
    const uint64_t groups = group_count_u32;
    const bool plan_ok =
        compact_semantic_rt_plan_add_count(
            &explicit_bytes, regions, sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox)) &&
        (plan.triangle
            ? compact_semantic_rt_plan_add_count(
                  &explicit_bytes, regions * 4u,
                  sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex)) &&
              compact_semantic_rt_plan_add_count(
                  &explicit_bytes, regions * 2u,
                  sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex))
            : compact_semantic_rt_plan_add_count(&explicit_bytes, regions, sizeof(OptixAabb))) &&
        (!plan.instanced || compact_semantic_rt_plan_add_count(
            &explicit_bytes, groups, sizeof(OptixInstance))) &&
        compact_semantic_rt_plan_add_count(&explicit_bytes, 1u, sizeof(GafimeRtParams)) &&
        compact_semantic_rt_plan_add_count(&explicit_bytes, 3u, sizeof(EmptyRecord)) &&
        compact_semantic_rt_plan_add_bytes(&explicit_bytes, plan.gas_temp_bytes) &&
        compact_semantic_rt_plan_add_bytes(&explicit_bytes, plan.gas_output_bytes) &&
        (!plan.instanced || compact_semantic_rt_plan_add_bytes(
            &explicit_bytes, plan.ias_sizes.tempSizeInBytes)) &&
        (!plan.instanced || compact_semantic_rt_plan_add_bytes(
            &explicit_bytes, plan.ias_sizes.outputSizeInBytes));
    if (!plan_ok) return GAFIME_STATUS_OUT_OF_MEMORY;
    uint64_t host_bytes = sizeof(CompactSemanticRtOptixAllocationPlan);
    const bool host_ok =
        compact_semantic_rt_plan_add_count(
            &host_bytes, static_cast<uint64_t>(plan.gas_sizes.capacity()), sizeof(OptixAccelBufferSizes)) &&
        compact_semantic_rt_plan_add_count(
            &host_bytes, static_cast<uint64_t>(plan.gas_output_offsets.capacity()), sizeof(size_t)) &&
        /* Actual geometry construction owns the aggregate vectors below.  A
         * triangle group is first assembled locally then appended, so reserve
         * one additional full geometry payload for that bounded staging peak. */
        compact_semantic_rt_plan_add_count(
            &host_bytes, groups, sizeof(OptixTraversableHandle)) &&
        compact_semantic_rt_plan_add_count(
            &host_bytes, groups, sizeof(size_t)) &&
        compact_semantic_rt_plan_add_count(
            &host_bytes, groups, sizeof(OptixAccelBufferSizes)) &&
        compact_semantic_rt_plan_add_count(
            &host_bytes, groups, sizeof(uint32_t) * 8u) &&
        (!plan.instanced || compact_semantic_rt_plan_add_count(
            &host_bytes, groups, sizeof(OptixInstance))) &&
        (plan.triangle
            ? compact_semantic_rt_plan_add_count(
                  &host_bytes, regions * 8u,
                  sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex)) &&
              compact_semantic_rt_plan_add_count(
                  &host_bytes, regions * 4u,
                  sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex)) &&
              /* `build_rt_triangles` receives a per-group RtBoxPlan copy;
               * the largest group is bounded by the submitted region count. */
              compact_semantic_rt_plan_add_count(
                  &host_bytes, regions,
                  sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox))
            : compact_semantic_rt_plan_add_count(&host_bytes, regions, sizeof(OptixAabb)));
    if (!host_ok) return GAFIME_STATUS_OUT_OF_MEMORY;
    plan.host_construction_bytes = host_bytes;
    plan.explicit_bytes = explicit_bytes;
    *plan_out = std::move(plan);
    return GAFIME_STATUS_OK;
}
#endif

int validate_compact_semantic_rt_query_shape(const GafimeSemanticRtRegionQueryDesc* desc) {
    if (desc == nullptr || !gafime_gpu_abi::naturally_aligned(desc) ||
        !compact_semantic_rt_abi_compatible(desc->abi_version, desc->struct_size, sizeof(*desc))) {
        return desc != nullptr && gafime_gpu_abi::naturally_aligned(desc)
            ? GAFIME_STATUS_ABI_MISMATCH
            : GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t known_flags = GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT |
        GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM |
        GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE;
    const uint32_t selected_flags = desc->flags & known_flags;
    if ((desc->flags & ~known_flags) != 0u ||
        (selected_flags != GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT &&
         selected_flags != GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM &&
         selected_flags != GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE) ||
        desc->reserved32 != 0u ||
        !compact_semantic_rt_reserved_zero(desc->reserved) || desc->primary_bank == nullptr ||
        desc->region_count == 0u ||
        desc->region_count > GAFIME_CUDA_RT_REGION_QUERY_MAX_REGIONS ||
        desc->partition_count == 0u ||
        desc->partition_count > GAFIME_CUDA_RT_REGION_QUERY_MAX_GROUPS ||
        desc->term_count == 0u || desc->terms == nullptr || desc->region_offsets == nullptr ||
        desc->partition_offsets == nullptr ||
        (desc->paired_bank == nullptr && desc->paired_term_slots != nullptr) ||
        (desc->paired_bank != nullptr && desc->paired_term_slots == nullptr)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t max_terms = static_cast<uint64_t>(desc->region_count) *
        GAFIME_CUDA_RT_REGION_QUERY_MAX_TERMS_PER_REGION;
    const uint64_t region_offset_count = static_cast<uint64_t>(desc->region_count) + 1u;
    const uint64_t partition_offset_count = static_cast<uint64_t>(desc->partition_count) + 1u;
    /* The external arrays remain caller-owned only for this synchronous
     * create call.  Validate their natural alignment and host addressable
     * byte extent before any offset/term indexing or CUDA copy. */
    if (desc->term_count > max_terms || desc->term_count > static_cast<uint64_t>(UINT32_MAX) ||
        !gafime_gpu_abi::naturally_aligned(desc->terms) ||
        !gafime_gpu_abi::naturally_aligned(desc->region_offsets) ||
        !gafime_gpu_abi::naturally_aligned(desc->partition_offsets) ||
        (desc->paired_bank != nullptr &&
         !gafime_gpu_abi::naturally_aligned(desc->paired_term_slots)) ||
        !gafime_gpu_abi::fits_host_bytes(
            desc->term_count, sizeof(GafimeSemanticFrozenRegionTerm)) ||
        (desc->paired_bank != nullptr && !gafime_gpu_abi::fits_host_bytes(
            desc->term_count, sizeof(uint32_t))) ||
        !gafime_gpu_abi::fits_host_bytes(region_offset_count, sizeof(uint32_t)) ||
        !gafime_gpu_abi::fits_host_bytes(partition_offset_count, sizeof(uint32_t)) ||
        desc->region_offsets[0] != 0u ||
        desc->region_offsets[desc->region_count] != desc->term_count ||
        desc->partition_offsets[0] != 0u ||
        desc->partition_offsets[desc->partition_count] != desc->region_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t partition = 0u; partition < desc->partition_count; ++partition) {
        const uint32_t begin = desc->partition_offsets[partition];
        const uint32_t end = desc->partition_offsets[partition + 1u];
        if (begin >= end || end > desc->region_count) return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    return GAFIME_STATUS_OK;
}

int validate_compact_semantic_rt_query_desc(
    const GafimeSemanticRtRegionQueryDesc* desc,
    std::vector<GafimeDecisionPathTerm>* primary_terms_out,
    std::vector<GafimeDecisionPathTerm>* paired_terms_out
) {
    if (primary_terms_out == nullptr || paired_terms_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const int shape_status = validate_compact_semantic_rt_query_shape(desc);
    if (shape_status != GAFIME_STATUS_OK) return shape_status;

    primary_terms_out->clear();
    paired_terms_out->clear();
    primary_terms_out->reserve(static_cast<size_t>(desc->term_count));
    paired_terms_out->reserve(static_cast<size_t>(desc->term_count));
    for (uint32_t region = 0u; region < desc->region_count; ++region) {
        const uint32_t begin = desc->region_offsets[region];
        const uint32_t end = desc->region_offsets[region + 1u];
        if (begin >= end || end > desc->term_count ||
            end - begin > GAFIME_CUDA_RT_REGION_QUERY_MAX_TERMS_PER_REGION) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t term_index = begin; term_index < end; ++term_index) {
            const GafimeSemanticFrozenRegionTerm& source = desc->terms[term_index];
            if ((source.threshold_bits >> 32u) != 0u ||
                (source.relation != GAFIME_SEMANTIC_REGION_LESS_EQUAL &&
                 source.relation != GAFIME_SEMANTIC_REGION_GREATER_THAN)) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
            const uint32_t bits = static_cast<uint32_t>(source.threshold_bits);
            float threshold = 0.0f;
            std::memcpy(&threshold, &bits, sizeof(threshold));
            if (!std::isfinite(threshold) || std::fpclassify(threshold) == FP_SUBNORMAL) {
                return GAFIME_STATUS_UNSUPPORTED_BACKEND;
            }
            const uint32_t sign = source.relation == GAFIME_SEMANTIC_REGION_LESS_EQUAL
                ? GAFIME_DECISION_PATH_SIGN_LE
                : GAFIME_DECISION_PATH_SIGN_GT;
            primary_terms_out->push_back({source.input_slot, sign, threshold, 0u, {0u, 0u}});
            const uint32_t paired_slot = desc->paired_bank == nullptr
                ? source.input_slot
                : desc->paired_term_slots[term_index];
            paired_terms_out->push_back({paired_slot, sign, threshold, 0u, {0u, 0u}});
        }
    }
    return GAFIME_STATUS_OK;
}

int validate_compact_semantic_rt_query_banks(
    const GafimeSemanticRtRegionQueryDesc* desc,
    const std::vector<GafimeDecisionPathTerm>& primary_terms,
    const std::vector<GafimeDecisionPathTerm>& paired_terms,
    gafime_cuda_v1::detail::CudaSemanticBankView* primary_out,
    gafime_cuda_v1::detail::CudaSemanticBankView* paired_out
) {
    if (desc == nullptr || primary_out == nullptr || paired_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    int status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(desc->primary_bank, primary_out);
    if (status != GAFIME_STATUS_OK) return status;
    if (primary_out->rows == 0u || primary_out->rows > GAFIME_CUDA_RT_REGION_QUERY_MAX_ROWS ||
        primary_out->initialized_slots == nullptr ||
        primary_out->initialized_slots->size() != primary_out->slot_capacity) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    for (const GafimeDecisionPathTerm& term : primary_terms) {
        if (term.feature >= primary_out->slot_capacity ||
            (*primary_out->initialized_slots)[term.feature] == 0u) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    *paired_out = {};
    if (desc->paired_bank == nullptr) return GAFIME_STATUS_OK;
    status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(desc->paired_bank, paired_out);
    if (status != GAFIME_STATUS_OK) return status;
    if (paired_out->rows != primary_out->rows || paired_out->device_id != primary_out->device_id ||
        !gafime_gpu_abi::route_fields_equal(primary_out->route, paired_out->route) ||
        paired_out->initialized_slots == nullptr ||
        paired_out->initialized_slots->size() != paired_out->slot_capacity) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (const GafimeDecisionPathTerm& term : paired_terms) {
        if (term.feature >= paired_out->slot_capacity ||
            (*paired_out->initialized_slots)[term.feature] == 0u) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    return GAFIME_STATUS_OK;
}

template <typename T>
int allocate_compact_semantic_rt_persistent(
    T** ptr,
    uint64_t count,
    uint64_t max_bytes,
    uint64_t* used_bytes
) {
    if (ptr == nullptr || used_bytes == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    *ptr = nullptr;
    uint64_t bytes = 0u;
    uint64_t next = 0u;
    if (!checked_element_bytes(count, sizeof(T), &bytes) ||
        !checked_add_u64(*used_bytes, bytes, &next)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    if (next > max_bytes) return GAFIME_STATUS_OUT_OF_MEMORY;
    if (bytes == 0u) return GAFIME_STATUS_OK;
    const int status = cuda_status(cudaMalloc(reinterpret_cast<void**>(ptr), static_cast<size_t>(bytes)));
    if (status != GAFIME_STATUS_OK) return status;
    *used_bytes = next;
    return GAFIME_STATUS_OK;
}

template <typename T>
int copy_compact_semantic_rt_persistent(T* device, const std::vector<T>& host) {
    if (host.empty()) return GAFIME_STATUS_OK;
    if (device == nullptr || !allocation_fits_size_t(host.size(), sizeof(T))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return cuda_status(cudaMemcpy(
        device,
        host.data(),
        host.size() * sizeof(T),
        cudaMemcpyHostToDevice
    ));
}

void collect_compact_semantic_rt_slots(
    const std::vector<GafimeDecisionPathTerm>& terms,
    std::vector<uint32_t>* slots_out
) {
    slots_out->clear();
    slots_out->reserve(terms.size());
    for (const GafimeDecisionPathTerm& term : terms) {
        if (std::find(slots_out->begin(), slots_out->end(), term.feature) == slots_out->end()) {
            slots_out->push_back(term.feature);
        }
    }
}

int validate_compact_semantic_rt_input_domain(
    const gafime_cuda_v1::detail::CudaSemanticBankView& bank,
    const std::vector<uint32_t>& slots
) {
    if (slots.empty()) return GAFIME_STATUS_INVALID_ARGUMENT;
    uint64_t slot_bytes = 0u;
    uint64_t bytes = 0u;
    if (!checked_element_bytes(slots.size(), sizeof(uint32_t), &slot_bytes) ||
        !checked_add_u64(slot_bytes, sizeof(uint32_t), &bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    CompactSemanticRtTemporaryAllocation allocation;
    int status = allocation.allocate(bytes);
    if (status != GAFIME_STATUS_OK) return status;
    uint32_t* slots_device = allocation.as<uint32_t>();
    uint32_t* invalid_device = allocation.as<uint32_t>(slot_bytes);
    status = cuda_status(cudaMemcpy(
        slots_device, slots.data(), static_cast<size_t>(slot_bytes), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(invalid_device, 0, sizeof(uint32_t)));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256u;
        const uint64_t value_count = bank.rows * static_cast<uint64_t>(slots.size());
        const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(
            (value_count + threads - 1u) / threads, 65'535u));
        gafime_cuda_v1::rt_kernel::validate_semantic_region_input_domain_kernel<<<blocks, threads>>>(
            bank.columns, bank.rows, slots_device, static_cast<uint32_t>(slots.size()), invalid_device);
        status = cuda_status(cudaGetLastError());
    }
    uint32_t invalid = 0u;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(&invalid, invalid_device, sizeof(invalid), cudaMemcpyDeviceToHost));
    }
    if (status != GAFIME_STATUS_OK) return status;
    return invalid == 0u ? GAFIME_STATUS_OK : GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
/* A query owns this single-group GAS outright.  It deliberately does not use
 * the legacy device-state map: the query's immutable snapshot and the bank
 * lifetime retained by Rust are the cache/lifetime boundary.  The companion
 * grouped routine below owns its IAS/GAS chain under the same boundary. */
int prepare_compact_semantic_rt_single_group_geometry(
    CompactSemanticRtRegionQuery* query,
    const CompactSemanticRtOptixAllocationPlan& allocation
) {
    if (query == nullptr || query->optix_state == nullptr || query->plan.groups.size() != 1u ||
        query->plan.flat_boxes.size() != query->region_count || query->rows > UINT32_MAX ||
        allocation.geometry_mode != RtGeometryMode::CustomAabb || allocation.instanced ||
        allocation.triangle || allocation.gas_sizes.size() != 1u ||
        allocation.gas_output_offsets.size() != 1u || allocation.gas_output_offsets[0] != 0u) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    constexpr RtGeometryMode geometry_mode = RtGeometryMode::CustomAabb;
    int status = ensure_optix_program(*query->optix_state, geometry_mode);
    if (status != GAFIME_STATUS_OK) return status;
    RtOptixProgram& program = query->optix_state->program(geometry_mode);
    if (program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
        if (status != GAFIME_STATUS_OK) return status;
    }
    if (static_cast<size_t>(query->region_count) > program.box_capacity) {
        program.gas_valid = false;
    }
    status = ensure_device_capacity(
        &program.boxes_device, program.box_capacity, static_cast<size_t>(query->region_count));
    std::vector<OptixAabb> aabbs;
    aabbs.reserve(query->plan.flat_boxes.size());
    if (status == GAFIME_STATUS_OK) {
        for (const auto& box : query->plan.flat_boxes) {
            aabbs.push_back(make_rt_conservative_aabb(box));
        }
        if (aabbs.size() > program.aabb_capacity) program.gas_valid = false;
        status = ensure_device_capacity(&program.aabbs_device, program.aabb_capacity, aabbs.size());
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.params_device, program.params_capacity, size_t{1u});
    }
    if (status != GAFIME_STATUS_OK) return status;

    status = cuda_status(cudaMemcpy(
        program.boxes_device,
        query->plan.flat_boxes.data(),
        query->plan.flat_boxes.size() * sizeof(query->plan.flat_boxes[0]),
        cudaMemcpyHostToDevice
    ));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            program.aabbs_device, aabbs.data(), aabbs.size() * sizeof(aabbs[0]), cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) return status;

    CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(program.aabbs_device);
    uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives = query->region_count;
    build_input.customPrimitiveArray.flags = geometry_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1u;
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    const OptixAccelBufferSizes& sizes = allocation.gas_sizes[0];
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(
            &program.gas_temp_device, program.gas_temp_capacity, allocation.gas_temp_bytes);
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(
            &program.gas_output_device, program.gas_output_capacity, allocation.gas_output_bytes);
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixAccelBuild(
            program.context,
            program.stream,
            &options,
            &build_input,
            1u,
            reinterpret_cast<CUdeviceptr>(program.gas_temp_device),
            sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(program.gas_output_device),
            sizes.outputSizeInBytes,
            &program.gas_handle,
            nullptr,
            0u
        ));
    }
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    if (status != GAFIME_STATUS_OK) {
        program.gas_valid = false;
        program.gas_handle = 0;
        return status;
    }
    program.gas_valid = true;
    program.gas_signature = 1u;
    query->geometry_available = true;
    query->optix_geometry_mode = geometry_mode;
    query->optix_geometry_ready = true;
    return GAFIME_STATUS_OK;
}

/* Reuse the mature instanced GAS/IAS construction for the immutable compact
 * query snapshot.  The query already owns grouped point packing, exact-term
 * offsets, and result-ordinal map; only the legacy physical AS preparation is
 * borrowed here.  No global device-state cache participates. */
int prepare_compact_semantic_rt_grouped_geometry(
    CompactSemanticRtRegionQuery* query,
    const CompactSemanticRtOptixAllocationPlan& allocation
) {
    if (query == nullptr || query->optix_state == nullptr || query->plan.groups.empty() ||
        query->plan.groups.size() != query->group_count || query->rows > UINT32_MAX ||
        !allocation.instanced || allocation.gas_sizes.size() != query->plan.groups.size() ||
        allocation.gas_output_offsets.size() != query->plan.groups.size()) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    const RtGeometryMode geometry_mode = allocation.geometry_mode;
    if (geometry_mode != RtGeometryMode::Triangle2dInstanced &&
        geometry_mode != RtGeometryMode::CustomAabbInstanced) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    const bool triangle_mode = geometry_mode == RtGeometryMode::Triangle2dInstanced;
    int status = ensure_optix_program(*query->optix_state, geometry_mode);
    if (status != GAFIME_STATUS_OK) return status;
    RtOptixProgram& program = query->optix_state->program(geometry_mode);
    if (program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
        if (status != GAFIME_STATUS_OK) return status;
    }

    const size_t group_count = query->plan.groups.size();
    std::vector<OptixAabb> aabbs;
    std::vector<uint32_t> aabb_offsets(group_count, 0u);
    std::vector<uint32_t> aabb_counts(group_count, 0u);
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> vertices;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> indices;
    aabbs.reserve(triangle_mode ? 0u : query->region_count);
    vertices.reserve(triangle_mode ? static_cast<size_t>(query->region_count) * 4u : 0u);
    indices.reserve(triangle_mode ? static_cast<size_t>(query->region_count) * 2u : 0u);
    std::vector<uint32_t> vertex_offsets(group_count, 0u);
    std::vector<uint32_t> vertex_counts(group_count, 0u);
    std::vector<uint32_t> index_offsets(group_count, 0u);
    std::vector<uint32_t> index_counts(group_count, 0u);
    for (size_t group_idx = 0u; group_idx < group_count; ++group_idx) {
        const SemanticRtRegionGroup& group = query->plan.groups[group_idx];
        if (group.result_regions.empty() || group.boxes.size() != group.result_regions.size()) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        if (triangle_mode) {
            RtBoxPlan triangle_plan{};
            triangle_plan.dims = static_cast<uint32_t>(group.axes.size());
            triangle_plan.boxes = group.boxes;
            std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> group_vertices;
            std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> group_indices;
            build_rt_triangles(triangle_plan, group_vertices, group_indices);
            vertex_offsets[group_idx] = static_cast<uint32_t>(vertices.size());
            index_offsets[group_idx] = static_cast<uint32_t>(indices.size());
            vertex_counts[group_idx] = static_cast<uint32_t>(group_vertices.size());
            index_counts[group_idx] = static_cast<uint32_t>(group_indices.size());
            vertices.insert(vertices.end(), group_vertices.begin(), group_vertices.end());
            indices.insert(indices.end(), group_indices.begin(), group_indices.end());
        } else {
            aabb_offsets[group_idx] = static_cast<uint32_t>(aabbs.size());
            aabb_counts[group_idx] = static_cast<uint32_t>(group.boxes.size());
            for (const auto& box : group.boxes) aabbs.push_back(make_rt_conservative_aabb(box));
        }
    }
    if ((!triangle_mode && aabbs.size() != query->region_count) ||
        (triangle_mode && (vertices.size() != static_cast<size_t>(query->region_count) * 4u ||
                           indices.size() != static_cast<size_t>(query->region_count) * 2u))) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }

    if (static_cast<size_t>(query->region_count) > program.box_capacity) program.gas_valid = false;
    status = ensure_device_capacity(
        &program.boxes_device, program.box_capacity, static_cast<size_t>(query->region_count));
    if (status == GAFIME_STATUS_OK && !triangle_mode) {
        if (aabbs.size() > program.aabb_capacity) program.gas_valid = false;
        status = ensure_device_capacity(&program.aabbs_device, program.aabb_capacity, aabbs.size());
    }
    if (status == GAFIME_STATUS_OK && triangle_mode) {
        if (vertices.size() > program.vertex_capacity || indices.size() > program.index_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.vertices_device, program.vertex_capacity, vertices.size());
    }
    if (status == GAFIME_STATUS_OK && triangle_mode) {
        status = ensure_device_capacity(&program.indices_device, program.index_capacity, indices.size());
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.instances_device, program.instance_capacity, group_count);
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_capacity(&program.params_device, program.params_capacity, size_t{1u});
    }
    if (status != GAFIME_STATUS_OK) return status;

    status = cuda_status(cudaMemcpy(
        program.boxes_device,
        query->plan.flat_boxes.data(),
        query->plan.flat_boxes.size() * sizeof(query->plan.flat_boxes[0]),
        cudaMemcpyHostToDevice
    ));
    if (status == GAFIME_STATUS_OK && !triangle_mode) {
        status = cuda_status(cudaMemcpy(
            program.aabbs_device, aabbs.data(), aabbs.size() * sizeof(aabbs[0]), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK && triangle_mode) {
        status = cuda_status(cudaMemcpy(
            program.vertices_device, vertices.data(), vertices.size() * sizeof(vertices[0]), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK && triangle_mode) {
        status = cuda_status(cudaMemcpy(
            program.indices_device, indices.data(), indices.size() * sizeof(indices[0]), cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) return status;

    std::vector<OptixTraversableHandle> group_handles(group_count, 0u);
    const std::vector<size_t>& gas_output_offsets = allocation.gas_output_offsets;
    const std::vector<OptixAccelBufferSizes>& gas_sizes = allocation.gas_sizes;
    uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    status = ensure_device_bytes(
        &program.gas_temp_device, program.gas_temp_capacity, allocation.gas_temp_bytes);
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(
            &program.gas_output_device, program.gas_output_capacity, allocation.gas_output_bytes);
    }
    if (status != GAFIME_STATUS_OK) return status;

    for (size_t group_idx = 0u; group_idx < group_count; ++group_idx) {
        CUdeviceptr aabb_buffer = triangle_mode ? 0u : reinterpret_cast<CUdeviceptr>(
            program.aabbs_device + aabb_offsets[group_idx]);
        CUdeviceptr vertex_buffer = triangle_mode ? reinterpret_cast<CUdeviceptr>(
            program.vertices_device + vertex_offsets[group_idx]) : 0u;
        const CUdeviceptr index_buffer = triangle_mode ? reinterpret_cast<CUdeviceptr>(
            program.indices_device + index_offsets[group_idx]) : 0u;
        OptixBuildInput input = {};
        if (triangle_mode) {
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            input.triangleArray.vertexBuffers = &vertex_buffer;
            input.triangleArray.numVertices = vertex_counts[group_idx];
            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes =
                sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
            input.triangleArray.indexBuffer = index_buffer;
            input.triangleArray.numIndexTriplets = index_counts[group_idx];
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes =
                sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
            input.triangleArray.flags = geometry_flags;
            input.triangleArray.numSbtRecords = 1u;
        } else {
            input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
            input.customPrimitiveArray.numPrimitives = aabb_counts[group_idx];
            input.customPrimitiveArray.flags = geometry_flags;
            input.customPrimitiveArray.numSbtRecords = 1u;
        }
        OptixAccelBuildOptions options = {};
        options.buildFlags = triangle_mode ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_NONE;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        status = optix_status(optixAccelBuild(
            program.context,
            program.stream,
            &options,
            &input,
            1u,
            reinterpret_cast<CUdeviceptr>(program.gas_temp_device),
            gas_sizes[group_idx].tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(
                static_cast<char*>(program.gas_output_device) + gas_output_offsets[group_idx]),
            gas_sizes[group_idx].outputSizeInBytes,
            &group_handles[group_idx],
            nullptr,
            0u
        ));
        if (status != GAFIME_STATUS_OK) return status;
    }

    std::vector<OptixInstance> instances(group_count);
    for (size_t group_idx = 0u; group_idx < group_count; ++group_idx) {
        OptixInstance instance = {};
        const float z = static_cast<float>(group_idx) * 4.0f;
        const float transform[12] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, z,
        };
        std::memcpy(instance.transform, transform, sizeof(transform));
        instance.instanceId = static_cast<uint32_t>(group_idx);
        instance.visibilityMask = 1u;
        instance.sbtOffset = 0u;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = group_handles[group_idx];
        instances[group_idx] = instance;
    }
    status = cuda_status(cudaMemcpy(
        program.instances_device, instances.data(), instances.size() * sizeof(instances[0]), cudaMemcpyHostToDevice));
    if (status != GAFIME_STATUS_OK) return status;
    OptixBuildInput ias_input = {};
    ias_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(program.instances_device);
    ias_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());
    OptixAccelBuildOptions ias_options = {};
    ias_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ias_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    const OptixAccelBufferSizes& ias_sizes = allocation.ias_sizes;
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(
            &program.ias_temp_device, program.ias_temp_capacity, ias_sizes.tempSizeInBytes);
    }
    if (status == GAFIME_STATUS_OK) {
        status = ensure_device_bytes(&program.ias_output_device, program.ias_output_capacity, ias_sizes.outputSizeInBytes);
    }
    OptixTraversableHandle ias_handle = 0u;
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixAccelBuild(
            program.context,
            program.stream,
            &ias_options,
            &ias_input,
            1u,
            reinterpret_cast<CUdeviceptr>(program.ias_temp_device),
            ias_sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(program.ias_output_device),
            ias_sizes.outputSizeInBytes,
            &ias_handle,
            nullptr,
            0u
        ));
    }
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    if (status != GAFIME_STATUS_OK) {
        program.gas_valid = false;
        program.gas_handle = 0u;
        return status;
    }
    program.gas_handle = ias_handle;
    program.gas_signature = 2u;
    program.gas_valid = true;
    query->geometry_available = true;
    query->optix_geometry_mode = geometry_mode;
    query->optix_geometry_ready = true;
    return GAFIME_STATUS_OK;
}

int prepare_compact_semantic_rt_geometry(
    CompactSemanticRtRegionQuery* query,
    const CompactSemanticRtOptixAllocationPlan& allocation
) {
    if (query == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (allocation.geometry_mode == RtGeometryMode::CustomAabb) {
        return prepare_compact_semantic_rt_single_group_geometry(query, allocation);
    }
    return prepare_compact_semantic_rt_grouped_geometry(query, allocation);
}
#else
int prepare_compact_semantic_rt_geometry(CompactSemanticRtRegionQuery*) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}
#endif

bool compact_semantic_rt_add_planned_bytes(
    uint64_t count,
    size_t element_size,
    uint64_t* total
) {
    uint64_t bytes = 0u;
    return checked_element_bytes(count, element_size, &bytes) && checked_plan_add(total, bytes);
}

/* Small admission bound used before materializing descriptor copies or the
 * retained host plan.  It deliberately over-reserves the bounded grouping
 * and bin-construction shape; a later full plan replaces it with measured
 * capacities and queried OptiX buffers. */
bool compact_semantic_rt_prequery_host_reservation(
    const GafimeSemanticRtRegionQueryDesc* desc,
    uint64_t* bytes_out
) {
    if (desc == nullptr || bytes_out == nullptr) return false;
    const uint64_t regions = desc->region_count;
    const uint64_t terms = desc->term_count;
    const uint64_t groups = std::min<uint64_t>(
        regions, GAFIME_CUDA_RT_REGION_QUERY_MAX_GROUPS);
    uint64_t total = sizeof(CompactSemanticRtRegionQuery);
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    if (!checked_plan_add(&total, sizeof(RtDeviceState))) return false;
#endif
    const bool ok =
        /* primary/paired physical copies and slot-domain staging */
        compact_semantic_rt_add_planned_bytes(terms, sizeof(GafimeDecisionPathTerm), &total) &&
        compact_semantic_rt_add_planned_bytes(terms, sizeof(GafimeDecisionPathTerm), &total) &&
        compact_semantic_rt_add_planned_bytes(terms, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(terms, sizeof(uint32_t), &total) &&
        /* retained group and flattened exact-conjunction snapshots */
        compact_semantic_rt_add_planned_bytes(regions, sizeof(SemanticRtRegionGroup), &total) &&
        compact_semantic_rt_add_planned_bytes(regions, sizeof(SemanticRtRegionGroupBuild), &total) &&
        compact_semantic_rt_add_planned_bytes(terms, sizeof(GafimeDecisionPathTerm), &total) &&
        compact_semantic_rt_add_planned_bytes(terms, sizeof(GafimeDecisionPathTerm), &total) &&
        compact_semantic_rt_add_planned_bytes(regions + groups + 2u, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(regions + groups + 2u, sizeof(uint32_t), &total) &&
        /* Flattened boxes, retained group boxes, and one bounded triangle
         * RtBoxPlan staging copy may coexist during create. */
        compact_semantic_rt_add_planned_bytes(
            regions * 3u, sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox), &total) &&
        compact_semantic_rt_add_planned_bytes(regions * 2u, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(groups * 6u, sizeof(uint32_t), &total) &&
        /* Every region can conservatively appear in every fixed x-bin while
         * original-predicate checks still preserve exactness. */
        compact_semantic_rt_add_planned_bytes(
            regions * kSemanticRtRegionSmBinCount * 2u, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(
            groups * (kSemanticRtRegionSmBinCount + 1u), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(
            groups * kSemanticRtRegionSmBinCount, sizeof(std::vector<uint32_t>), &total) &&
        compact_semantic_rt_add_planned_bytes(groups * 2u, sizeof(float), &total);
    if (!ok) return false;
    *bytes_out = total;
    return true;
}

bool compact_semantic_rt_generic_device_reservation(
    uint64_t rows,
    uint32_t region_count,
    uint64_t words_per_region,
    uint64_t point_count,
    bool has_paired,
    bool direct_first_hit,
    const std::vector<GafimeDecisionPathTerm>& primary_terms,
    const CompactSemanticRtRegionPlan& plan,
    uint64_t* bytes_out
) {
    if (bytes_out == nullptr) return false;
    uint64_t membership_words = 0u;
    if (!checked_mul_u64(words_per_region, region_count, &membership_words)) return false;
    uint64_t total = 0u;
    const bool ok =
        compact_semantic_rt_add_planned_bytes(
            primary_terms.size(), sizeof(GafimeDecisionPathTerm), &total) &&
        (!has_paired || compact_semantic_rt_add_planned_bytes(
            primary_terms.size(), sizeof(GafimeDecisionPathTerm), &total)) &&
        compact_semantic_rt_add_planned_bytes(
            static_cast<uint64_t>(region_count) + 1u, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(
            plan.flat_exact_terms.size(), sizeof(GafimeDecisionPathTerm), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.flat_exact_offsets.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.group_path_offsets.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.group_region_ids.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.group_primary_axes.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.group_paired_axes.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.group_dims.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.bin_offsets.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.bin_candidates.size(), sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.bin_lo.size(), sizeof(float), &total) &&
        compact_semantic_rt_add_planned_bytes(plan.bin_inv_span.size(), sizeof(float), &total) &&
        compact_semantic_rt_add_planned_bytes(point_count, sizeof(float), &total) &&
        (!has_paired || compact_semantic_rt_add_planned_bytes(point_count, sizeof(float), &total)) &&
        compact_semantic_rt_add_planned_bytes(words_per_region, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(words_per_region, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(rows, sizeof(uint32_t), &total) &&
        compact_semantic_rt_add_planned_bytes(
            region_count, sizeof(GafimeSemanticRtRegionExactStats), &total) &&
        (direct_first_hit || compact_semantic_rt_add_planned_bytes(
            membership_words, sizeof(uint32_t), &total)) &&
        (direct_first_hit || !has_paired || compact_semantic_rt_add_planned_bytes(
            membership_words, sizeof(uint32_t), &total));
    if (!ok) return false;
    *bytes_out = total;
    return true;
}

bool compact_semantic_rt_full_host_reservation(
    const CompactSemanticRtRegionPlan& plan,
    const std::vector<GafimeDecisionPathTerm>& primary_terms,
    const std::vector<GafimeDecisionPathTerm>& paired_terms,
    bool has_paired,
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    const CompactSemanticRtOptixAllocationPlan* optix_plan,
#endif
    uint64_t* bytes_out
) {
    if (bytes_out == nullptr) return false;
    uint64_t total = sizeof(CompactSemanticRtRegionQuery);
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    if (!checked_plan_add(&total, sizeof(RtDeviceState))) return false;
#endif
    uint64_t term_bytes = 0u;
    uint64_t paired_bytes = 0u;
    if (!semantic_rt_vector_capacity_bytes(primary_terms, &term_bytes) ||
        !semantic_rt_vector_capacity_bytes(paired_terms, &paired_bytes) ||
        !checked_plan_add(&total, plan.host_build_peak_bytes) ||
        !checked_plan_add(&total, term_bytes) ||
        !checked_plan_add(&total, paired_bytes)) {
        return false;
    }
    /* `collect_compact_semantic_rt_slots` reserves at most one entry per
     * term for each view.  Keep both host vectors and the larger sequential
     * device validator staging inside the create reservation. */
    if (!compact_semantic_rt_add_planned_bytes(primary_terms.size(), sizeof(uint32_t), &total) ||
        (has_paired && !compact_semantic_rt_add_planned_bytes(
            paired_terms.size(), sizeof(uint32_t), &total))) {
        return false;
    }
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    if (optix_plan != nullptr && !checked_plan_add(&total, optix_plan->host_construction_bytes)) {
        return false;
    }
#endif
    *bytes_out = total;
    return true;
}

bool compact_semantic_rt_input_validation_reservation(
    const std::vector<GafimeDecisionPathTerm>& primary_terms,
    const std::vector<GafimeDecisionPathTerm>& paired_terms,
    bool has_paired,
    uint64_t* bytes_out
) {
    if (bytes_out == nullptr) return false;
    uint64_t primary_bytes = 0u;
    uint64_t paired_bytes = 0u;
    uint64_t total = 0u;
    if (!checked_element_bytes(primary_terms.size(), sizeof(uint32_t), &primary_bytes) ||
        !checked_add_u64(primary_bytes, sizeof(uint32_t), &primary_bytes) ||
        !checked_element_bytes(paired_terms.size(), sizeof(uint32_t), &paired_bytes) ||
        !checked_add_u64(paired_bytes, sizeof(uint32_t), &paired_bytes)) {
        return false;
    }
    total = has_paired ? std::max(primary_bytes, paired_bytes) : primary_bytes;
    *bytes_out = total;
    return true;
}

int create_compact_semantic_rt_region_query(
    const GafimeSemanticRtRegionQueryDesc* desc,
    GafimeGpuSemanticRegionQuery* query_out,
    uint64_t* persistent_bytes_out
) {
    if (query_out == nullptr || persistent_bytes_out == nullptr ||
        !gafime_gpu_abi::naturally_aligned(query_out) ||
        !gafime_gpu_abi::naturally_aligned(persistent_bytes_out)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *query_out = nullptr;
    *persistent_bytes_out = 0u;

    int status = validate_compact_semantic_rt_query_shape(desc);
    if (status != GAFIME_STATUS_OK) return status;
    uint64_t prequery_reservation = 0u;
    if (!compact_semantic_rt_prequery_host_reservation(desc, &prequery_reservation)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    *persistent_bytes_out = prequery_reservation;
    if (prequery_reservation > desc->max_persistent_bytes) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }

    std::vector<GafimeDecisionPathTerm> primary_terms;
    std::vector<GafimeDecisionPathTerm> paired_terms;
    status = validate_compact_semantic_rt_query_desc(desc, &primary_terms, &paired_terms);
    if (status != GAFIME_STATUS_OK) return status;
    gafime_cuda_v1::detail::CudaSemanticBankView primary{};
    gafime_cuda_v1::detail::CudaSemanticBankView paired{};
    status = validate_compact_semantic_rt_query_banks(desc, primary_terms, paired_terms, &primary, &paired);
    if (status != GAFIME_STATUS_OK) return status;

    CompactSemanticRtRegionPlan plan{};
    status = build_compact_semantic_rt_region_plan(desc, primary_terms, paired_terms, &plan);
    if (status != GAFIME_STATUS_OK) return status;
    if (plan.groups.empty() || plan.groups.size() > GAFIME_CUDA_RT_REGION_QUERY_MAX_GROUPS ||
        plan.flat_exact_terms.size() != primary_terms.size()) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    uint64_t point_count = 0u;
    uint64_t grouped_rows = 0u;
    const uint64_t words_per_region = (primary.rows + 31u) / 32u;
    uint64_t membership_words = 0u;
    if (!checked_mul_u64(primary.rows, static_cast<uint64_t>(plan.groups.size()), &grouped_rows) ||
        !checked_mul_u64(grouped_rows, 3u, &point_count) ||
        !checked_mul_u64(words_per_region, desc->region_count, &membership_words) ||
        words_per_region > UINT32_MAX) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const bool has_paired = desc->paired_bank != nullptr;
    ScopedCudaDevice device(primary.device_id);
    if (device.status() != cudaSuccess) return cuda_status(device.status());

    const bool require_rt =
        desc->flags == GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT;
    const bool force_sm =
        desc->flags == GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM;
    const bool force_sm_exhaustive =
        desc->flags == GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE;
    const bool direct_candidate = require_rt && plan.groups.size() == 1u && plan.first_hit_eligible;
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    CompactSemanticRtOptixAllocationPlan optix_plan{};
    if (require_rt) {
        /* This temporary state builds only opaque vendor pipeline state and
         * queries allocation sizes.  It intentionally has no SBT or CUDA
         * query buffers, and is destroyed before real query allocation. */
        {
            RtDeviceState planning_state(primary.device_id);
            status = plan_compact_semantic_rt_optix_allocations(
                planning_state,
                primary.device_id,
                primary.rows,
                desc->region_count,
                plan,
                &optix_plan
            );
        }
        if (status != GAFIME_STATUS_OK) return status;
    }
#else
    if (require_rt) return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif

    uint64_t generic_device_bytes = 0u;
    uint64_t host_reservation = 0u;
    uint64_t validation_bytes = 0u;
    if (!compact_semantic_rt_generic_device_reservation(
            primary.rows,
            desc->region_count,
            words_per_region,
            point_count,
            has_paired,
            direct_candidate,
            primary_terms,
            plan,
            &generic_device_bytes) ||
        !compact_semantic_rt_input_validation_reservation(
            primary_terms, paired_terms, has_paired, &validation_bytes) ||
        !compact_semantic_rt_full_host_reservation(
            plan,
            primary_terms,
            paired_terms,
            has_paired,
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
            require_rt ? &optix_plan : nullptr,
#endif
            &host_reservation)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    uint64_t non_optix_reservation = 0u;
    if (!checked_add_u64(host_reservation, validation_bytes, &non_optix_reservation) ||
        !checked_add_u64(non_optix_reservation, generic_device_bytes, &non_optix_reservation)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    /* The first gate was intentionally conservative enough for host descriptor
     * validation and planning.  Retain that reservation in the final public
     * bound even if measured vector capacities are smaller, then add every
     * RT-owned explicit CUDA/OptiX buffer separately. */
    uint64_t full_reservation = std::max(non_optix_reservation, prequery_reservation);
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    if (!checked_add_u64(
            full_reservation, require_rt ? optix_plan.explicit_bytes : 0u, &full_reservation)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
#endif
    *persistent_bytes_out = full_reservation;
    if (full_reservation > desc->max_persistent_bytes) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }

    std::vector<uint32_t> primary_slots;
    std::vector<uint32_t> paired_slots;
    collect_compact_semantic_rt_slots(primary_terms, &primary_slots);
    status = validate_compact_semantic_rt_input_domain(primary, primary_slots);
    if (status == GAFIME_STATUS_OK && has_paired) {
        collect_compact_semantic_rt_slots(paired_terms, &paired_slots);
        status = validate_compact_semantic_rt_input_domain(paired, paired_slots);
    }
    if (status != GAFIME_STATUS_OK) return status;

    std::unique_ptr<CompactSemanticRtRegionQuery> query(
        new CompactSemanticRtRegionQuery(primary.device_id));
    query->primary_bank = desc->primary_bank;
    query->paired_bank = desc->paired_bank;
    query->primary_columns = primary.columns;
    query->paired_columns = has_paired ? paired.columns : nullptr;
    query->rows = primary.rows;
    query->region_count = desc->region_count;
    query->group_count = static_cast<uint32_t>(plan.groups.size());
    query->words_per_region = static_cast<uint32_t>(words_per_region);
    query->has_paired = has_paired;
    query->force_sm = force_sm;
    query->force_sm_exhaustive = force_sm_exhaustive;
    query->require_rt = require_rt;
    query->direct_first_hit = direct_candidate;
    query->plan = std::move(plan);

    uint64_t used = 0u;
    auto allocate = [&](auto** pointer, uint64_t count) -> int {
        using Element = std::remove_pointer_t<
            std::remove_reference_t<decltype(*pointer)>>;
        return allocate_compact_semantic_rt_persistent<Element>(
            pointer, count, generic_device_bytes, &used);
    };
    status = allocate(&query->source_primary_terms_device, primary_terms.size());
    if (status == GAFIME_STATUS_OK && has_paired) status = allocate(
        &query->source_paired_terms_device, paired_terms.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->source_offsets_device, static_cast<uint64_t>(desc->region_count) + 1u);
    if (status == GAFIME_STATUS_OK) status = allocate(&query->exact_terms_device, query->plan.flat_exact_terms.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->exact_offsets_device, query->plan.flat_exact_offsets.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->group_path_offsets_device, query->plan.group_path_offsets.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->group_region_ids_device, query->plan.group_region_ids.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->group_primary_axes_device, query->plan.group_primary_axes.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->group_paired_axes_device, query->plan.group_paired_axes.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->group_dims_device, query->plan.group_dims.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->bin_offsets_device, query->plan.bin_offsets.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->bin_candidates_device, query->plan.bin_candidates.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->bin_lo_device, query->plan.bin_lo.size());
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->bin_inv_span_device, query->plan.bin_inv_span.size());
    if (status == GAFIME_STATUS_OK) status = allocate(&query->primary_points_device, point_count);
    if (status == GAFIME_STATUS_OK && has_paired) status = allocate(&query->paired_points_device, point_count);
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->label_zero_words_device, words_per_region);
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->label_one_words_device, words_per_region);
    if (status == GAFIME_STATUS_OK) status = allocate(
        &query->direct_region_ordinals_device, primary.rows);
    if (status == GAFIME_STATUS_OK) status = allocate(&query->stats_device, desc->region_count);

    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->source_primary_terms_device, primary_terms);
    if (status == GAFIME_STATUS_OK && has_paired) status = copy_compact_semantic_rt_persistent(
        query->source_paired_terms_device, paired_terms);
    if (status == GAFIME_STATUS_OK) {
        const size_t bytes = (static_cast<size_t>(desc->region_count) + 1u) * sizeof(uint32_t);
        status = cuda_status(cudaMemcpy(
            query->source_offsets_device, desc->region_offsets, bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->exact_terms_device, query->plan.flat_exact_terms);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->exact_offsets_device, query->plan.flat_exact_offsets);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->group_path_offsets_device, query->plan.group_path_offsets);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->group_region_ids_device, query->plan.group_region_ids);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->group_primary_axes_device, query->plan.group_primary_axes);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->group_paired_axes_device, query->plan.group_paired_axes);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->group_dims_device, query->plan.group_dims);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->bin_offsets_device, query->plan.bin_offsets);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->bin_candidates_device, query->plan.bin_candidates);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->bin_lo_device, query->plan.bin_lo);
    if (status == GAFIME_STATUS_OK) status = copy_compact_semantic_rt_persistent(
        query->bin_inv_span_device, query->plan.bin_inv_span);
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256u;
        const uint32_t row_blocks = static_cast<uint32_t>((primary.rows + threads - 1u) / threads);
        const dim3 grid(row_blocks, query->group_count);
        gafime_cuda_v1::rt_kernel::pack_grouped_decision_path_points_kernel<<<grid, threads>>>(
            primary.columns,
            primary.rows,
            query->group_primary_axes_device,
            query->group_dims_device,
            query->group_count,
            3u,
            query->primary_points_device
        );
        status = cuda_status(cudaGetLastError());
        if (status == GAFIME_STATUS_OK && has_paired) {
            gafime_cuda_v1::rt_kernel::pack_grouped_decision_path_points_kernel<<<grid, threads>>>(
                paired.columns,
                primary.rows,
                query->group_paired_axes_device,
                query->group_dims_device,
                query->group_count,
                3u,
                query->paired_points_device
            );
            status = cuda_status(cudaGetLastError());
        }
        if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    }

    if (status == GAFIME_STATUS_OK && query->require_rt) {
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
        status = prepare_compact_semantic_rt_geometry(query.get(), optix_plan);
#else
        status = GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
    }
    if (status == GAFIME_STATUS_OK && !query->direct_first_hit) {
        status = allocate(&query->primary_membership_words_device, membership_words);
    }
    if (status == GAFIME_STATUS_OK && !query->direct_first_hit && has_paired) {
        status = allocate(&query->paired_membership_words_device, membership_words);
    }
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(query->reset());
        return status;
    }
    if (used != generic_device_bytes) {
        static_cast<void>(query->reset());
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    query->persistent_bytes = full_reservation;
    *persistent_bytes_out = full_reservation;
    *query_out = static_cast<GafimeGpuSemanticRegionQuery>(query.release());
    return GAFIME_STATUS_OK;
}

struct CompactSemanticRtLabelPlan {
    bool enabled = false;
    uint64_t count = 0u;
    uint64_t zero_count = 0u;
    uint64_t one_count = 0u;
    uint64_t staging_bytes = 0u;
};

int validate_compact_semantic_rt_execute(
    const CompactSemanticRtRegionQuery& query,
    const GafimeSemanticRtRegionExecuteDesc* desc,
    const GafimeSemanticRtRegionStatsTable* stats_out,
    CompactSemanticRtLabelPlan* labels_out
) {
    if (desc == nullptr || labels_out == nullptr ||
        !gafime_gpu_abi::naturally_aligned(desc)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const int table_header_status = validate_compact_semantic_rt_stats_table_header(stats_out);
    if (table_header_status != GAFIME_STATUS_OK) return table_header_status;
    if (!compact_semantic_rt_abi_compatible(desc->abi_version, desc->struct_size, sizeof(*desc))) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    if (!compact_semantic_rt_reserved_zero(desc->reserved) ||
        !compact_semantic_rt_reserved_zero(stats_out->reserved) ||
        (desc->statistic_mask & ~GAFIME_SEMANTIC_RT_REGION_STAT_MASK_ALL) != 0u ||
        (desc->finalizer_mask & ~GAFIME_SEMANTIC_RT_REGION_FINALIZE_MASK_ALL) != 0u ||
        desc->statistic_mask == 0u || stats_out->capacity < query.region_count ||
        stats_out->records == nullptr || !gafime_gpu_abi::naturally_aligned(stats_out->records) ||
        !gafime_gpu_abi::fits_host_bytes(
            stats_out->capacity, sizeof(GafimeSemanticRtRegionExactStats))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t occupancy = GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY;
    const uint32_t paired = GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED;
    const uint32_t labeled = GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED;
    if (((desc->finalizer_mask & GAFIME_SEMANTIC_RT_REGION_FINALIZE_OCCUPANCY) != 0u &&
         (desc->statistic_mask & occupancy) == 0u) ||
        ((desc->finalizer_mask & (GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_AGREEMENT |
                                  GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_IOU)) != 0u &&
         (desc->statistic_mask & paired) == 0u) ||
        ((desc->finalizer_mask & GAFIME_SEMANTIC_RT_REGION_FINALIZE_LABELED_GINI_GAIN) != 0u &&
         (desc->statistic_mask & labeled) == 0u) ||
        ((desc->statistic_mask & paired) != 0u && !query.has_paired)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    *labels_out = {};
    if ((desc->statistic_mask & labeled) == 0u) {
        return GAFIME_STATUS_OK;
    }
    const GafimeSemanticRtBinaryLabelContext& labels = desc->labels;
    if (!compact_semantic_rt_abi_compatible(labels.abi_version, labels.struct_size, sizeof(labels)) ||
        !compact_semantic_rt_reserved_zero(labels.reserved) || labels.count > query.rows ||
        (labels.count != 0u &&
         (labels.row_indices == nullptr || labels.values == nullptr ||
          !gafime_gpu_abi::naturally_aligned(labels.row_indices) ||
          !gafime_gpu_abi::naturally_aligned(labels.values) ||
          !gafime_gpu_abi::fits_host_bytes(labels.count, sizeof(uint64_t)) ||
          !gafime_gpu_abi::fits_host_bytes(labels.count, sizeof(uint8_t))))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    /* A labeled request with an explicitly empty LabelSet is semantically
     * present, not MissingLabels: finalization must observe support zero and
     * return InsufficientSupport.  There are no host arrays to inspect. */
    if (labels.count == 0u) {
        labels_out->enabled = true;
        return GAFIME_STATUS_OK;
    }
    uint64_t previous = 0u;
    for (uint64_t index = 0u; index < labels.count; ++index) {
        const uint64_t row = labels.row_indices[index];
        const uint8_t value = labels.values[index];
        if (row >= query.rows || (index != 0u && row <= previous) || value > 1u) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        previous = row;
        if (value == 0u) ++labels_out->zero_count;
        else ++labels_out->one_count;
    }
    uint64_t row_bytes = 0u;
    if (!checked_element_bytes(labels.count, sizeof(uint64_t), &row_bytes) ||
        !checked_add_u64(row_bytes, labels.count, &labels_out->staging_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    labels_out->enabled = true;
    labels_out->count = labels.count;
    return GAFIME_STATUS_OK;
}

int validate_compact_semantic_rt_snapshot(const CompactSemanticRtRegionQuery& query) {
    gafime_cuda_v1::detail::CudaSemanticBankView primary{};
    int status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(query.primary_bank, &primary);
    if (status != GAFIME_STATUS_OK) return status;
    if (primary.columns != query.primary_columns || primary.rows != query.rows ||
        primary.device_id != query.device_id) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (!query.has_paired) return GAFIME_STATUS_OK;
    gafime_cuda_v1::detail::CudaSemanticBankView paired{};
    status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(query.paired_bank, &paired);
    if (status != GAFIME_STATUS_OK) return status;
    return paired.columns == query.paired_columns && paired.rows == query.rows &&
            paired.device_id == query.device_id
        ? GAFIME_STATUS_OK
        : GAFIME_STATUS_INVALID_ARGUMENT;
}

int stage_compact_semantic_rt_labels(
    CompactSemanticRtRegionQuery& query,
    const GafimeSemanticRtRegionExecuteDesc& desc,
    const CompactSemanticRtLabelPlan& labels,
    CompactSemanticRtTemporaryAllocation* staging_out
) {
    if (staging_out == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    const uint64_t word_bytes = static_cast<uint64_t>(query.words_per_region) * sizeof(uint32_t);
    int status = cuda_status(cudaMemset(query.label_zero_words_device, 0, static_cast<size_t>(word_bytes)));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(query.label_one_words_device, 0, static_cast<size_t>(word_bytes)));
    }
    if (!labels.enabled || labels.count == 0u || status != GAFIME_STATUS_OK) return status;
    status = staging_out->allocate(labels.staging_bytes);
    if (status != GAFIME_STATUS_OK) return status;
    const uint64_t row_bytes = labels.count * sizeof(uint64_t);
    uint64_t* row_indices = staging_out->as<uint64_t>();
    uint8_t* values = staging_out->as<uint8_t>(row_bytes);
    status = cuda_status(cudaMemcpy(
        row_indices, desc.labels.row_indices, static_cast<size_t>(row_bytes), cudaMemcpyHostToDevice));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            values, desc.labels.values, static_cast<size_t>(labels.count), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256u;
        const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(
            (labels.count + threads - 1u) / threads, 65'535u));
        gafime_cuda_v1::rt_kernel::semantic_region_binary_label_masks_kernel<<<blocks, threads>>>(
            row_indices,
            values,
            labels.count,
            query.label_zero_words_device,
            query.label_one_words_device
        );
        status = cuda_status(cudaGetLastError());
    }
    return status;
}

int run_compact_semantic_rt_sm(
    CompactSemanticRtRegionQuery& query,
    uint32_t statistic_mask,
    uint32_t finalizer_mask,
    const CompactSemanticRtLabelPlan& labels
) {
    if (query.primary_membership_words_device == nullptr ||
        (query.has_paired && query.paired_membership_words_device == nullptr)) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    uint64_t membership_words = 0u;
    uint64_t membership_bytes = 0u;
    if (!checked_mul_u64(query.region_count, query.words_per_region, &membership_words) ||
        !checked_element_bytes(membership_words, sizeof(uint32_t), &membership_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    int status = cuda_status(cudaMemset(query.stats_device, 0,
        static_cast<size_t>(query.region_count) * sizeof(GafimeSemanticRtRegionExactStats)));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(query.primary_membership_words_device, 0,
            static_cast<size_t>(membership_bytes)));
    }
    if (status == GAFIME_STATUS_OK && query.has_paired) {
        status = cuda_status(cudaMemset(query.paired_membership_words_device, 0,
            static_cast<size_t>(membership_bytes)));
    }
    constexpr uint32_t threads = 256u;
    const uint32_t row_blocks = static_cast<uint32_t>((query.rows + threads - 1u) / threads);
    if (status == GAFIME_STATUS_OK && query.force_sm_exhaustive) {
        const dim3 grid(query.region_count, row_blocks);
        gafime_cuda_v1::rt_kernel::semantic_region_membership_masks_sm_kernel<<<grid, threads>>>(
            query.primary_columns,
            query.has_paired ? query.paired_columns : nullptr,
            query.rows,
            query.source_primary_terms_device,
            query.has_paired ? query.source_paired_terms_device : nullptr,
            query.source_offsets_device,
            query.region_count,
            query.words_per_region,
            query.primary_membership_words_device,
            query.has_paired ? query.paired_membership_words_device : nullptr
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK && !query.force_sm_exhaustive) {
        const dim3 grid(row_blocks, query.group_count);
        gafime_cuda_v1::rt_kernel::semantic_region_membership_masks_binned_sm_kernel<<<grid, threads>>>(
            query.primary_points_device,
            query.has_paired ? query.paired_points_device : nullptr,
            query.rows,
            query.exact_terms_device,
            query.exact_offsets_device,
            query.group_path_offsets_device,
            query.group_region_ids_device,
            query.bin_offsets_device,
            query.bin_candidates_device,
            query.bin_lo_device,
            query.bin_inv_span_device,
            query.group_count,
            3u,
            static_cast<uint32_t>(query.rows * 3u),
            query.words_per_region,
            query.primary_membership_words_device,
            query.has_paired ? query.paired_membership_words_device : nullptr
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        gafime_cuda_v1::rt_kernel::reduce_semantic_region_membership_masks_kernel<<<
            query.region_count, threads>>>(
            query.primary_membership_words_device,
            query.has_paired ? query.paired_membership_words_device : nullptr,
            query.label_zero_words_device,
            query.label_one_words_device,
            query.rows,
            query.region_count,
            query.words_per_region,
            statistic_mask,
            labels.zero_count,
            labels.one_count,
            query.stats_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        const uint32_t blocks = (query.region_count + threads - 1u) / threads;
        gafime_cuda_v1::rt_kernel::finalize_semantic_region_stats_kernel<<<blocks, threads>>>(
            query.region_count,
            statistic_mask,
            finalizer_mask,
            query.rows,
            labels.zero_count,
            labels.one_count,
            query.stats_device
        );
        status = cuda_status(cudaGetLastError());
    }
    return status;
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
int run_compact_semantic_rt_optix(
    CompactSemanticRtRegionQuery& query,
    uint32_t statistic_mask,
    uint32_t finalizer_mask,
    const CompactSemanticRtLabelPlan& labels
) {
    if (!query.geometry_available || query.optix_state == nullptr ||
        !query.optix_geometry_ready) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    RtOptixProgram& program = query.optix_state->program(query.optix_geometry_mode);
    if (!program.gas_valid || program.gas_handle == 0u || program.params_device == nullptr ||
        program.stream == nullptr) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    uint64_t membership_words = 0u;
    uint64_t membership_bytes = 0u;
    if (!query.direct_first_hit &&
        (!checked_mul_u64(query.region_count, query.words_per_region, &membership_words) ||
         !checked_element_bytes(membership_words, sizeof(uint32_t), &membership_bytes))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    int status = cuda_status(cudaMemsetAsync(
        query.stats_device,
        0,
        static_cast<size_t>(query.region_count) * sizeof(GafimeSemanticRtRegionExactStats),
        program.stream
    ));
    if (status == GAFIME_STATUS_OK && query.direct_first_hit) {
        status = cuda_status(cudaMemsetAsync(
            query.direct_region_ordinals_device,
            0,
            static_cast<size_t>(query.rows) * sizeof(uint32_t),
            program.stream));
    }
    if (status == GAFIME_STATUS_OK && !query.direct_first_hit) {
        status = cuda_status(cudaMemsetAsync(
            query.primary_membership_words_device, 0, static_cast<size_t>(membership_bytes), program.stream));
    }
    if (status == GAFIME_STATUS_OK && !query.direct_first_hit && query.has_paired) {
        status = cuda_status(cudaMemsetAsync(
            query.paired_membership_words_device, 0, static_cast<size_t>(membership_bytes), program.stream));
    }
    if (status != GAFIME_STATUS_OK) return status;

    GafimeRtParams params = {};
    params.handle = program.gas_handle;
    params.points_xyz = query.primary_points_device;
    params.boxes = program.boxes_device;
    params.membership_words = query.direct_first_hit ? nullptr : query.primary_membership_words_device;
    params.rows = static_cast<uint32_t>(query.rows);
    params.path_count = query.region_count;
    params.geometry_mode = static_cast<uint32_t>(query.optix_geometry_mode);
    params.words_per_path = query.words_per_region;
    params.group_path_offsets = query.group_path_offsets_device;
    params.group_count = query.group_count;
    params.point_group_stride = static_cast<uint32_t>(query.rows * 3u);
    params.point_stride = 3u;
    params.direct_first_hit = query.direct_first_hit ? 1u : 0u;
    params.semantic_exact_terms = query.exact_terms_device;
    params.semantic_exact_offsets = query.exact_offsets_device;
    params.semantic_region_ids = query.group_region_ids_device;
    params.semantic_counterpart_points_xyz = query.has_paired ? query.paired_points_device : nullptr;
    params.semantic_label_zero_words = query.label_zero_words_device;
    params.semantic_label_one_words = query.label_one_words_device;
    params.semantic_stats = query.stats_device;
    params.semantic_direct_region_ordinals = query.direct_first_hit
        ? query.direct_region_ordinals_device
        : nullptr;
    params.semantic_statistic_mask = statistic_mask;
    params.semantic_direct_stats = query.direct_first_hit ? 1u : 0u;
    params.semantic_view = 0u;
    status = cuda_status(cudaMemcpyAsync(
        program.params_device, &params, sizeof(params), cudaMemcpyHostToDevice, program.stream));
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixLaunch(
            program.pipeline,
            program.stream,
            reinterpret_cast<CUdeviceptr>(program.params_device),
            sizeof(params),
            &program.sbt,
            static_cast<uint32_t>(query.rows),
            query.group_count,
            1u
        ));
    }
    if (status == GAFIME_STATUS_OK &&
        (statistic_mask & GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED) != 0u) {
        params.points_xyz = query.paired_points_device;
        params.semantic_counterpart_points_xyz = query.primary_points_device;
        params.membership_words = query.direct_first_hit ? nullptr : query.paired_membership_words_device;
        params.semantic_direct_region_ordinals = nullptr;
        params.semantic_view = 1u;
        status = cuda_status(cudaMemcpyAsync(
            program.params_device, &params, sizeof(params), cudaMemcpyHostToDevice, program.stream));
        if (status == GAFIME_STATUS_OK) {
            status = optix_status(optixLaunch(
                program.pipeline,
                program.stream,
                reinterpret_cast<CUdeviceptr>(program.params_device),
                sizeof(params),
                &program.sbt,
                static_cast<uint32_t>(query.rows),
                query.group_count,
                1u
            ));
        }
    }
    constexpr uint32_t threads = 256u;
    if (status == GAFIME_STATUS_OK && !query.direct_first_hit) {
        gafime_cuda_v1::rt_kernel::reduce_semantic_region_membership_masks_kernel<<<
            query.region_count, threads, 0, program.stream>>>(
            query.primary_membership_words_device,
            query.has_paired ? query.paired_membership_words_device : nullptr,
            query.label_zero_words_device,
            query.label_one_words_device,
            query.rows,
            query.region_count,
            query.words_per_region,
            statistic_mask,
            labels.zero_count,
            labels.one_count,
            query.stats_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        const uint32_t blocks = (query.region_count + threads - 1u) / threads;
        gafime_cuda_v1::rt_kernel::finalize_semantic_region_stats_kernel<<<
            blocks, threads, 0, program.stream>>>(
            query.region_count,
            statistic_mask,
            finalizer_mask,
            query.rows,
            labels.zero_count,
            labels.one_count,
            query.stats_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaStreamSynchronize(program.stream));
    if (status != GAFIME_STATUS_OK) {
        program.gas_valid = false;
        program.gas_handle = 0u;
    }
    return status;
}
#else
int run_compact_semantic_rt_optix(
    CompactSemanticRtRegionQuery&,
    uint32_t,
    uint32_t,
    const CompactSemanticRtLabelPlan&
) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}
#endif

int execute_compact_semantic_rt_region_query(
    GafimeGpuSemanticRegionQuery handle,
    const GafimeSemanticRtRegionExecuteDesc* desc,
    GafimeSemanticRtRegionStatsTable* stats_out,
    uint64_t* temporary_peak_out
) {
    if (temporary_peak_out == nullptr || !gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *temporary_peak_out = 0u;
    const int table_header_status = validate_compact_semantic_rt_stats_table_header(stats_out);
    if (table_header_status != GAFIME_STATUS_OK) return table_header_status;
    clear_compact_semantic_rt_stats_table(stats_out);
    CompactSemanticRtRegionQuery* query = compact_semantic_rt_query_from_handle(handle);
    if (query == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    CompactSemanticRtLabelPlan labels{};
    int status = validate_compact_semantic_rt_execute(*query, desc, stats_out, &labels);
    if (status != GAFIME_STATUS_OK) return status;
    *temporary_peak_out = labels.staging_bytes;
    if (labels.staging_bytes > desc->max_temporary_bytes) return GAFIME_STATUS_OUT_OF_MEMORY;

    std::lock_guard<std::mutex> guard(query->mutex);
    query->membership_valid = false;
    ScopedCudaDevice device(query->device_id);
    if (device.status() != cudaSuccess) return cuda_status(device.status());
    status = validate_compact_semantic_rt_snapshot(*query);
    if (status != GAFIME_STATUS_OK) return status;
    CompactSemanticRtTemporaryAllocation staging;
    status = stage_compact_semantic_rt_labels(*query, *desc, labels, &staging);
    if (status != GAFIME_STATUS_OK) return status;
    /* Label staging uses the caller/default stream; synchronize before an
     * independently owned OptiX stream consumes its bit masks. */
    if (query->geometry_available && !query->force_sm && !query->force_sm_exhaustive) {
        status = cuda_status(cudaDeviceSynchronize());
        if (status == GAFIME_STATUS_OK) {
            status = run_compact_semantic_rt_optix(*query, desc->statistic_mask, desc->finalizer_mask, labels);
        }
    } else if (query->require_rt) {
        status = GAFIME_STATUS_UNSUPPORTED_BACKEND;
    } else {
        status = run_compact_semantic_rt_sm(*query, desc->statistic_mask, desc->finalizer_mask, labels);
        if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    }
    if (status != GAFIME_STATUS_OK) return status;
    const size_t stats_bytes = static_cast<size_t>(query->region_count) *
        sizeof(GafimeSemanticRtRegionExactStats);
    status = cuda_status(cudaMemcpy(
        stats_out->records, query->stats_device, stats_bytes, cudaMemcpyDeviceToHost));
    if (status != GAFIME_STATUS_OK) return status;
    query->membership_valid = true;
    stats_out->requested_statistic_mask = desc->statistic_mask;
    stats_out->finalized_mask = desc->finalizer_mask;
    stats_out->count = query->region_count;
    return GAFIME_STATUS_OK;
}

int materialize_compact_semantic_rt_coverage(
    GafimeGpuSemanticRegionQuery handle,
    uint32_t output_slot,
    uint64_t max_temporary_bytes,
    uint64_t* temporary_peak_out
) {
    if (temporary_peak_out == nullptr || !gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *temporary_peak_out = 0u;
    CompactSemanticRtRegionQuery* query = compact_semantic_rt_query_from_handle(handle);
    if (query == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(query->mutex);
    if (!query->membership_valid) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (max_temporary_bytes != 0u) {
        /* No workspace is needed, but a positive caller ceiling is harmless. */
    }
    ScopedCudaDevice device(query->device_id);
    if (device.status() != cudaSuccess) return cuda_status(device.status());
    gafime_cuda_v1::detail::CudaSemanticBankView primary{};
    int status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(query->primary_bank, &primary);
    if (status != GAFIME_STATUS_OK) return status;
    if (primary.columns != query->primary_columns || primary.rows != query->rows ||
        output_slot < primary.source_slots || output_slot >= primary.slot_capacity ||
        primary.initialized_slots == nullptr || (*primary.initialized_slots)[output_slot] != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    float* output = primary.columns + static_cast<uint64_t>(output_slot) * query->rows;
    constexpr uint32_t threads = 256u;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(
        (query->rows + threads - 1u) / threads, 65'535u));
    if (query->direct_first_hit) {
        gafime_cuda_v1::rt_kernel::scatter_semantic_region_coverage_counts_kernel<<<blocks, threads>>>(
            query->direct_region_ordinals_device, query->rows, output);
    } else {
        gafime_cuda_v1::rt_kernel::materialize_semantic_region_coverage_kernel<<<blocks, threads>>>(
            query->primary_membership_words_device,
            query->rows,
            query->region_count,
            query->words_per_region,
            output
        );
    }
    status = cuda_status(cudaGetLastError());
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    if (status != GAFIME_STATUS_OK) return status;
    return gafime_cuda_v1::detail::commit_cuda_semantic_bank_outputs(
        query->primary_bank, &output_slot, 1u);
}

int materialize_compact_semantic_rt_weighted_sum(
    GafimeGpuSemanticRegionQuery handle,
    uint32_t output_slot,
    const float* region_weights,
    uint64_t region_weight_count,
    uint64_t max_temporary_bytes,
    uint64_t* temporary_peak_out
) {
    if (temporary_peak_out == nullptr || !gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *temporary_peak_out = 0u;
    CompactSemanticRtRegionQuery* query = compact_semantic_rt_query_from_handle(handle);
    if (query == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(query->mutex);
    /* This optional endpoint has a narrower public domain than the general
     * compact query ABI: Core only constructs weighted candidates for two
     * through sixty-four distinct canonical regions.  Check every count
     * before dereferencing caller memory. */
    if (!query->membership_valid || query->region_count < 2u ||
        query->region_count > 64u || region_weight_count < 2u ||
        region_weight_count > 64u || region_weights == nullptr ||
        !gafime_gpu_abi::naturally_aligned(region_weights) ||
        region_weight_count != static_cast<uint64_t>(query->region_count) ||
        !gafime_gpu_abi::fits_host_bytes(region_weight_count, sizeof(float))) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    /* Validate before a CUDA allocation or output write.  The caller retains
     * the host array through this synchronous call; the device never caches
     * its address or makes it part of query identity. */
    for (uint64_t index = 0u; index < region_weight_count; ++index) {
        if (!std::isfinite(region_weights[index])) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
    }
    uint64_t weight_bytes = 0u;
    uint64_t temporary_bytes = 0u;
    if (!checked_element_bytes(region_weight_count, sizeof(float), &weight_bytes) ||
        !checked_add_u64(weight_bytes, sizeof(uint32_t), &temporary_bytes)) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    *temporary_peak_out = temporary_bytes;
    if (temporary_bytes > max_temporary_bytes) return GAFIME_STATUS_OUT_OF_MEMORY;

    ScopedCudaDevice device(query->device_id);
    if (device.status() != cudaSuccess) return cuda_status(device.status());
    gafime_cuda_v1::detail::CudaSemanticBankView primary{};
    int status = gafime_cuda_v1::detail::inspect_cuda_semantic_bank(query->primary_bank, &primary);
    if (status != GAFIME_STATUS_OK) return status;
    if (primary.columns != query->primary_columns || primary.rows != query->rows ||
        output_slot < primary.source_slots || output_slot >= primary.slot_capacity ||
        primary.initialized_slots == nullptr || (*primary.initialized_slots)[output_slot] != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }

    CompactSemanticRtTemporaryAllocation temporary;
    status = temporary.allocate(temporary_bytes);
    if (status != GAFIME_STATUS_OK) return status;
    float* weights_device = temporary.as<float>();
    uint32_t* nonfinite_device = temporary.as<uint32_t>(weight_bytes);
    status = cuda_status(cudaMemcpy(
        weights_device,
        region_weights,
        static_cast<size_t>(weight_bytes),
        cudaMemcpyHostToDevice
    ));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(nonfinite_device, 0, sizeof(uint32_t)));
    }
    if (status != GAFIME_STATUS_OK) return status;

    float* output = primary.columns + static_cast<uint64_t>(output_slot) * query->rows;
    constexpr uint32_t threads = 256u;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(
        (query->rows + threads - 1u) / threads, 65'535u));
    if (query->direct_first_hit) {
        if (query->direct_region_ordinals_device == nullptr) return GAFIME_STATUS_DEVICE_ERROR;
        gafime_cuda_v1::rt_kernel::materialize_semantic_region_weighted_sum_ordinals_kernel<<<
            blocks, threads
        >>>(
            query->direct_region_ordinals_device,
            query->rows,
            query->region_count,
            weights_device,
            output,
            nonfinite_device
        );
    } else {
        if (query->primary_membership_words_device == nullptr) return GAFIME_STATUS_DEVICE_ERROR;
        gafime_cuda_v1::rt_kernel::materialize_semantic_region_weighted_sum_kernel<<<
            blocks, threads
        >>>(
            query->primary_membership_words_device,
            query->rows,
            query->region_count,
            query->words_per_region,
            weights_device,
            output,
            nonfinite_device
        );
    }
    status = cuda_status(cudaGetLastError());
    if (status == GAFIME_STATUS_OK) status = cuda_status(cudaDeviceSynchronize());
    uint32_t has_nonfinite = 0u;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(
            &has_nonfinite, nonfinite_device, sizeof(has_nonfinite), cudaMemcpyDeviceToHost));
    }
    if (status != GAFIME_STATUS_OK) return status;
    /* The temporary flag distinguishes legal finite subnormal results from
     * overflow/invalid arithmetic.  On failure the bank slot stays logically
     * uninitialized even though an implementation may have written other
     * finite rows into its private backing storage. */
    if (has_nonfinite != 0u) return GAFIME_STATUS_INVALID_ARGUMENT;
    return gafime_cuda_v1::detail::commit_cuda_semantic_bank_outputs(
        query->primary_bank, &output_slot, 1u);
}

int free_compact_semantic_rt_region_query(GafimeGpuSemanticRegionQuery handle) {
    CompactSemanticRtRegionQuery* query = compact_semantic_rt_query_from_handle(handle);
    if (query == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> guard(query->mutex);
        ScopedCudaDevice device(query->device_id);
        if (device.status() != cudaSuccess) return cuda_status(device.status());
        const int status = query->reset();
        if (status != GAFIME_STATUS_OK) return status;
        query->magic = 0u;
    }
    delete query;
    return GAFIME_STATUS_OK;
}

}  // namespace

namespace gafime_cuda_v1 {

void tune_rt_kernels_for_device(const cudaDeviceProp& props) {
    const cudaFuncCache cache_mode = props.major >= 7 ? cudaFuncCachePreferShared : cudaFuncCachePreferL1;
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::rt_kernel::decision_path_membership_kernel,
        cache_mode
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::rt_kernel::pack_decision_path_points_kernel,
        cache_mode
    ));
}

cudaError_t launch_decision_path_membership(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership,
    cudaStream_t stream
) {
    if (path_count == 0 || n_samples == 0) {
        return cudaSuccess;
    }
    uint32_t max_grid_y = 0;
    cudaError_t status = current_device_max_grid_y(&max_grid_y);
    if (status != cudaSuccess) {
        return status;
    }
    const uint64_t tile_count = detail::decision_path_row_tile_count(n_samples, max_grid_y);
    for (uint64_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const detail::DecisionPathRowTile tile =
            detail::decision_path_row_tile(n_samples, max_grid_y, tile_idx);
        const dim3 grid(path_count, tile.block_count);
        rt_kernel::decision_path_membership_kernel<<<
            grid,
            detail::kDecisionPathThreads,
            0,
            stream
        >>>(
            features,
            n_samples,
            tile.row_offset,
            n_features,
            terms,
            path_offsets,
            path_count,
            membership
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

int execute_decision_path_membership(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint32_t device_flags,
    bool features_are_finite,
    uint64_t feature_generation,
    const GafimeDecisionPathBatch* paths
) {
    static_cast<void>(device_flags);
    int status = validate_decision_path_batch(resident_features, rows, cols, paths);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    ScopedCudaDevice device(device_id);
    if (device.status() != cudaSuccess) {
        return cuda_status(device.status());
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, static_cast<int>(device_id)) != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    tune_rt_kernels_for_device(props);

    if (!rt_disabled_by_env()) {
        status = GAFIME_STATUS_UNSUPPORTED_BACKEND;
        if (features_are_finite) {
            status = execute_decision_path_membership_optix(
                resident_features,
                rows,
                cols,
                device_id,
                arch_class,
                feature_generation,
                paths
            );
        }
        if (status == GAFIME_STATUS_OK) {
            return status;
        }
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || rt_required(paths)) {
            return status;
        }
    } else if (rt_required(paths)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    return execute_decision_path_membership_sm(resident_features, rows, cols, paths);
}

int execute_decision_path_score(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint32_t device_flags,
    bool features_are_finite,
    uint64_t feature_generation,
    uint64_t target_generation,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
) {
    static_cast<void>(device_flags);
    int status = validate_decision_path_score_batch(resident_features, target, rows, cols, paths, result);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    ScopedCudaDevice device(device_id);
    if (device.status() != cudaSuccess) {
        return cuda_status(device.status());
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, static_cast<int>(device_id)) != cudaSuccess) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
    tune_rt_kernels_for_device(props);
    if (!features_are_finite) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    if (!rt_disabled_by_env()) {
        status = execute_decision_path_score_optix(
            resident_features,
            target,
            rows,
            cols,
            device_id,
            arch_class,
            feature_generation,
            target_generation,
            paths,
            result
        );
        if (status == GAFIME_STATUS_OK) {
            return status;
        }
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND ||
            rt_score_first_hit_requested_env() ||
            (paths->flags & GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
            return status;
        }
    } else if (rt_score_first_hit_requested_env() ||
               (paths->flags & GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    return execute_decision_path_score_sm(resident_features, target, rows, cols, paths, result);
}

int release_decision_path_device_state(uint32_t device_id) {
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    return release_rt_device_state(device_id);
#else
    static_cast<void>(device_id);
    return GAFIME_STATUS_OK;
#endif
}

}  // namespace gafime_cuda_v1

extern "C" GAFIME_GPU_API int gafime_gpu_decision_path_membership(
    GafimeGpuMatrix matrix_handle,
    const GafimeDecisionPathBatch* paths
) try {
    if (matrix_handle == nullptr || paths == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (paths->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    gafime_cuda_v1::detail::CudaMatrixView matrix{};
    const int content_status =
        gafime_cuda_v1::detail::inspect_cuda_matrix(matrix_handle, &matrix);
    if (content_status != GAFIME_STATUS_OK) {
        return content_status;
    }
    return gafime_cuda_v1::execute_decision_path_membership(
        matrix.features,
        matrix.rows,
        matrix.cols,
        matrix.device_id,
        matrix.architecture_class,
        matrix.device_flags,
        matrix.features_are_finite,
        matrix.feature_generation,
        paths
    );
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_decision_path_score(
    GafimeGpuMatrix matrix_handle,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result_out
) try {
    if (matrix_handle == nullptr || paths == nullptr || result_out == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (paths->abi_version != GAFIME_ABI_VERSION ||
        result_out->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_ABI_MISMATCH;
    }
    gafime_cuda_v1::detail::CudaMatrixView matrix{};
    const int content_status =
        gafime_cuda_v1::detail::inspect_cuda_matrix(matrix_handle, &matrix);
    if (content_status != GAFIME_STATUS_OK) {
        return content_status;
    }
    return gafime_cuda_v1::execute_decision_path_score(
        matrix.features,
        matrix.target,
        matrix.rows,
        matrix.cols,
        matrix.device_id,
        matrix.architecture_class,
        matrix.device_flags,
        matrix.features_are_finite,
        matrix.feature_generation,
        matrix.target_generation,
        paths,
        result_out
    );
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_decision_path_release_device_state(uint32_t device_id) {
    try {
        return gafime_cuda_v1::release_decision_path_device_state(device_id);
    } catch (const std::bad_alloc&) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return GAFIME_STATUS_DEVICE_ERROR;
    }
}

extern "C" GAFIME_GPU_API int gafime_gpu_semantic_region_materialize_rt_v1(
    GafimeGpuSemanticBank bank,
    const GafimeSemanticProgramBatch* batch,
    uint64_t max_temporary_bytes,
    uint64_t* peak_bytes_out
) try {
#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
    return execute_semantic_region_materialize_rt_optix(
        bank, batch, max_temporary_bytes, peak_bytes_out);
#else
    static_cast<void>(bank);
    static_cast<void>(batch);
    static_cast<void>(max_temporary_bytes);
    if (peak_bytes_out != nullptr) *peak_bytes_out = 0u;
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
#endif
} catch (const std::bad_alloc&) {
    if (peak_bytes_out != nullptr) *peak_bytes_out = 0u;
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    if (peak_bytes_out != nullptr) *peak_bytes_out = 0u;
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_semantic_region_query_create_rt_v1(
    const GafimeSemanticRtRegionQueryDesc* desc,
    GafimeGpuSemanticRegionQuery* query_out,
    uint64_t* persistent_bytes_out
) try {
    return create_compact_semantic_rt_region_query(desc, query_out, persistent_bytes_out);
} catch (const std::bad_alloc&) {
    if (query_out != nullptr && gafime_gpu_abi::naturally_aligned(query_out)) *query_out = nullptr;
    if (persistent_bytes_out != nullptr && gafime_gpu_abi::naturally_aligned(persistent_bytes_out)) {
        *persistent_bytes_out = 0u;
    }
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    if (query_out != nullptr && gafime_gpu_abi::naturally_aligned(query_out)) *query_out = nullptr;
    if (persistent_bytes_out != nullptr && gafime_gpu_abi::naturally_aligned(persistent_bytes_out)) {
        *persistent_bytes_out = 0u;
    }
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_semantic_region_query_execute_rt_v1(
    GafimeGpuSemanticRegionQuery query,
    const GafimeSemanticRtRegionExecuteDesc* desc,
    GafimeSemanticRtRegionStatsTable* stats_out,
    uint64_t* temporary_peak_out
) try {
    return execute_compact_semantic_rt_region_query(query, desc, stats_out, temporary_peak_out);
} catch (const std::bad_alloc&) {
    if (temporary_peak_out != nullptr && gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        *temporary_peak_out = 0u;
    }
    if (validate_compact_semantic_rt_stats_table_header(stats_out) == GAFIME_STATUS_OK) {
        clear_compact_semantic_rt_stats_table(stats_out);
    }
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    if (temporary_peak_out != nullptr && gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        *temporary_peak_out = 0u;
    }
    if (validate_compact_semantic_rt_stats_table_header(stats_out) == GAFIME_STATUS_OK) {
        clear_compact_semantic_rt_stats_table(stats_out);
    }
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_semantic_region_query_materialize_coverage_rt_v1(
    GafimeGpuSemanticRegionQuery query,
    uint32_t output_slot,
    uint64_t max_temporary_bytes,
    uint64_t* temporary_peak_out
) try {
    return materialize_compact_semantic_rt_coverage(
        query, output_slot, max_temporary_bytes, temporary_peak_out);
} catch (const std::bad_alloc&) {
    if (temporary_peak_out != nullptr && gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        *temporary_peak_out = 0u;
    }
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    if (temporary_peak_out != nullptr && gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        *temporary_peak_out = 0u;
    }
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_semantic_region_query_materialize_weighted_sum_rt_v1(
    GafimeGpuSemanticRegionQuery query,
    uint32_t output_slot,
    const float* region_weights,
    uint64_t region_weight_count,
    uint64_t max_temporary_bytes,
    uint64_t* temporary_peak_out
) try {
    return materialize_compact_semantic_rt_weighted_sum(
        query,
        output_slot,
        region_weights,
        region_weight_count,
        max_temporary_bytes,
        temporary_peak_out
    );
} catch (const std::bad_alloc&) {
    if (temporary_peak_out != nullptr && gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        *temporary_peak_out = 0u;
    }
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    if (temporary_peak_out != nullptr && gafime_gpu_abi::naturally_aligned(temporary_peak_out)) {
        *temporary_peak_out = 0u;
    }
    return GAFIME_STATUS_DEVICE_ERROR;
}

extern "C" GAFIME_GPU_API int gafime_gpu_semantic_region_query_free_rt_v1(
    GafimeGpuSemanticRegionQuery query
) try {
    return free_compact_semantic_rt_region_query(query);
} catch (const std::bad_alloc&) {
    return GAFIME_STATUS_OUT_OF_MEMORY;
} catch (...) {
    return GAFIME_STATUS_DEVICE_ERROR;
}
