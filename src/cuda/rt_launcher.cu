#include "rt_launcher.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "rt_kernels.cuh"

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
#include <cuda.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "gafime_rt_optix_ptx.hpp"
#endif

namespace {

int cuda_status(cudaError_t status) {
    return status == cudaSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

template <typename T>
int ensure_device_capacity(T** ptr, size_t& capacity, size_t count) {
    if (count <= capacity) {
        return GAFIME_STATUS_OK;
    }
    cudaFree(*ptr);
    *ptr = nullptr;
    capacity = 0;
    if (count == 0) {
        return GAFIME_STATUS_OK;
    }
    const int status = cuda_status(cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)));
    if (status == GAFIME_STATUS_OK) {
        capacity = count;
    }
    return status;
}

int ensure_device_bytes(void** ptr, size_t& capacity, size_t bytes) {
    if (bytes <= capacity) {
        return GAFIME_STATUS_OK;
    }
    cudaFree(*ptr);
    *ptr = nullptr;
    capacity = 0;
    if (bytes == 0) {
        return GAFIME_STATUS_OK;
    }
    const int status = cuda_status(cudaMalloc(ptr, bytes));
    if (status == GAFIME_STATUS_OK) {
        capacity = bytes;
    }
    return status;
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
    if (resident_features == nullptr || batch == nullptr || batch->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if ((batch->flags & ~GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_count == 0 || batch->path_count == UINT32_MAX || batch->term_count == 0 ||
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
    if (output_count > static_cast<uint64_t>(SIZE_MAX / sizeof(float))) {
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
    if (resident_features == nullptr || target == nullptr || batch == nullptr ||
        batch->abi_version != GAFIME_ABI_VERSION || result == nullptr ||
        result->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if ((batch->flags & ~GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_count == 0 || batch->term_count == 0 || batch->metric_count == 0 ||
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
    return GAFIME_STATUS_OK;
}

bool rt_disabled_by_env() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT");
    if (mode == nullptr) {
        return false;
    }
    if (mode[0] == '0' || mode[0] == 'n' || mode[0] == 'N' ||
        mode[0] == 'f' || mode[0] == 'F' || mode[0] == 's' || mode[0] == 'S') {
        return true;
    }
    return (mode[0] == 'o' || mode[0] == 'O') && (mode[1] == 'f' || mode[1] == 'F');
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)

enum class RtGeometryMode : uint32_t {
    CustomAabb = 0,
    Triangle2d = 1,
    Triangle2dInstanced = 2,
};

bool cuda_arch_has_rt_cores(uint64_t arch_class) {
    return arch_class == GAFIME_GPU_ARCH_NVIDIA_TURING ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_AMPERE ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_ADA ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_HOPPER ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_BLACKWELL;
}

bool rt_force_custom_aabb() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT_GEOMETRY");
    return mode != nullptr && (mode[0] == 'a' || mode[0] == 'A' ||
        mode[0] == 'c' || mode[0] == 'C');
}

bool rt_score_direct_stats_requested() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT_SCORE");
    return mode != nullptr && (mode[0] == 'd' || mode[0] == 'D');
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
        if (!std::isfinite(term.threshold)) {
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

RtGeometryMode choose_rt_geometry_mode(const RtBoxPlan& plan) {
    if (!rt_force_custom_aabb() && plan.dims == 2 && plan.all_boxes_bounded) {
        return RtGeometryMode::Triangle2d;
    }
    return RtGeometryMode::CustomAabb;
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
        if (!std::isfinite(term.threshold) || !append_unique_axis(axes, term.feature)) {
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
    for (uint32_t path_idx = 0; path_idx < paths->path_count; ++path_idx) {
        if (!collect_path_axes(paths, path_idx, path_axes)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        bool placed = false;
        for (RtScoreGroup& group : groups) {
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
    uint32_t path_count
) {
    uint64_t hash = 0xcbf29ce484222325ull;
    hash = rt_hash_mix(hash, static_cast<uint32_t>(RtGeometryMode::Triangle2dInstanced));
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
        if (choose_rt_geometry_mode(group_plan) != RtGeometryMode::Triangle2d) {
            plan.all_instanced_triangle2d = false;
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
    plan.instanced_geometry_signature = rt_instanced_group_signature(plan.group_plans, paths->path_count);
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
    const size_t offset_bytes = static_cast<size_t>(paths->path_count + 1u) * sizeof(uint32_t);
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
    const size_t offset_bytes = static_cast<size_t>(paths->path_count + 1u) * sizeof(uint32_t);
    const size_t mask_bytes = static_cast<size_t>(word_count) * sizeof(uint32_t);
    const size_t metric_id_bytes = static_cast<size_t>(paths->metric_count) * sizeof(uint32_t);
    const size_t metric_value_bytes = static_cast<size_t>(metric_value_count) * sizeof(float);

    GafimeDecisionPathTerm* terms_device = nullptr;
    uint32_t* offsets_device = nullptr;
    uint32_t* metric_ids_device = nullptr;
    uint32_t* mask_device = nullptr;
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
        status = cuda_status(cudaMalloc(&metric_values_device, metric_value_bytes));
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
        constexpr uint32_t threads = 256;
        const uint32_t row_blocks = static_cast<uint32_t>((rows + threads - 1u) / threads);
        dim3 grid(paths->path_count, row_blocks);
        gafime_cuda_v1::rt_kernel::decision_path_bitset_kernel<<<grid, threads>>>(
            resident_features,
            rows,
            cols,
            terms_device,
            offsets_device,
            paths->path_count,
            words_per_path,
            mask_device
        );
        status = cuda_status(cudaGetLastError());
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        gafime_cuda_v1::rt_kernel::score_decision_path_bitset_kernel<<<paths->path_count, threads>>>(
            mask_device,
            target,
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
    float* membership;
    uint32_t* membership_words;
    uint32_t* direct_inside_counts;
    float* direct_inside_sum_y;
    uint32_t rows;
    uint32_t path_count;
    uint32_t geometry_mode;
    uint32_t words_per_path;
    const uint32_t* group_path_offsets;
    uint32_t group_count;
    uint32_t point_group_stride;
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
    float* direct_inside_sum_y_device = nullptr;
    float* direct_target_stats_device = nullptr;
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
        geometry_mode = RtGeometryMode::CustomAabb;
    }

    bool ready(uint32_t wanted_device_id, RtGeometryMode wanted_geometry_mode) const {
        return context != nullptr && pipeline != nullptr && device_id == wanted_device_id &&
            geometry_mode == wanted_geometry_mode;
    }
};

RtOptixProgram& optix_program(RtGeometryMode mode) {
    static RtOptixProgram custom_program;
    static RtOptixProgram triangle_program;
    static RtOptixProgram instanced_triangle_program;
    if (mode == RtGeometryMode::Triangle2dInstanced) {
        return instanced_triangle_program;
    }
    return mode == RtGeometryMode::Triangle2d ? triangle_program : custom_program;
}

int ensure_optix_program(uint32_t device_id, RtGeometryMode geometry_mode) {
    RtOptixProgram& program = optix_program(geometry_mode);
    if (program.ready(device_id, geometry_mode)) {
        return GAFIME_STATUS_OK;
    }
    program.reset();

    if (cudaFree(nullptr) != cudaSuccess || optixInit() != OPTIX_SUCCESS) {
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

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = geometry_mode == RtGeometryMode::Triangle2dInstanced
        ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 1;
    pipeline_options.numAttributeValues = 0;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    const bool triangle_mode = geometry_mode == RtGeometryMode::Triangle2d ||
        geometry_mode == RtGeometryMode::Triangle2dInstanced;
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
        const uint32_t max_traversable_depth =
            geometry_mode == RtGeometryMode::Triangle2dInstanced ? 2u : 1u;
        status = optix_status(optixPipelineSetStackSize(program.pipeline, 0, 0, 0, max_traversable_depth));
    }
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    EmptyRecord raygen_record = {};
    EmptyRecord miss_record = {};
    EmptyRecord hitgroup_record = {};
    status = optix_status(optixSbtRecordPackHeader(program.program_groups[0], &raygen_record));
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixSbtRecordPackHeader(program.program_groups[1], &miss_record));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixSbtRecordPackHeader(program.program_groups[2], &hitgroup_record));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program.raygen_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program.miss_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program.hitgroup_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(program.raygen_record, &raygen_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(program.miss_record, &miss_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(program.hitgroup_record, &hitgroup_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    program.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(program.raygen_record);
    program.sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(program.miss_record);
    program.sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    program.sbt.missRecordCount = 1;
    program.sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(program.hitgroup_record);
    program.sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    program.sbt.hitgroupRecordCount = 1;
    program.device_id = device_id;
    program.geometry_mode = geometry_mode;
    return GAFIME_STATUS_OK;
}

int execute_decision_path_membership_optix(
    const float* resident_features,
    uint64_t rows,
    uint32_t device_id,
    uint64_t arch_class,
    bool features_are_finite,
    const GafimeDecisionPathBatch* paths
) {
    if (!features_are_finite || !cuda_arch_has_rt_cores(arch_class)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    RtBoxPlan plan;
    int status = build_rt_box_plan(paths, plan);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    const RtGeometryMode geometry_mode = choose_rt_geometry_mode(plan);
    status = ensure_optix_program(device_id, geometry_mode);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    RtOptixProgram& program = optix_program(geometry_mode);

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
    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        status = cuda_status(cudaMemcpy(program.boxes_device, plan.boxes.data(), box_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK && rebuild_gas && geometry_mode == RtGeometryMode::CustomAabb) {
        std::vector<OptixAabb> aabbs;
        aabbs.reserve(plan.boxes.size());
        for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
            aabbs.push_back({box.lo_x, box.lo_y, box.lo_z, box.hi_x, box.hi_y, box.hi_z});
        }
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
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> vertices;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> indices;
    if (status == GAFIME_STATUS_OK && rebuild_gas && geometry_mode == RtGeometryMode::Triangle2d) {
        build_rt_triangles(plan, vertices, indices);
        if (vertices.size() > program.vertex_capacity || indices.size() > program.index_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.vertices_device, program.vertex_capacity, vertices.size());
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_capacity(&program.indices_device, program.index_capacity, indices.size());
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(
                program.vertices_device,
                vertices.data(),
                vertices.size() * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex),
                cudaMemcpyHostToDevice
            ));
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(
                program.indices_device,
                indices.data(),
                indices.size() * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex),
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
        CUdeviceptr vertex_buffer = reinterpret_cast<CUdeviceptr>(program.vertices_device);
        const CUdeviceptr index_buffer = reinterpret_cast<CUdeviceptr>(program.indices_device);
        uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        OptixBuildInput build_input = {};
        if (geometry_mode == RtGeometryMode::Triangle2d) {
            build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_input.triangleArray.vertexBuffers = &vertex_buffer;
            build_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
            build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_input.triangleArray.vertexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
            build_input.triangleArray.indexBuffer = index_buffer;
            build_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
            build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_input.triangleArray.indexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
            build_input.triangleArray.flags = geometry_flags;
            build_input.triangleArray.numSbtRecords = 1;
        } else {
            build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
            build_input.customPrimitiveArray.numPrimitives = paths->path_count;
            build_input.customPrimitiveArray.flags = geometry_flags;
            build_input.customPrimitiveArray.numSbtRecords = 1;
        }

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = geometry_mode == RtGeometryMode::Triangle2d
            ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
            : OPTIX_BUILD_FLAG_NONE;
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

    return status;
}

int execute_decision_path_score_optix_planned(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t device_id,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result,
    const RtBoxPlan& plan,
    const float* precomputed_target_stats_device = nullptr,
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
    status = ensure_optix_program(device_id, geometry_mode);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    RtOptixProgram& program = optix_program(geometry_mode);
    const bool direct_stats = rt_score_direct_stats_requested();

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
    if (status == GAFIME_STATUS_OK && !direct_stats) {
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
    const float* target_stats_device = precomputed_target_stats_device;
    if (status == GAFIME_STATUS_OK && direct_stats && target_stats_device == nullptr) {
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
    if (status == GAFIME_STATUS_OK && rebuild_gas) {
        status = cuda_status(cudaMemcpy(program.boxes_device, plan.boxes.data(), box_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK && rebuild_gas && geometry_mode == RtGeometryMode::CustomAabb) {
        std::vector<OptixAabb> aabbs;
        aabbs.reserve(plan.boxes.size());
        for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
            aabbs.push_back({box.lo_x, box.lo_y, box.lo_z, box.hi_x, box.hi_y, box.hi_z});
        }
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
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> vertices;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> indices;
    if (status == GAFIME_STATUS_OK && rebuild_gas && geometry_mode == RtGeometryMode::Triangle2d) {
        build_rt_triangles(plan, vertices, indices);
        if (vertices.size() > program.vertex_capacity || indices.size() > program.index_capacity) {
            program.gas_valid = false;
        }
        status = ensure_device_capacity(&program.vertices_device, program.vertex_capacity, vertices.size());
        if (status == GAFIME_STATUS_OK) {
            status = ensure_device_capacity(&program.indices_device, program.index_capacity, indices.size());
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(
                program.vertices_device,
                vertices.data(),
                vertices.size() * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex),
                cudaMemcpyHostToDevice
            ));
        }
        if (status == GAFIME_STATUS_OK) {
            status = cuda_status(cudaMemcpy(
                program.indices_device,
                indices.data(),
                indices.size() * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex),
                cudaMemcpyHostToDevice
            ));
        }
    }
    if (status == GAFIME_STATUS_OK && !direct_stats) {
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
            static_cast<size_t>(paths->path_count) * sizeof(float),
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
    if (status == GAFIME_STATUS_OK && direct_stats && precomputed_target_stats_device == nullptr) {
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
        CUdeviceptr vertex_buffer = reinterpret_cast<CUdeviceptr>(program.vertices_device);
        const CUdeviceptr index_buffer = reinterpret_cast<CUdeviceptr>(program.indices_device);
        uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        OptixBuildInput build_input = {};
        if (geometry_mode == RtGeometryMode::Triangle2d) {
            build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_input.triangleArray.vertexBuffers = &vertex_buffer;
            build_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
            build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_input.triangleArray.vertexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
            build_input.triangleArray.indexBuffer = index_buffer;
            build_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
            build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_input.triangleArray.indexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
            build_input.triangleArray.flags = geometry_flags;
            build_input.triangleArray.numSbtRecords = 1;
        } else {
            build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
            build_input.customPrimitiveArray.numPrimitives = paths->path_count;
            build_input.customPrimitiveArray.flags = geometry_flags;
            build_input.customPrimitiveArray.numSbtRecords = 1;
        }

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = geometry_mode == RtGeometryMode::Triangle2d
            ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
            : OPTIX_BUILD_FLAG_NONE;
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
    params.membership = nullptr;
    params.membership_words = direct_stats ? nullptr : program.membership_words_device;
    params.direct_inside_counts = direct_stats ? program.direct_inside_counts_device : nullptr;
    params.direct_inside_sum_y = direct_stats ? program.direct_inside_sum_y_device : nullptr;
    params.rows = static_cast<uint32_t>(rows);
    params.path_count = paths->path_count;
    params.geometry_mode = static_cast<uint32_t>(geometry_mode);
    params.words_per_path = direct_stats ? 0u : words_per_path;
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
        const uint32_t blocks = (paths->path_count + threads - 1u) / threads;
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
    return status;
}

int execute_decision_path_score_optix_grouped_instanced(
    const float* resident_features,
    const float* target,
    uint64_t rows,
    uint32_t device_id,
    const GafimeDecisionPathScoreBatch* paths,
    const RtGroupedScorePlan& grouped_plan,
    uint64_t feature_generation,
    const float* precomputed_target_stats_device,
    const uint32_t* flattened_original_paths_device,
    float* final_metric_values_device
) {
    const std::vector<RtScoreGroup>& groups = grouped_plan.groups;
    if (!rt_score_direct_stats_requested() || groups.size() <= 1u || rows > UINT32_MAX / 3u) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (!grouped_plan.all_instanced_triangle2d) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    int status = ensure_optix_program(device_id, RtGeometryMode::Triangle2dInstanced);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    RtOptixProgram& program = optix_program(RtGeometryMode::Triangle2dInstanced);
    if (program.stream == nullptr) {
        status = cuda_status(cudaStreamCreate(&program.stream));
        if (status != GAFIME_STATUS_OK) {
            return status;
        }
    }

    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> vertices;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> indices;
    std::vector<uint32_t> vertex_offsets(groups.size(), 0u);
    std::vector<uint32_t> vertex_counts(groups.size(), 0u);
    std::vector<uint32_t> index_offsets(groups.size(), 0u);
    std::vector<uint32_t> index_counts(groups.size(), 0u);

    const size_t point_count = static_cast<size_t>(rows) * groups.size() * 3u;
    const size_t direct_stats_count = static_cast<size_t>(paths->path_count);
    const uint64_t geometry_signature = grouped_plan.instanced_geometry_signature;
    bool rebuild_geometry = !program.gas_valid || program.gas_signature != geometry_signature;
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
            std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriVertex> group_vertices;
            std::vector<gafime_cuda_v1::rt_kernel::GafimeRtTriIndex> group_indices;
            build_rt_triangles(grouped_plan.group_plans[group_idx], group_vertices, group_indices);
            vertex_offsets[group_idx] = static_cast<uint32_t>(vertices.size());
            vertex_counts[group_idx] = static_cast<uint32_t>(group_vertices.size());
            index_offsets[group_idx] = static_cast<uint32_t>(indices.size());
            index_counts[group_idx] = static_cast<uint32_t>(group_indices.size());
            vertices.insert(vertices.end(), group_vertices.begin(), group_vertices.end());
            indices.insert(indices.end(), group_indices.begin(), group_indices.end());
        }
    }
    if (point_count > program.points_capacity) {
        program.packed_points_valid = false;
    }
    status = ensure_device_capacity(&program.points_device, program.points_capacity, point_count);
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.boxes_device, program.box_capacity, grouped_plan.flat_boxes.size());
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.vertices_device, program.vertex_capacity, vertices.size());
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = ensure_device_capacity(&program.indices_device, program.index_capacity, indices.size());
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
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = cuda_status(cudaMemcpy(program.vertices_device, vertices.data(), vertices.size() * sizeof(vertices[0]), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK && rebuild_geometry) {
        status = cuda_status(cudaMemcpy(program.indices_device, indices.data(), indices.size() * sizeof(indices[0]), cudaMemcpyHostToDevice));
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
            CUdeviceptr vertex_buffer = reinterpret_cast<CUdeviceptr>(
                program.vertices_device + vertex_offsets[group_idx]
            );
            OptixBuildInput build_input = {};
            build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_input.triangleArray.vertexBuffers = &vertex_buffer;
            build_input.triangleArray.numVertices = vertex_counts[group_idx];
            build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_input.triangleArray.vertexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
            build_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(
                program.indices_device + index_offsets[group_idx]
            );
            build_input.triangleArray.numIndexTriplets = index_counts[group_idx];
            build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_input.triangleArray.indexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
            build_input.triangleArray.flags = geometry_flags;
            build_input.triangleArray.numSbtRecords = 1;

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
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
            CUdeviceptr vertex_buffer = reinterpret_cast<CUdeviceptr>(
                program.vertices_device + vertex_offsets[group_idx]
            );
            OptixBuildInput build_input = {};
            build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_input.triangleArray.vertexBuffers = &vertex_buffer;
            build_input.triangleArray.numVertices = vertex_counts[group_idx];
            build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_input.triangleArray.vertexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriVertex);
            build_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(
                program.indices_device + index_offsets[group_idx]
            );
            build_input.triangleArray.numIndexTriplets = index_counts[group_idx];
            build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_input.triangleArray.indexStrideInBytes = sizeof(gafime_cuda_v1::rt_kernel::GafimeRtTriIndex);
            build_input.triangleArray.flags = geometry_flags;
            build_input.triangleArray.numSbtRecords = 1;

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
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

    status = cuda_status(cudaMemsetAsync(program.direct_inside_counts_device, 0, direct_stats_count * sizeof(uint32_t), program.stream));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemsetAsync(program.direct_inside_sum_y_device, 0, direct_stats_count * sizeof(float), program.stream));
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
    params.direct_inside_counts = program.direct_inside_counts_device;
    params.direct_inside_sum_y = program.direct_inside_sum_y_device;
    params.rows = static_cast<uint32_t>(rows);
    params.path_count = paths->path_count;
    params.geometry_mode = static_cast<uint32_t>(RtGeometryMode::Triangle2dInstanced);
    params.group_path_offsets = program.group_path_offsets_device;
    params.group_count = static_cast<uint32_t>(groups.size());
    params.point_group_stride = static_cast<uint32_t>(rows * 3u);
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
        const uint32_t blocks = (paths->path_count + threads - 1u) / threads;
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
    return status;
}

int execute_decision_path_score_optix_grouped(
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
        status = ensure_optix_program(device_id, RtGeometryMode::Triangle2dInstanced);
        if (status == GAFIME_STATUS_OK) {
            direct_program = &optix_program(RtGeometryMode::Triangle2dInstanced);
        } else if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
            return status;
        }
    }

    const uint64_t grouped_plan_signature = rt_score_batch_signature(paths);
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
        if (status == GAFIME_STATUS_OK) {
            if (direct_program->stream == nullptr) {
                status = cuda_status(cudaStreamCreate(&direct_program->stream));
            }
            if (status == GAFIME_STATUS_OK) {
                status = ensure_device_capacity(
                    &direct_program->direct_target_stats_device,
                    direct_program->direct_target_stats_capacity,
                    static_cast<size_t>(3u)
                );
            }
            const bool reuse_target_stats =
                status == GAFIME_STATUS_OK &&
                direct_program->target_stats_valid &&
                target_generation != 0u &&
                direct_program->target_stats_target == target &&
                direct_program->target_stats_rows == rows &&
                direct_program->target_stats_generation == target_generation;
            const bool reuse_original_paths =
                status == GAFIME_STATUS_OK &&
                direct_program->grouped_original_paths_valid &&
                flattened_original_paths.size() <= direct_program->grouped_original_path_capacity &&
                direct_program->grouped_original_paths_signature == original_paths_signature &&
                direct_program->grouped_original_paths_count == flattened_original_paths.size();
            if (status == GAFIME_STATUS_OK) {
                status = ensure_device_capacity(
                    &direct_program->grouped_final_metric_values_device,
                    direct_program->grouped_final_metric_value_capacity,
                    static_cast<size_t>(final_metric_value_count)
                );
            }
            if (status == GAFIME_STATUS_OK) {
                status = ensure_device_capacity(
                    &direct_program->grouped_original_paths_device,
                    direct_program->grouped_original_path_capacity,
                    flattened_original_paths.size()
                );
            }
            if (status == GAFIME_STATUS_OK && !reuse_original_paths) {
                status = cuda_status(cudaMemcpyAsync(
                    direct_program->grouped_original_paths_device,
                    flattened_original_paths.data(),
                    flattened_original_paths.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    direct_program->stream
                ));
                if (status == GAFIME_STATUS_OK) {
                    direct_program->grouped_original_paths_valid = true;
                    direct_program->grouped_original_paths_signature = original_paths_signature;
                    direct_program->grouped_original_paths_count = flattened_original_paths.size();
                } else {
                    direct_program->grouped_original_paths_valid = false;
                }
            }
            if (status == GAFIME_STATUS_OK && !reuse_target_stats) {
                constexpr uint32_t threads = 256;
                gafime_cuda_v1::rt_kernel::decision_path_target_stats_kernel<<<1, threads, 0, direct_program->stream>>>(
                    target,
                    rows,
                    direct_program->direct_target_stats_device
                );
                status = cuda_status(cudaGetLastError());
                if (status == GAFIME_STATUS_OK) {
                    direct_program->target_stats_valid = true;
                    direct_program->target_stats_target = target;
                    direct_program->target_stats_rows = rows;
                    direct_program->target_stats_generation = target_generation;
                } else {
                    direct_program->target_stats_valid = false;
                }
            }
            if (status == GAFIME_STATUS_OK) {
                status = execute_decision_path_score_optix_grouped_instanced(
                    resident_features,
                    target,
                    rows,
                    device_id,
                    paths,
                    *grouped_plan,
                    feature_generation,
                    direct_program->direct_target_stats_device,
                    direct_program->grouped_original_paths_device,
                    direct_program->grouped_final_metric_values_device
                );
            }
            if (status == GAFIME_STATUS_OK) {
                if (result->metric_count == paths->metric_count) {
                    status = cuda_status(cudaMemcpyAsync(
                        result->metric_values,
                        direct_program->grouped_final_metric_values_device,
                        final_metric_value_bytes,
                        cudaMemcpyDeviceToHost,
                        direct_program->stream
                    ));
                    if (status == GAFIME_STATUS_OK) {
                        status = cuda_status(cudaStreamSynchronize(direct_program->stream));
                    }
                    if (status != GAFIME_STATUS_OK) {
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
                    direct_program->grouped_final_metric_values_device,
                    final_metric_value_bytes,
                    cudaMemcpyDeviceToHost,
                    direct_program->stream
                ));
                if (status == GAFIME_STATUS_OK) {
                    status = cuda_status(cudaStreamSynchronize(direct_program->stream));
                }
                if (status != GAFIME_STATUS_OK) {
                    return status;
                }
                return write_decision_path_score_rows_host(paths, result, final_metric_values);
            }
        }
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
            return status;
        }
    }

    struct ScopedTargetStats {
        float* ptr = nullptr;
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
        status = cuda_status(cudaMalloc(reinterpret_cast<void**>(&shared_target_stats.ptr), 3u * sizeof(float)));
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
    uint32_t device_id,
    uint64_t arch_class,
    bool features_are_finite,
    uint64_t feature_generation,
    uint64_t target_generation,
    const GafimeDecisionPathScoreBatch* paths,
    GafimeResultTable* result
) {
    if (!features_are_finite || !cuda_arch_has_rt_cores(arch_class)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    RtBoxPlan plan;
    const int status = build_rt_box_plan(paths, plan);
    if (status == GAFIME_STATUS_OK) {
        return execute_decision_path_score_optix_planned(
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
    uint64_t,
    bool,
    const GafimeDecisionPathBatch*
) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

int execute_decision_path_score_optix(
    const float*,
    const float*,
    uint64_t,
    uint32_t,
    uint64_t,
    bool,
    uint64_t,
    uint64_t,
    const GafimeDecisionPathScoreBatch*,
    GafimeResultTable*
) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

#endif

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
    constexpr uint32_t threads = 256;
    const uint32_t row_blocks = static_cast<uint32_t>((n_samples + threads - 1) / threads);
    dim3 grid(path_count, row_blocks);
    dim3 block(threads);
    rt_kernel::decision_path_membership_kernel<<<grid, block, 0, stream>>>(
        features,
        n_samples,
        n_features,
        terms,
        path_offsets,
        path_count,
        membership
    );
    return cudaGetLastError();
}

int execute_decision_path_membership(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint32_t device_flags,
    bool features_are_finite,
    const GafimeDecisionPathBatch* paths
) {
    static_cast<void>(device_flags);
    int status = validate_decision_path_batch(resident_features, rows, cols, paths);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    if (!rt_disabled_by_env()) {
        status = execute_decision_path_membership_optix(
            resident_features,
            rows,
            device_id,
            arch_class,
            features_are_finite,
            paths
        );
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
    if (!features_are_finite) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    if (!rt_disabled_by_env()) {
        status = execute_decision_path_score_optix(
            resident_features,
            target,
            rows,
            device_id,
            arch_class,
            features_are_finite,
            feature_generation,
            target_generation,
            paths,
            result
        );
        if (status == GAFIME_STATUS_OK) {
            return status;
        }
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND ||
            (paths->flags & GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
            return status;
        }
    } else if ((paths->flags & GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    return execute_decision_path_score_sm(resident_features, target, rows, cols, paths, result);
}

}  // namespace gafime_cuda_v1
