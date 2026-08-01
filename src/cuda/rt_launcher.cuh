#ifndef GAFIME_CUDA_RT_LAUNCHER_CUH
#define GAFIME_CUDA_RT_LAUNCHER_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "rt_abi.hpp"

namespace gafime_cuda_v1 {

namespace detail {

constexpr uint32_t kDecisionPathThreads = 256u;

struct DecisionPathRowTile {
    uint64_t row_offset;
    uint64_t row_count;
    uint32_t block_count;
};

constexpr uint64_t decision_path_tile_row_capacity(uint32_t max_grid_y) {
    return static_cast<uint64_t>(kDecisionPathThreads) * max_grid_y;
}

constexpr uint64_t decision_path_row_tile_count(uint64_t rows, uint32_t max_grid_y) {
    const uint64_t capacity = decision_path_tile_row_capacity(max_grid_y);
    return rows == 0u || capacity == 0u ? 0u : 1u + (rows - 1u) / capacity;
}

constexpr bool decision_path_group_count_fits_grid(size_t group_count, uint32_t max_grid_y) {
    return group_count != 0u &&
        max_grid_y != 0u &&
        group_count <= static_cast<size_t>(max_grid_y);
}

constexpr bool decision_path_score_needs_duplicate_guard(bool direct_first_hit) {
    return !direct_first_hit;
}

constexpr DecisionPathRowTile decision_path_row_tile(
    uint64_t rows,
    uint32_t max_grid_y,
    uint64_t tile_index
) {
    const uint64_t capacity = decision_path_tile_row_capacity(max_grid_y);
    const uint64_t row_offset = capacity == 0u ? 0u : capacity * tile_index;
    const uint64_t row_count = row_offset >= rows ? 0u : std::min(capacity, rows - row_offset);
    return {
        row_offset,
        row_count,
        static_cast<uint32_t>((row_count + kDecisionPathThreads - 1u) / kDecisionPathThreads),
    };
}

template <typename State>
class DeviceStateMap {
public:
    template <typename Factory>
    std::shared_ptr<State> get_or_create(uint32_t device_id, Factory&& factory) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto existing = states_.find(device_id);
        if (existing != states_.end()) {
            return existing->second;
        }
        std::shared_ptr<State> state = std::forward<Factory>(factory)(device_id);
        states_.emplace(device_id, state);
        return state;
    }

    template <typename Release>
    int release(uint32_t device_id, Release&& release) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto existing = states_.find(device_id);
        if (existing == states_.end()) {
            return GAFIME_STATUS_OK;
        }
        const int status = std::forward<Release>(release)(*existing->second);
        if (status == GAFIME_STATUS_OK) {
            states_.erase(existing);
        }
        return status;
    }

    size_t size() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return states_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint32_t, std::shared_ptr<State>> states_;
};

}  // namespace detail

void tune_rt_kernels_for_device(const cudaDeviceProp& props);

cudaError_t launch_decision_path_membership(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership,
    cudaStream_t stream
);

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
);

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
);

int release_decision_path_device_state(uint32_t device_id);

}  // namespace gafime_cuda_v1

#endif  // GAFIME_CUDA_RT_LAUNCHER_CUH
