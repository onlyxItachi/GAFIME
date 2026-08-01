#include "../../src/cuda/rt_abi.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kRows = 4096u;
constexpr uint32_t kCols = 2u;
constexpr uint32_t kPathCount = 2u;
constexpr uint32_t kMetricCount = 2u;
constexpr uint32_t kThreadCount = 8u;
constexpr uint32_t kIterations = 40u;

struct ScoreStorage {
    uint32_t combos[kPathCount]{};
    float metrics[kPathCount * kMetricCount]{};
    uint32_t ranks[kPathCount]{};
    uint32_t families[kPathCount]{};
    uint64_t candidate_ids[kPathCount]{};
    uint32_t row_flags[kPathCount]{};

    GafimeResultTable table() {
        GafimeResultTable result{};
        result.abi_version = GAFIME_ABI_VERSION;
        result.max_arity = 1u;
        result.metric_count = kMetricCount;
        result.capacity = kPathCount;
        result.combo_indices = combos;
        result.metric_values = metrics;
        result.ranks = ranks;
        result.families = families;
        result.candidate_ids = candidate_ids;
        result.row_flags = row_flags;
        return result;
    }
};

bool same_bits(float left, float right) {
    return std::bit_cast<uint32_t>(left) == std::bit_cast<uint32_t>(right);
}

int require_status(int status, const char* operation) {
    if (status == GAFIME_STATUS_OK) {
        return 0;
    }
    std::fprintf(stderr, "%s returned status %d\n", operation, status);
    return 1;
}

int require_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return 0;
    }
    std::fprintf(stderr, "%s returned %s\n", operation, cudaGetErrorString(status));
    return 1;
}

int execute_once(
    GafimeGpuMatrix matrix,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* offsets,
    const uint32_t* metrics,
    const std::vector<float>& expected_membership,
    const float* expected_scores
) {
    std::vector<float> membership(static_cast<size_t>(kRows) * kPathCount, -1.0f);
    GafimeDecisionPathBatch membership_batch{};
    membership_batch.abi_version = GAFIME_ABI_VERSION;
    membership_batch.path_count = kPathCount;
    membership_batch.term_count = 3u;
    membership_batch.flags = GAFIME_DECISION_PATH_FLAG_REQUIRE_RT;
    membership_batch.terms = terms;
    membership_batch.path_offsets = offsets;
    membership_batch.membership_host = membership.data();
    if (require_status(
            gafime_gpu_decision_path_membership(matrix, &membership_batch),
            "gafime_gpu_decision_path_membership"
        )) {
        return 1;
    }
    for (size_t index = 0; index < membership.size(); ++index) {
        if (!same_bits(membership[index], expected_membership[index])) {
            std::fprintf(stderr, "membership mismatch at %zu\n", index);
            return 1;
        }
    }

    GafimeDecisionPathScoreBatch score_batch{};
    score_batch.abi_version = GAFIME_ABI_VERSION;
    score_batch.path_count = kPathCount;
    score_batch.term_count = 3u;
    score_batch.flags = GAFIME_DECISION_PATH_FLAG_REQUIRE_RT;
    score_batch.terms = terms;
    score_batch.path_offsets = offsets;
    score_batch.metric_ids = metrics;
    score_batch.metric_count = kMetricCount;
    ScoreStorage score;
    GafimeResultTable result = score.table();
    if (require_status(
            gafime_gpu_decision_path_score(matrix, &score_batch, &result),
            "gafime_gpu_decision_path_score"
        )) {
        return 1;
    }
    if (result.row_count != kPathCount) {
        std::fprintf(stderr, "unexpected score row count %llu\n", static_cast<unsigned long long>(result.row_count));
        return 1;
    }
    for (uint32_t index = 0; index < kPathCount * kMetricCount; ++index) {
        if (!same_bits(score.metrics[index], expected_scores[index])) {
            std::fprintf(stderr, "score mismatch at %u\n", index);
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_count_status = cudaGetDeviceCount(&device_count);
    if (device_count_status == cudaErrorNoDevice ||
        device_count_status == cudaErrorInsufficientDriver || device_count == 0) {
        return 77;
    }
    if (require_cuda(device_count_status, "cudaGetDeviceCount")) {
        return 1;
    }
    GafimeGpuDeviceInfo info{};
    if (require_status(gafime_gpu_device_info(0u, &info), "gafime_gpu_device_info")) {
        return 1;
    }
    if ((info.flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) == 0u) {
        std::fprintf(stderr, "CUDA payload does not advertise OptiX RT\n");
        return 77;
    }
    std::vector<float> features(static_cast<size_t>(kRows) * kCols);
    std::vector<float> target(kRows);
    std::vector<float> expected_membership(static_cast<size_t>(kRows) * kPathCount);
    for (uint32_t row = 0; row < kRows; ++row) {
        const float x0 = static_cast<float>(static_cast<int32_t>(row % 257u) - 128) / 64.0f;
        const float x1 = static_cast<float>((row * 17u) % 251u) / 100.0f - 1.0f;
        features[static_cast<size_t>(row) * kCols] = x0;
        features[static_cast<size_t>(row) * kCols + 1u] = x1;
        target[row] = 0.75f * x0 - 0.25f * x1 + static_cast<float>(row % 5u) * 0.1f;
        expected_membership[row] = x0 > 0.25f && x1 <= 0.75f ? 1.0f : 0.0f;
        expected_membership[static_cast<size_t>(kRows) + row] = x0 <= -0.4f ? 1.0f : 0.0f;
    }

    const GafimeDecisionPathTerm terms[3] = {
        {0u, GAFIME_DECISION_PATH_SIGN_GT, 0.25f, 0u, {0u, 0u}},
        {1u, GAFIME_DECISION_PATH_SIGN_LE, 0.75f, 0u, {0u, 0u}},
        {0u, GAFIME_DECISION_PATH_SIGN_LE, -0.4f, 0u, {0u, 0u}},
    };
    const uint32_t offsets[3] = {0u, 2u, 3u};
    const uint32_t metrics[kMetricCount] = {GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};

    GafimeMatrixDesc desc{};
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = kRows;
    desc.cols = kCols;
    desc.row_stride = kCols;
    desc.bytes = static_cast<uint64_t>(features.size()) * sizeof(float);

    std::vector<GafimeGpuMatrix> matrices(kThreadCount, nullptr);

    if (device_count >= 2) {
        GafimeGpuMatrix teardown_probe = nullptr;
        if (require_status(
                gafime_gpu_matrix_alloc(0u, &desc, &teardown_probe),
                "gafime_gpu_matrix_alloc teardown probe"
            ) ||
            require_cuda(cudaSetDevice(1), "cudaSetDevice(1) before matrix free")) {
            gafime_gpu_matrix_free(teardown_probe);
            return 1;
        }
        gafime_gpu_matrix_free(teardown_probe);
        int current_device = -1;
        if (require_cuda(cudaGetDevice(&current_device), "cudaGetDevice after matrix free") ||
            current_device != 1) {
            std::fprintf(stderr, "matrix free changed caller device from 1 to %d\n", current_device);
            return 1;
        }
    }

    for (GafimeGpuMatrix& matrix : matrices) {
        if (require_status(gafime_gpu_matrix_alloc(0u, &desc, &matrix), "gafime_gpu_matrix_alloc") ||
            require_status(
                gafime_gpu_matrix_upload(matrix, features.data(), target.data(), kRows, kCols),
                "gafime_gpu_matrix_upload"
            )) {
            for (GafimeGpuMatrix allocated : matrices) {
                gafime_gpu_matrix_free(allocated);
            }
            return 1;
        }
    }

    float expected_scores[kPathCount * kMetricCount]{};
    {
        GafimeDecisionPathScoreBatch score_batch{};
        score_batch.abi_version = GAFIME_ABI_VERSION;
        score_batch.path_count = kPathCount;
        score_batch.term_count = 3u;
        score_batch.flags = GAFIME_DECISION_PATH_FLAG_REQUIRE_RT;
        score_batch.terms = terms;
        score_batch.path_offsets = offsets;
        score_batch.metric_ids = metrics;
        score_batch.metric_count = kMetricCount;
        ScoreStorage baseline;
        GafimeResultTable result = baseline.table();
        if (require_status(
                gafime_gpu_decision_path_score(matrices[0], &score_batch, &result),
                "baseline decision-path score"
            )) {
            for (GafimeGpuMatrix matrix : matrices) {
                gafime_gpu_matrix_free(matrix);
            }
            return 1;
        }
        for (uint32_t index = 0; index < kPathCount * kMetricCount; ++index) {
            expected_scores[index] = baseline.metrics[index];
        }
    }

    if (device_count >= 2) {
        if (require_cuda(cudaSetDevice(1), "cudaSetDevice(1) before RT calls") ||
            execute_once(
                matrices[0],
                terms,
                offsets,
                metrics,
                expected_membership,
                expected_scores
            )) {
            for (GafimeGpuMatrix matrix : matrices) {
                gafime_gpu_matrix_free(matrix);
            }
            return 1;
        }
        int current_device = -1;
        if (require_cuda(cudaGetDevice(&current_device), "cudaGetDevice after RT calls") ||
            current_device != 1) {
            std::fprintf(stderr, "RT calls changed caller device from 1 to %d\n", current_device);
            for (GafimeGpuMatrix matrix : matrices) {
                gafime_gpu_matrix_free(matrix);
            }
            return 1;
        }
    }

    std::atomic<uint32_t> ready{0u};
    std::atomic<bool> start{false};
    std::atomic<uint32_t> failures{0u};
    std::atomic<uint32_t> active{0u};
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (uint32_t thread_idx = 0; thread_idx < kThreadCount; ++thread_idx) {
        workers.emplace_back([&, thread_idx] {
            ready.fetch_add(1u, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
                active.fetch_add(1u, std::memory_order_release);
                const int execute_status = execute_once(
                        matrices[thread_idx],
                        terms,
                        offsets,
                        metrics,
                        expected_membership,
                        expected_scores
                    );
                active.fetch_sub(1u, std::memory_order_release);
                if (execute_status != 0) {
                    failures.fetch_add(1u, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreadCount) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    std::thread releaser([&] {
        while (active.load(std::memory_order_acquire) == 0u) {
            std::this_thread::yield();
        }
        for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
            if (require_status(
                    gafime_gpu_decision_path_release_device_state(0u),
                    "concurrent gafime_gpu_decision_path_release_device_state"
                )) {
                failures.fetch_add(1u, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
    });
    for (std::thread& worker : workers) {
        worker.join();
    }
    releaser.join();
    for (GafimeGpuMatrix matrix : matrices) {
        gafime_gpu_matrix_free(matrix);
    }
    if (failures.load(std::memory_order_relaxed) != 0u) {
        return 1;
    }
    if (device_count >= 2 &&
        require_cuda(cudaSetDevice(1), "cudaSetDevice(1) before RT cleanup")) {
        return 1;
    }
    const int release_status = gafime_gpu_decision_path_release_device_state(0u);
    if (device_count >= 2) {
        int current_device = -1;
        if (require_cuda(cudaGetDevice(&current_device), "cudaGetDevice after RT cleanup") ||
            current_device != 1) {
            std::fprintf(stderr, "RT cleanup changed caller device from 1 to %d\n", current_device);
            return 1;
        }
    }
    if (require_status(release_status, "gafime_gpu_decision_path_release_device_state")) {
        return 1;
    }
    std::printf(
        "same-device OptiX concurrency: %u threads x %u iterations passed\n",
        kThreadCount,
        kIterations
    );
    return 0;
}
