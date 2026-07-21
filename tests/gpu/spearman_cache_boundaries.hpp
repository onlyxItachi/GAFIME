#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "../../src/common/gafime_gpu_abi.hpp"

namespace gafime_gpu_test {

inline int spearman_require_status(int status, const char* label) {
    if (status != GAFIME_STATUS_OK) {
        std::fprintf(stderr, "%s failed with status %d\n", label, status);
        return 1;
    }
    return 0;
}

inline int spearman_require_close(float actual, float expected, const char* label) {
    if (std::fabs(actual - expected) > 1e-4f) {
        std::fprintf(stderr, "%s mismatch: actual=%f expected=%f\n", label, actual, expected);
        return 1;
    }
    return 0;
}

inline float pairwise_spearman_reference(
    const std::vector<float>& x,
    const std::vector<float>& y
) {
    double sum_rx = 0.0;
    double sum_ry = 0.0;
    double sum_rxx = 0.0;
    double sum_ryy = 0.0;
    double sum_rxy = 0.0;
    double count = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const float xi = x[i];
        const float yi = y[i];
        if (!std::isfinite(xi) || !std::isfinite(yi)) {
            continue;
        }
        double less_x = 0.0;
        double equal_x = 0.0;
        double less_y = 0.0;
        double equal_y = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            const float xj = x[j];
            const float yj = y[j];
            if (!std::isfinite(xj) || !std::isfinite(yj)) {
                continue;
            }
            if (xj < xi) {
                less_x += 1.0;
            } else if (xj == xi) {
                equal_x += 1.0;
            }
            if (yj < yi) {
                less_y += 1.0;
            } else if (yj == yi) {
                equal_y += 1.0;
            }
        }
        const double rank_x = less_x + 0.5 * (equal_x - 1.0);
        const double rank_y = less_y + 0.5 * (equal_y - 1.0);
        sum_rx += rank_x;
        sum_ry += rank_y;
        sum_rxx += rank_x * rank_x;
        sum_ryy += rank_y * rank_y;
        sum_rxy += rank_x * rank_y;
        count += 1.0;
    }
    if (count <= 1.0) {
        return 0.0f;
    }
    const double covariance = count * sum_rxy - sum_rx * sum_ry;
    const double variance_x = count * sum_rxx - sum_rx * sum_rx;
    const double variance_y = count * sum_ryy - sum_ry * sum_ry;
    const double denominator = std::sqrt(variance_x * variance_y);
    if (!(denominator > 0.0)) {
        return 0.0f;
    }
    return static_cast<float>(std::max(-1.0, std::min(1.0, covariance / denominator)));
}

struct SpearmanResultStorage {
    explicit SpearmanResultStorage(uint64_t capacity, uint32_t max_arity = 1)
        : combo_indices(capacity * max_arity, std::numeric_limits<uint32_t>::max()),
          metric_values(capacity, 0.0f),
          ranks(capacity, 0),
          families(capacity, 0),
          candidate_ids(capacity, 0),
          row_flags(capacity, 0) {
        table.abi_version = GAFIME_ABI_VERSION;
        table.max_arity = max_arity;
        table.metric_count = 1;
        table.capacity = capacity;
        table.combo_indices = combo_indices.data();
        table.metric_values = metric_values.data();
        table.ranks = ranks.data();
        table.families = families.data();
        table.candidate_ids = candidate_ids.data();
        table.row_flags = row_flags.data();
    }

    std::vector<uint32_t> combo_indices;
    std::vector<float> metric_values;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> families;
    std::vector<uint64_t> candidate_ids;
    std::vector<uint32_t> row_flags;
    GafimeResultTable table{};
};

inline int verify_spearman_cache_boundaries(uint32_t backend_kind) {
    constexpr uint64_t kRows = 128;
    constexpr uint32_t kColumns = 2;

    std::vector<float> features(kRows * kColumns, 0.0f);
    std::vector<float> first_feature(kRows, 0.0f);
    std::vector<float> second_feature(kRows, 0.0f);
    std::vector<float> target(kRows, 0.0f);
    for (uint64_t row = 0; row < kRows; ++row) {
        const float first = static_cast<float>((row * 3 + row / 7) % 9) - 4.0f;
        const float second = static_cast<float>((row * 5 + row / 11) % 7) - 3.0f;
        const float y = static_cast<float>((row * 7 + row / 5) % 11) - 5.0f;
        features[row * kColumns] = first;
        features[row * kColumns + 1] = second;
        first_feature[row] = first;
        second_feature[row] = second;
        target[row] = y;
    }
    const auto centered_interaction = [](const std::vector<float>& first, const std::vector<float>& second) {
        double first_sum = 0.0;
        double second_sum = 0.0;
        for (size_t idx = 0; idx < first.size(); ++idx) {
            first_sum += first[idx];
            second_sum += second[idx];
        }
        const double inverse_count = 1.0 / static_cast<double>(first.size());
        const float first_mean = static_cast<float>(first_sum * inverse_count);
        const float second_mean = static_cast<float>(second_sum * inverse_count);
        std::vector<float> values(first.size(), 0.0f);
        for (size_t idx = 0; idx < first.size(); ++idx) {
            values[idx] = (first[idx] - first_mean) * (second[idx] - second_mean);
        }
        return values;
    };
    const std::vector<float> interaction = centered_interaction(first_feature, second_feature);

    GafimeMatrixDesc desc{};
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = kRows;
    desc.cols = kColumns;
    desc.row_stride = kColumns;
    desc.bytes = kRows * kColumns * sizeof(float);

    GafimeGpuMatrix matrix = nullptr;
    if (spearman_require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "spearman_cache_matrix_alloc")) {
        return 1;
    }
    const auto free_matrix = [&]() {
        gafime_gpu_matrix_free(matrix);
        matrix = nullptr;
    };
    if (spearman_require_status(
            gafime_gpu_matrix_upload(matrix, features.data(), target.data(), kRows, kColumns),
            "spearman_cache_matrix_upload"
        )) {
        free_matrix();
        return 1;
    }

    const uint32_t metric_id = GAFIME_METRIC_SPEARMAN;
    const uint32_t cached_combos[] = {0, 1, 0, 1};
    GafimeArityChunk cached_chunks[2]{};
    cached_chunks[0].arity = 1;
    cached_chunks[0].family = GAFIME_FAMILY_CONTINUOUS;
    cached_chunks[0].combo_count = 2;
    cached_chunks[0].descriptor_count = 2;
    cached_chunks[1].arity = 2;
    cached_chunks[1].family = GAFIME_FAMILY_CONTINUOUS;
    cached_chunks[1].combo_row_offset = 2;
    cached_chunks[1].combo_count = 1;
    cached_chunks[1].local_chunk_id = 1;
    cached_chunks[1].descriptor_offset = 2;
    cached_chunks[1].descriptor_count = 1;
    GafimeLaunchProtocol cached_protocol{};
    cached_protocol.abi_version = GAFIME_ABI_VERSION;
    cached_protocol.backend_kind = backend_kind;
    cached_protocol.max_arity = 2;
    cached_protocol.n_samples = kRows;
    cached_protocol.n_features = kColumns;
    cached_protocol.family_count = 1;
    cached_protocol.combo_indices = {cached_combos, 4};
    cached_protocol.metric_ids = {&metric_id, 1};
    cached_protocol.chunks = cached_chunks;
    cached_protocol.chunk_count = 2;

    SpearmanResultStorage cached_result(3, 2);
    if (spearman_require_status(
            gafime_gpu_execute(matrix, &cached_protocol, &cached_result.table),
            "spearman_cache_tied_batch"
        )) {
        free_matrix();
        return 1;
    }
    if (cached_result.table.row_count != 3 ||
        spearman_require_close(
            cached_result.metric_values[0],
            pairwise_spearman_reference(first_feature, target),
            "spearman_cache_tied_first"
        ) ||
        spearman_require_close(
            cached_result.metric_values[1],
            pairwise_spearman_reference(second_feature, target),
            "spearman_cache_tied_second"
        ) ||
        spearman_require_close(
            cached_result.metric_values[2],
            pairwise_spearman_reference(interaction, target),
            "spearman_pairwise_interaction"
        )) {
        free_matrix();
        return 1;
    }

    // One unary candidate falls below the cache crossover and must retain the
    // pairwise path. Compare it with the matching finite cached-batch result.
    const uint32_t single_combo = 0;
    GafimeArityChunk single_chunk{};
    single_chunk.arity = 1;
    single_chunk.family = GAFIME_FAMILY_CONTINUOUS;
    single_chunk.combo_count = 1;
    single_chunk.descriptor_count = 1;
    GafimeLaunchProtocol single_protocol = cached_protocol;
    single_protocol.max_arity = 1;
    single_protocol.combo_indices = {&single_combo, 1};
    single_protocol.chunks = &single_chunk;
    single_protocol.chunk_count = 1;
    SpearmanResultStorage single_result(1);
    if (spearman_require_status(
            gafime_gpu_execute(matrix, &single_protocol, &single_result.table),
            "spearman_pairwise_crossover"
        ) ||
        single_result.table.row_count != 1 ||
        spearman_require_close(
            single_result.metric_values[0],
            cached_result.metric_values[0],
            "spearman_pairwise_crossover_parity"
        )) {
        free_matrix();
        return 1;
    }

    std::vector<float> updated_target(kRows, 0.0f);
    for (uint64_t row = 0; row < kRows; ++row) {
        updated_target[row] = static_cast<float>((kRows - row + row / 9) % 13) - 6.0f;
    }
    if (spearman_require_status(
            gafime_gpu_matrix_update_target(matrix, updated_target.data(), kRows),
            "spearman_cache_target_update"
        ) ||
        spearman_require_status(
            gafime_gpu_execute(matrix, &cached_protocol, &cached_result.table),
            "spearman_cache_after_target_update"
        ) ||
        cached_result.table.row_count != 3 ||
        spearman_require_close(
            cached_result.metric_values[0],
            pairwise_spearman_reference(first_feature, updated_target),
            "spearman_cache_target_update_first"
        ) ||
        spearman_require_close(
            cached_result.metric_values[1],
            pairwise_spearman_reference(second_feature, updated_target),
            "spearman_cache_target_update_second"
        ) ||
        spearman_require_close(
            cached_result.metric_values[2],
            pairwise_spearman_reference(interaction, updated_target),
            "spearman_pairwise_interaction_target_update"
        )) {
        free_matrix();
        return 1;
    }

    std::vector<float> nonfinite_features = features;
    std::vector<float> nonfinite_first = first_feature;
    std::vector<float> nonfinite_second = second_feature;
    std::vector<float> nonfinite_target = updated_target;
    nonfinite_features[17 * kColumns] = std::numeric_limits<float>::quiet_NaN();
    nonfinite_first[17] = std::numeric_limits<float>::quiet_NaN();
    nonfinite_features[91 * kColumns + 1] = std::numeric_limits<float>::infinity();
    nonfinite_second[91] = std::numeric_limits<float>::infinity();
    nonfinite_target[43] = std::numeric_limits<float>::quiet_NaN();
    nonfinite_target[104] = -std::numeric_limits<float>::infinity();
    if (spearman_require_status(
            gafime_gpu_matrix_upload(
                matrix,
                nonfinite_features.data(),
                nonfinite_target.data(),
                kRows,
                kColumns
            ),
            "spearman_pairwise_nonfinite_upload"
        ) ||
        spearman_require_status(
            gafime_gpu_execute(matrix, &cached_protocol, &cached_result.table),
            "spearman_pairwise_nonfinite"
        ) ||
        cached_result.table.row_count != 3 ||
        spearman_require_close(
            cached_result.metric_values[0],
            pairwise_spearman_reference(nonfinite_first, nonfinite_target),
            "spearman_pairwise_nonfinite_first"
        ) ||
        spearman_require_close(
            cached_result.metric_values[1],
            pairwise_spearman_reference(nonfinite_second, nonfinite_target),
            "spearman_pairwise_nonfinite_second"
        )) {
        free_matrix();
        return 1;
    }

    free_matrix();
    return 0;
}

}  // namespace gafime_gpu_test
