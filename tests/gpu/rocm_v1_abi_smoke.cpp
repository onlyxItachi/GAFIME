#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "../../src/common/gafime_gpu_abi.hpp"
#include "spearman_cache_boundaries.hpp"

#ifndef GAFIME_EXPECT_MI_ACCUMULATION_FP64
#define GAFIME_EXPECT_MI_ACCUMULATION_FP64 0
#endif

namespace {

int require_status(int status, const char* label) {
    if (status != GAFIME_STATUS_OK) {
        std::fprintf(stderr, "%s failed with status %d\n", label, status);
        return 1;
    }
    return 0;
}

int require_close(float actual, float expected, const char* label) {
    if (std::fabs(actual - expected) > 1e-4f) {
        std::fprintf(stderr, "%s mismatch: actual=%f expected=%f\n", label, actual, expected);
        return 1;
    }
    return 0;
}

int verify_immutable_descriptor_generation(uint32_t backend_kind) {
    GafimeMatrixDesc desc{};
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 4;
    desc.cols = 3;
    desc.row_stride = 3;
    desc.bytes = 4 * 3 * sizeof(float);

    GafimeGpuMatrix matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix),
                       "descriptor_generation_matrix_alloc")) {
        return 1;
    }
    const float features[] = {
        1.0f, 5.0f, 1.0f,
        2.0f, 4.0f, 1.0f,
        3.0f, 3.0f, 1.0f,
        4.0f, 2.0f, 1.0f,
    };
    const float target[] = {1.0f, 2.0f, 3.0f, 4.0f};
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3),
                       "descriptor_generation_matrix_upload")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    uint32_t reused_combo = 0;
    const uint32_t metric_id = GAFIME_METRIC_PEARSON;
    GafimeArityChunk chunk{};
    chunk.arity = 1;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = 1;
    chunk.descriptor_count = 1;

    GafimeLaunchProtocol protocol{};
    protocol.abi_version = GAFIME_ABI_VERSION;
    protocol.backend_kind = backend_kind;
    protocol.flags = GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    protocol.max_arity = 1;
    protocol.n_samples = 4;
    protocol.n_features = 3;
    protocol.family_count = 1;
    protocol.combo_indices = {&reused_combo, 1};
    protocol.metric_ids = {&metric_id, 1};
    protocol.chunks = &chunk;
    protocol.chunk_count = 1;
    protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 101;

    uint32_t result_combo = UINT32_MAX;
    float result_metric = 0.0f;
    uint32_t rank = 0;
    uint32_t family = 0;
    uint64_t candidate_id = 0;
    uint32_t row_flag = 0;
    GafimeResultTable result{};
    result.abi_version = GAFIME_ABI_VERSION;
    result.max_arity = 1;
    result.metric_count = 1;
    result.capacity = 1;
    result.combo_indices = &result_combo;
    result.metric_values = &result_metric;
    result.ranks = &rank;
    result.families = &family;
    result.candidate_ids = &candidate_id;
    result.row_flags = &row_flag;

    auto execute_and_expect = [&](float expected, const char* label) {
        const int status = gafime_gpu_execute(matrix, &protocol, &result);
        return require_status(status, label) || require_close(result_metric, expected, label);
    };

    int failed = execute_and_expect(1.0f, "descriptor_generation_first");
    reused_combo = 1;
    protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 102;
    failed = failed || execute_and_expect(-1.0f, "descriptor_generation_reused_address");

    // Deliberately violate host immutability to observe that a valid replay does
    // not upload again: generation 102 must retain the feature-1 descriptor.
    reused_combo = 0;
    failed = failed || execute_and_expect(-1.0f, "descriptor_generation_replay");

    // Generation zero is the older same-ABI behavior and must upload every call.
    protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 0;
    failed = failed || execute_and_expect(1.0f, "descriptor_generation_legacy_first");
    reused_combo = 1;
    failed = failed || execute_and_expect(-1.0f, "descriptor_generation_legacy_repeat");

    gafime_gpu_matrix_free(matrix);
    return failed;
}

int verify_interaction_diagnostics() {
    constexpr uint64_t kRows = 4;
    constexpr uint32_t kCols = 5;
    GafimeMatrixDesc desc{};
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = kRows;
    desc.cols = kCols;
    desc.row_stride = kCols;
    desc.bytes = kRows * kCols * sizeof(float);

    GafimeGpuMatrix matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "diagnostics_matrix_alloc")) {
        return 1;
    }
    const float features[] = {
        -1.0e8f, -1.0e8f, -1.0e8f, -1.0e8f, -1.0e8f,
         1.0e8f,  1.0e8f,  1.0e8f,  1.0e8f,  1.0e8f,
         1.0f,    1.0f,    1.0f,    1.0f,    1.0f,
        -1.0f,   -1.0f,   -1.0f,   -1.0f,   -1.0f,
    };
    const float target[] = {0.0f, 1.0f, 2.0f, 3.0f};
    if (require_status(
            gafime_gpu_matrix_upload(matrix, features, target, kRows, kCols),
            "diagnostics_matrix_upload"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    const uint32_t combos[] = {
        0, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        0, 1,          UINT32_MAX, UINT32_MAX, UINT32_MAX,
        0, 1,          2,          UINT32_MAX, UINT32_MAX,
        0, 1,          2,          3,          UINT32_MAX,
        0, 1,          2,          3,          4,
    };
    std::vector<uint64_t> overflow_counts(5, UINT64_MAX);
    std::vector<uint32_t> flags(5, UINT32_MAX);
    GafimeInteractionDiagnosticBatch diagnostics{};
    diagnostics.abi_version = GAFIME_ABI_VERSION;
    diagnostics.max_arity = 5;
    diagnostics.row_count = 5;
    diagnostics.combo_indices = combos;
    diagnostics.combo_index_count = 25;
    diagnostics.overflow_row_counts = overflow_counts.data();
    diagnostics.flags = flags.data();
    if (require_status(
            gafime_gpu_interaction_diagnostics(matrix, &diagnostics),
            "interaction_diagnostics_partial_overflow"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const uint64_t expected_overflow[] = {0, 0, 0, 0, 2};
    for (size_t idx = 0; idx < overflow_counts.size(); ++idx) {
        if (overflow_counts[idx] != expected_overflow[idx] || flags[idx] != 0) {
            std::fprintf(
                stderr,
                "unexpected finite interaction diagnostics at %zu: count=%llu flags=%u\n",
                idx,
                static_cast<unsigned long long>(overflow_counts[idx]),
                flags[idx]
            );
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
    }

    GafimeGpuMatrix zero_prefix_matrix = nullptr;
    if (require_status(
            gafime_gpu_matrix_alloc(0, &desc, &zero_prefix_matrix),
            "zero_prefix_diagnostics_matrix_alloc"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const float max_finite = std::numeric_limits<float>::max();
    const float zero_prefix_features[] = {
        0.0f,  max_finite, 0.0f, 0.0f, 0.0f,
        0.0f, -max_finite, 0.0f, 0.0f, 0.0f,
        0.0f, -max_finite, 0.0f, 0.0f, 0.0f,
        0.0f, -max_finite, 0.0f, 0.0f, 0.0f,
    };
    const float zero_target[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t zero_prefix_combo[] = {
        0, 1, UINT32_MAX, UINT32_MAX, UINT32_MAX,
    };
    uint64_t zero_prefix_count = UINT64_MAX;
    uint32_t zero_prefix_flag = UINT32_MAX;
    GafimeInteractionDiagnosticBatch zero_prefix_diagnostics{};
    zero_prefix_diagnostics.abi_version = GAFIME_ABI_VERSION;
    zero_prefix_diagnostics.max_arity = 5;
    zero_prefix_diagnostics.row_count = 1;
    zero_prefix_diagnostics.combo_indices = zero_prefix_combo;
    zero_prefix_diagnostics.combo_index_count = 5;
    zero_prefix_diagnostics.overflow_row_counts = &zero_prefix_count;
    zero_prefix_diagnostics.flags = &zero_prefix_flag;
    const bool zero_prefix_failed =
        require_status(
            gafime_gpu_matrix_upload(
                zero_prefix_matrix,
                zero_prefix_features,
                zero_target,
                kRows,
                kCols
            ),
            "zero_prefix_diagnostics_matrix_upload"
        ) ||
        require_status(
            gafime_gpu_interaction_diagnostics(
                zero_prefix_matrix,
                &zero_prefix_diagnostics
            ),
            "zero_prefix_interaction_diagnostics"
        ) ||
        zero_prefix_count != 1 || zero_prefix_flag != 0;
    gafime_gpu_matrix_free(zero_prefix_matrix);
    if (zero_prefix_failed) {
        std::fprintf(stderr, "zero prefix hid later centered overflow\n");
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    const float target_with_nan[] = {NAN, 1.0f, 2.0f, 3.0f};
    if (require_status(
            gafime_gpu_matrix_update_target(matrix, target_with_nan, kRows),
            "diagnostics_target_nonfinite"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    if (require_status(
            gafime_gpu_interaction_diagnostics(matrix, &diagnostics),
            "interaction_diagnostics_target_nonfinite"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    for (size_t idx = 0; idx < overflow_counts.size(); ++idx) {
        if (overflow_counts[idx] != expected_overflow[idx] ||
            flags[idx] != GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE) {
            std::fprintf(stderr, "target non-finite diagnostics mismatch at %zu\n", idx);
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
    }

    std::vector<float> source_nonfinite_features(features, features + kRows * kCols);
    source_nonfinite_features[0] = NAN;
    if (require_status(
            gafime_gpu_matrix_upload(
                matrix,
                source_nonfinite_features.data(),
                target,
                kRows,
                kCols
            ),
            "diagnostics_feature_nonfinite"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const uint32_t source_combos[] = {
        0, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        1, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
    };
    std::vector<uint64_t> source_counts(2, UINT64_MAX);
    std::vector<uint32_t> source_flags(2, UINT32_MAX);
    GafimeInteractionDiagnosticBatch source_diagnostics{};
    source_diagnostics.abi_version = GAFIME_ABI_VERSION;
    source_diagnostics.max_arity = 5;
    source_diagnostics.row_count = 2;
    source_diagnostics.combo_indices = source_combos;
    source_diagnostics.combo_index_count = 10;
    source_diagnostics.overflow_row_counts = source_counts.data();
    source_diagnostics.flags = source_flags.data();
    if (require_status(
            gafime_gpu_interaction_diagnostics(matrix, &source_diagnostics),
            "interaction_diagnostics_feature_nonfinite"
        ) ||
        source_counts[0] != 0 || source_counts[1] != 0 ||
        source_flags[0] != GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE ||
        source_flags[1] != 0) {
        std::fprintf(stderr, "candidate-specific source non-finite diagnostics mismatch\n");
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    const uint32_t malformed_combo[] = {0, UINT32_MAX, 1, UINT32_MAX, UINT32_MAX};
    uint64_t malformed_count = UINT64_MAX;
    uint32_t malformed_flag = UINT32_MAX;
    GafimeInteractionDiagnosticBatch malformed{};
    malformed.abi_version = GAFIME_ABI_VERSION;
    malformed.max_arity = 5;
    malformed.row_count = 1;
    malformed.combo_indices = malformed_combo;
    malformed.combo_index_count = 5;
    malformed.overflow_row_counts = &malformed_count;
    malformed.flags = &malformed_flag;
    const int malformed_status = gafime_gpu_interaction_diagnostics(matrix, &malformed);
    gafime_gpu_matrix_free(matrix);
    if (malformed_status != GAFIME_STATUS_INVALID_ARGUMENT ||
        malformed_count != UINT64_MAX || malformed_flag != UINT32_MAX) {
        std::fprintf(stderr, "interaction diagnostics accepted malformed padding\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    GafimeGpuDeviceInfo info{};
    if (require_status(gafime_gpu_device_info(0, &info), "device_info")) {
        return 1;
    }
    if (info.backend_kind != GAFIME_BACKEND_ROCM || info.abi_version != GAFIME_ABI_VERSION) {
        std::fprintf(stderr, "device_info returned invalid ROCm ABI metadata\n");
        return 1;
    }
    const bool mi_accumulation_fp64 =
        (info.flags & GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64) != 0;
    if (mi_accumulation_fp64 != (GAFIME_EXPECT_MI_ACCUMULATION_FP64 != 0)) {
        std::fprintf(stderr, "ROCm payload reported the wrong MI accumulation policy\n");
        return 1;
    }
    if ((info.flags & GAFIME_GPU_DEVICE_FLAG_F64_STORAGE) != 0) {
        std::fprintf(stderr, "ROCm payload advertises unimplemented f64 storage\n");
        return 1;
    }
    GafimeMatrixDesc f64_desc{};
    f64_desc.abi_version = GAFIME_ABI_VERSION;
    f64_desc.dtype = GAFIME_DTYPE_F64;
    f64_desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    f64_desc.rows = 1;
    f64_desc.cols = 1;
    f64_desc.row_stride = 1;
    f64_desc.bytes = sizeof(double);
    GafimeGpuMatrix unsupported_f64_matrix = nullptr;
    if (gafime_gpu_matrix_alloc(0, &f64_desc, &unsupported_f64_matrix) !=
            GAFIME_STATUS_UNSUPPORTED_BACKEND ||
        unsupported_f64_matrix != nullptr) {
        std::fprintf(stderr, "ROCm payload did not fail closed on f64 storage\n");
        return 1;
    }
    if (verify_immutable_descriptor_generation(GAFIME_BACKEND_ROCM)) {
        return 1;
    }
    if (verify_interaction_diagnostics()) {
        return 1;
    }
    if (gafime_gpu_test::verify_spearman_cache_boundaries(GAFIME_BACKEND_ROCM)) {
        return 1;
    }

    GafimeMatrixDesc desc{};
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = 4;
    desc.cols = 3;
    desc.row_stride = 3;
    desc.bytes = 4 * 3 * sizeof(float);

    GafimeGpuMatrix matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc")) {
        return 1;
    }

    const float features[] = {
        1.0f, 5.0f, 1.0f,
        2.0f, 4.0f, 1.0f,
        3.0f, 3.0f, 1.0f,
        4.0f, 2.0f, 1.0f,
    };
    const float target[] = {1.0f, 2.0f, 3.0f, 4.0f};
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3), "matrix_upload")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    const uint32_t combos[] = {
        0, 1, 2,
        0, 1,
        0, 2,
        1, 2,
    };
    const uint32_t metric_ids[] = {GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};
    GafimeArityChunk chunks[2]{};
    chunks[0].arity = 1;
    chunks[0].family = GAFIME_FAMILY_CONTINUOUS;
    chunks[0].combo_count = 3;
    chunks[0].descriptor_offset = 0;
    chunks[0].descriptor_count = 3;
    chunks[1].arity = 2;
    chunks[1].family = GAFIME_FAMILY_CONTINUOUS;
    chunks[1].combo_count = 3;
    chunks[1].combo_row_offset = 3;
    chunks[1].descriptor_offset = 3;
    chunks[1].descriptor_count = 3;
    chunks[1].local_chunk_id = 1;

    GafimeLaunchProtocol protocol{};
    protocol.abi_version = GAFIME_ABI_VERSION;
    protocol.backend_kind = GAFIME_BACKEND_ROCM;
    protocol.max_arity = 2;
    protocol.n_samples = 4;
    protocol.n_features = 3;
    protocol.family_count = 1;
    protocol.combo_indices = {combos, 9};
    protocol.metric_ids = {metric_ids, 2};
    protocol.chunks = chunks;
    protocol.chunk_count = 2;

    std::vector<uint32_t> result_combos(6 * 2, UINT32_MAX);
    std::vector<float> result_metrics(6 * 2, 0.0f);
    std::vector<uint32_t> ranks(6, 0);
    std::vector<uint32_t> families(6, 0);
    std::vector<uint64_t> candidate_ids(6, 0);
    std::vector<uint32_t> row_flags(6, 0);

    GafimeResultTable result{};
    result.abi_version = GAFIME_ABI_VERSION;
    result.max_arity = 2;
    result.metric_count = 2;
    result.capacity = 6;
    result.combo_indices = result_combos.data();
    result.metric_values = result_metrics.data();
    result.ranks = ranks.data();
    result.families = families.data();
    result.candidate_ids = candidate_ids.data();
    result.row_flags = row_flags.data();

    uint64_t initial_execution_peak = 0;
    if (require_status(
            gafime_gpu_execution_memory_peak(matrix, &protocol, &initial_execution_peak),
            "execution_memory_peak") ||
        initial_execution_peak <= desc.bytes) {
        std::fprintf(stderr, "execution-memory preflight omitted resident storage\n");
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    int status = gafime_gpu_execute(matrix, &protocol, &result);
    uint64_t resident_execution_peak = 0;
    if (status == GAFIME_STATUS_OK) {
        status = gafime_gpu_execution_memory_peak(
            matrix,
            &protocol,
            &resident_execution_peak
        );
    }
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute")) {
        return 1;
    }
    if (resident_execution_peak > initial_execution_peak) {
        std::fprintf(stderr, "resident execution peak exceeded its cold preflight\n");
        return 1;
    }
    if (result.row_count != 6) {
        std::fprintf(stderr, "unexpected result row count: %llu\n",
                     static_cast<unsigned long long>(result.row_count));
        return 1;
    }
    if (require_close(result_metrics[0], 1.0f, "feature0 pearson")) {
        return 1;
    }
    if (require_close(result_metrics[1], 1.0f, "feature0 r2")) {
        return 1;
    }
    if (require_close(result_metrics[2], -1.0f, "feature1 pearson")) {
        return 1;
    }
    if (require_close(result_metrics[3], 1.0f, "feature1 r2")) {
        return 1;
    }

    GafimeLaunchProtocol permutation_protocol = protocol;
    permutation_protocol.permutations.permutation_count = 2;
    permutation_protocol.permutations.seed = 99;
    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc_permutation")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3), "matrix_upload_permutation")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const int permutation_status = gafime_gpu_execute(matrix, &permutation_protocol, &result);
    gafime_gpu_matrix_free(matrix);
    if (permutation_status != GAFIME_STATUS_GRAPH_UNSUPPORTED) {
        std::fprintf(stderr, "expected ROCm permutation graph unsupported, got %d\n", permutation_status);
        return 1;
    }

    GafimeMatrixDesc stable_desc = desc;
    stable_desc.rows = 256;
    stable_desc.cols = 1;
    stable_desc.row_stride = 1;
    stable_desc.bytes = 256 * sizeof(float);
    std::vector<float> stable_features;
    std::vector<float> stable_target;
    stable_features.reserve(256);
    stable_target.reserve(256);
    for (int row = 0; row < 256; ++row) {
        const float base = static_cast<float>(row % 7) * 0.125f;
        stable_features.push_back(100000.0f + base);
        stable_target.push_back(-300000.0f + base * 2.0f);
    }
    const uint32_t stable_combos[] = {0};
    GafimeArityChunk stable_chunk{};
    stable_chunk.arity = 1;
    stable_chunk.family = GAFIME_FAMILY_CONTINUOUS;
    stable_chunk.combo_count = 1;
    stable_chunk.descriptor_offset = 0;
    stable_chunk.descriptor_count = 1;
    GafimeLaunchProtocol stable_protocol{};
    stable_protocol.abi_version = GAFIME_ABI_VERSION;
    stable_protocol.backend_kind = GAFIME_BACKEND_ROCM;
    stable_protocol.max_arity = 1;
    stable_protocol.n_samples = 256;
    stable_protocol.n_features = 1;
    stable_protocol.family_count = 1;
    stable_protocol.combo_indices = {stable_combos, 1};
    stable_protocol.metric_ids = {metric_ids, 2};
    stable_protocol.chunks = &stable_chunk;
    stable_protocol.chunk_count = 1;
    std::vector<uint32_t> stable_result_combos(1, UINT32_MAX);
    std::vector<float> stable_result_metrics(2, 0.0f);
    std::vector<uint32_t> stable_ranks(1, 0);
    std::vector<uint32_t> stable_families(1, 0);
    std::vector<uint64_t> stable_candidate_ids(1, 0);
    std::vector<uint32_t> stable_row_flags(1, 0);
    GafimeResultTable stable_result{};
    stable_result.abi_version = GAFIME_ABI_VERSION;
    stable_result.max_arity = 1;
    stable_result.metric_count = 2;
    stable_result.capacity = 1;
    stable_result.combo_indices = stable_result_combos.data();
    stable_result.metric_values = stable_result_metrics.data();
    stable_result.ranks = stable_ranks.data();
    stable_result.families = stable_families.data();
    stable_result.candidate_ids = stable_candidate_ids.data();
    stable_result.row_flags = stable_row_flags.data();
    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &stable_desc, &matrix), "matrix_alloc_stable")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, stable_features.data(), stable_target.data(), 256, 1),
                       "matrix_upload_stable")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const int stable_status = gafime_gpu_execute(matrix, &stable_protocol, &stable_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(stable_status, "gpu_execute_stable")) {
        return 1;
    }
    if (require_close(stable_result_metrics[0], 1.0f, "high-offset pearson")) {
        return 1;
    }
    if (require_close(stable_result_metrics[1], 1.0f, "high-offset r2")) {
        return 1;
    }
    return 0;
}
