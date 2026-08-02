#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../common/gafime_gpu_abi.hpp"

namespace {

constexpr uint64_t kRows = 4;
constexpr uint32_t kCols = 2;
constexpr uint32_t kMetricCount = 4;
constexpr uint32_t kCandidateCount = 2;

int require_status(int status, const char* label) {
    if (status != GAFIME_STATUS_OK) {
        std::fprintf(stderr, "%s failed with status %d\n", label, status);
        return 1;
    }
    return 0;
}

int require_true(bool condition, const char* label) {
    if (!condition) {
        std::fprintf(stderr, "%s failed\n", label);
        return 1;
    }
    return 0;
}

struct ProtocolFixture {
    uint32_t combos[kCandidateCount] = {0u, 1u};
    uint32_t metrics[kMetricCount] = {
        GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_SPEARMAN,
        GAFIME_METRIC_MUTUAL_INFO,
        GAFIME_METRIC_R2,
    };
    GafimeShapeHint hint{};
    GafimeArityChunk chunk{};
    GafimeLaunchProtocol base{};

    ProtocolFixture() {
        hint.vendor_hint = 2;
        chunk.arity = 1;
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.shape_hint_index = 0;
        chunk.combo_count = kCandidateCount;
        chunk.descriptor_count = kCandidateCount;
        chunk.local_chunk_id = 0;
        base.abi_version = GAFIME_ABI_VERSION;
        base.backend_kind = GAFIME_BACKEND_ROCM;
        base.flags = GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
        base.max_arity = 1;
        base.n_samples = kRows;
        base.n_features = kCols;
        base.family_count = 1;
        base.combo_indices = {combos, kCandidateCount};
        base.metric_ids = {metrics, kMetricCount};
        base.chunks = &chunk;
        base.chunk_count = 1;
        base.shape_hints = &hint;
        base.shape_hint_count = 1;
        base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 919;
    }

    GafimePrecisionLaunchProtocol precision(uint32_t profile) const {
        GafimePrecisionLaunchProtocol protocol{};
        protocol.abi_version = GAFIME_PRECISION_ABI_VERSION;
        protocol.profile = profile;
        protocol.base = &base;
        return protocol;
    }
};

struct ResultF32 {
    uint32_t combos[kCandidateCount]{};
    float values[kCandidateCount * kMetricCount]{};
    uint32_t ranks[kCandidateCount]{};
    uint32_t families[kCandidateCount]{};
    uint64_t ids[kCandidateCount]{};
    uint32_t flags[kCandidateCount]{};
    GafimeResultTable table{};

    ResultF32() {
        table.abi_version = GAFIME_ABI_VERSION;
        table.max_arity = 1;
        table.metric_count = kMetricCount;
        table.capacity = kCandidateCount;
        table.combo_indices = combos;
        table.metric_values = values;
        table.ranks = ranks;
        table.families = families;
        table.candidate_ids = ids;
        table.row_flags = flags;
    }
};

struct ResultF64 {
    uint32_t combos[kCandidateCount]{};
    double values[kCandidateCount * kMetricCount]{};
    uint32_t ranks[kCandidateCount]{};
    uint32_t families[kCandidateCount]{};
    uint64_t ids[kCandidateCount]{};
    uint32_t flags[kCandidateCount]{};
    GafimeResultTableF64 table{};

    ResultF64() {
        table.abi_version = GAFIME_PRECISION_ABI_VERSION;
        table.max_arity = 1;
        table.metric_count = kMetricCount;
        table.capacity = kCandidateCount;
        table.combo_indices = combos;
        table.metric_values = values;
        table.ranks = ranks;
        table.families = families;
        table.candidate_ids = ids;
        table.row_flags = flags;
    }
};

GafimePrecisionMatrixDesc matrix_desc(uint32_t profile, uint32_t dtype) {
    GafimePrecisionMatrixDesc desc{};
    desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
    desc.profile = profile;
    desc.dtype = dtype;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = kRows;
    desc.cols = kCols;
    desc.row_stride = kCols;
    desc.bytes = kRows * kCols * (dtype == GAFIME_DTYPE_F32 ? sizeof(float) : sizeof(double));
    return desc;
}

int run_fp32_and_mixed(
    uint32_t profile,
    const float* features,
    const float* target,
    ProtocolFixture* fixture,
    ResultF32* fp32_result,
    ResultF64* f64_result,
    uint64_t* peak_out
) {
    GafimeGpuMatrix matrix = nullptr;
    const auto desc = matrix_desc(profile, GAFIME_DTYPE_F32);
    int failed = require_status(
        gafime_gpu_matrix_alloc_v2(0, &desc, &matrix), "precision_f32_matrix_alloc");
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_upload_f32_v2(matrix, features, target, kRows, kCols),
            "precision_f32_matrix_upload");
    }
    const GafimePrecisionLaunchProtocol protocol = fixture->precision(profile);
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_execute")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_execute");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execution_memory_peak_v2(matrix, &protocol, peak_out),
            "precision_f32_memory_peak");
    }
    // Same descriptor generation and pointer, now captured/replayed.  It must
    // stay matrix-local even though fp32/mixed/fp64 use the same generation.
    fixture->base.flags |= GAFIME_LAUNCH_FLAG_GRAPH;
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_graph")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_graph");
    }
    fixture->base.flags &= ~GAFIME_LAUNCH_FLAG_GRAPH;
    if (!failed) {
        const uint32_t flags = profile == GAFIME_PRECISION_FP32
            ? fp32_result->table.flags
            : f64_result->table.flags;
        failed = require_true(
            (flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "precision_graph_replay_profile_local");
    }
    // ROCm significance is orchestrated by Rust using target replacement and
    // device execution/ranking.  The payload intentionally has no compact
    // native permutation ABI, so validate the supported primitive and restore
    // the baseline resident target before returning.
    if (!failed) {
        const float replacement[kRows] = {3.0f, 2.0f, 1.0f, 0.0f};
        failed = require_status(
            gafime_gpu_matrix_update_target_f32_v2(matrix, replacement, kRows),
            "precision_f32_target_replacement");
    }
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_replacement_execute")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_replacement_execute");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_update_target_f32_v2(matrix, target, kRows),
            "precision_f32_target_restore");
    }
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_restored_execute")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_restored_execute");
    }
    // The high target requires scaled covariance in fp32 (its square would
    // overflow), while target replacement below returns the same descriptors
    // to the unscaled path. Both routes must remain finite and profile-typed.
    const float covariance_features[kRows * kCols] = {
        -3.0f, 0.0f,
        -1.0f, 1.0f,
        1.0f, 2.0f,
        3.0f, 3.0f,
    };
    const float scaled_target[kRows] = {-3.0e30f, -1.0e30f, 1.0e30f, 3.0e30f};
    const float unscaled_target[kRows] = {-3.0f, -1.0f, 1.0f, 3.0f};
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_upload_f32_v2(
                matrix, covariance_features, scaled_target, kRows, kCols),
            "precision_f32_scaled_covariance_upload");
    }
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_scaled_covariance_execute")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_scaled_covariance_execute");
    }
    if (!failed) {
        const double pearson = profile == GAFIME_PRECISION_FP32
            ? static_cast<double>(fp32_result->values[0]) : f64_result->values[0];
        failed = require_true(
            std::isfinite(pearson) && pearson > 0.99,
            "precision_f32_scaled_covariance_finite");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_update_target_f32_v2(matrix, unscaled_target, kRows),
            "precision_f32_covariance_mode_replacement");
    }
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_unscaled_covariance_execute")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_unscaled_covariance_execute");
    }
    if (!failed) {
        const double pearson = profile == GAFIME_PRECISION_FP32
            ? static_cast<double>(fp32_result->values[0]) : f64_result->values[0];
        failed = require_true(
            std::isfinite(pearson) && pearson > 0.99,
            "precision_f32_unscaled_covariance_finite");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_upload_f32_v2(matrix, features, target, kRows, kCols),
            "precision_f32_baseline_restore");
    }
    if (!failed) {
        failed = profile == GAFIME_PRECISION_FP32
            ? require_status(gafime_gpu_execute_f32_v2(matrix, &protocol, &fp32_result->table),
                             "precision_fp32_baseline_restore_execute")
            : require_status(gafime_gpu_execute_f64_v2(matrix, &protocol, &f64_result->table),
                             "precision_mixed_baseline_restore_execute");
    }
    gafime_gpu_matrix_free(matrix);
    return failed;
}

int run_fp64(
    const double* features,
    const double* target,
    ProtocolFixture* fixture,
    ResultF64* result,
    uint64_t* peak_out
) {
    GafimeGpuMatrix matrix = nullptr;
    const auto desc = matrix_desc(GAFIME_PRECISION_FP64, GAFIME_DTYPE_F64);
    int failed = require_status(
        gafime_gpu_matrix_alloc_v2(0, &desc, &matrix), "precision_fp64_matrix_alloc");
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_upload_f64_v2(matrix, features, target, kRows, kCols),
            "precision_fp64_matrix_upload");
    }
    GafimePrecisionLaunchProtocol protocol = fixture->precision(GAFIME_PRECISION_FP64);
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table), "precision_fp64_execute");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execution_memory_peak_v2(matrix, &protocol, peak_out),
            "precision_fp64_memory_peak");
    }
    if (!failed) {
        failed = require_true(
            std::fabs(result->values[0]) > 0.99,
            "fp64_preserves_values_that_collapse_to_fp32");
    }
    // Diagnostics share their public ABI-1.0 batch shape, but must dispatch
    // from an ABI-1.1 precision matrix without reinterpreting the f64 storage
    // as a legacy float matrix.
    if (!failed) {
        uint32_t diagnostic_combo[] = {0u};
        uint64_t overflow_rows[] = {UINT64_C(99)};
        uint32_t diagnostic_flags[] = {UINT32_MAX};
        GafimeInteractionDiagnosticBatch diagnostics{};
        diagnostics.abi_version = GAFIME_ABI_VERSION;
        diagnostics.max_arity = 1;
        diagnostics.row_count = 1;
        diagnostics.combo_indices = diagnostic_combo;
        diagnostics.combo_index_count = 1;
        diagnostics.overflow_row_counts = overflow_rows;
        diagnostics.flags = diagnostic_flags;
        failed = require_status(
            gafime_gpu_interaction_diagnostics(matrix, &diagnostics),
            "precision_fp64_interaction_diagnostics");
        if (!failed) {
            failed = require_true(
                overflow_rows[0] == 0 && diagnostic_flags[0] == 0,
                "precision_fp64_interaction_diagnostics_values");
        }
    }
    // Route the same resident numeric family through the non-continuous
    // descriptors too; planning stays integer and output family identity is
    // preserved by the precision result writer.
    fixture->chunk.family = GAFIME_FAMILY_TIME_SERIES;
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table), "precision_time_series_execute");
    }
    if (!failed) {
        failed = require_true(
            result->families[0] == GAFIME_FAMILY_TIME_SERIES,
            "precision_time_series_family_identity");
    }
    fixture->chunk.family = GAFIME_FAMILY_DECISION_PATH;
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table), "precision_decision_path_execute");
    }
    if (!failed) {
        failed = require_true(
            result->families[0] == GAFIME_FAMILY_DECISION_PATH,
            "precision_decision_path_family_identity");
    }
    fixture->chunk.family = GAFIME_FAMILY_CONTINUOUS;
    if (!failed) {
        const double replacement[kRows] = {3.0, 2.0, 1.0, 0.0};
        failed = require_status(
            gafime_gpu_matrix_update_target_f64_v2(matrix, replacement, kRows),
            "precision_fp64_target_replacement");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table),
            "precision_fp64_replacement_execute");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_update_target_f64_v2(matrix, target, kRows),
            "precision_fp64_target_restore");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table),
            "precision_fp64_restored_execute");
    }
    const double covariance_features[kRows * kCols] = {
        -3.0, 0.0,
        -1.0, 1.0,
        1.0, 2.0,
        3.0, 3.0,
    };
    const double scaled_target[kRows] = {-3.0e200, -1.0e200, 1.0e200, 3.0e200};
    const double unscaled_target[kRows] = {-3.0, -1.0, 1.0, 3.0};
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_upload_f64_v2(
                matrix, covariance_features, scaled_target, kRows, kCols),
            "precision_fp64_scaled_covariance_upload");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table),
            "precision_fp64_scaled_covariance_execute");
    }
    if (!failed) {
        failed = require_true(
            std::isfinite(result->values[0]) && result->values[0] > 0.99,
            "precision_fp64_scaled_covariance_finite");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_update_target_f64_v2(matrix, unscaled_target, kRows),
            "precision_fp64_covariance_mode_replacement");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table),
            "precision_fp64_unscaled_covariance_execute");
    }
    if (!failed) {
        failed = require_true(
            std::isfinite(result->values[0]) && result->values[0] > 0.99,
            "precision_fp64_unscaled_covariance_finite");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_matrix_upload_f64_v2(matrix, features, target, kRows, kCols),
            "precision_fp64_baseline_restore");
    }
    if (!failed) {
        failed = require_status(
            gafime_gpu_execute_f64_v2(matrix, &protocol, &result->table),
            "precision_fp64_baseline_restore_execute");
    }
    gafime_gpu_matrix_free(matrix);
    return failed;
}

}  // namespace

int main() {
    GafimePrecisionCapabilities capabilities{};
    int failed = require_status(
        gafime_gpu_precision_capabilities(0, &capabilities), "precision_capabilities");
    if (!failed) {
        failed = require_true(
            capabilities.abi_version == GAFIME_PRECISION_ABI_VERSION &&
            capabilities.backend_kind == GAFIME_BACKEND_ROCM &&
            capabilities.profile_mask == (GAFIME_PRECISION_PROFILE_MASK_FP32 |
                GAFIME_PRECISION_PROFILE_MASK_MIXED | GAFIME_PRECISION_PROFILE_MASK_FP64) &&
            capabilities.storage_dtype_mask == (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64) &&
            capabilities.result_dtype_mask == (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64),
            "precision_capability_matrix");
    }

    constexpr double epsilon = 0x1p-30;
    const double features_f64[] = {
        1.0 + 0.0 * epsilon, 3.0,
        1.0 + 1.0 * epsilon, 2.0,
        1.0 + 2.0 * epsilon, 1.0,
        1.0 + 3.0 * epsilon, 0.0,
    };
    const double target_f64[] = {0.0, 1.0, 2.0, 3.0};
    float features_f32[kRows * kCols]{};
    float target_f32[kRows]{};
    for (uint64_t index = 0; index < kRows * kCols; ++index) {
        features_f32[index] = static_cast<float>(features_f64[index]);
    }
    for (uint64_t index = 0; index < kRows; ++index) {
        target_f32[index] = static_cast<float>(target_f64[index]);
    }

    ProtocolFixture fixture;
    ResultF32 fp32_result;
    ResultF64 mixed_result;
    ResultF64 fp64_result;
    uint64_t fp32_peak = 0;
    uint64_t mixed_peak = 0;
    uint64_t fp64_peak = 0;
    if (!failed) {
        failed = run_fp32_and_mixed(
            GAFIME_PRECISION_FP32, features_f32, target_f32, &fixture,
            &fp32_result, &mixed_result, &fp32_peak);
    }
    if (!failed) {
        failed = run_fp32_and_mixed(
            GAFIME_PRECISION_MIXED, features_f32, target_f32, &fixture,
            &fp32_result, &mixed_result, &mixed_peak);
    }
    if (!failed) {
        failed = run_fp64(features_f64, target_f64, &fixture, &fp64_result, &fp64_peak);
    }
    if (!failed) {
        failed = require_true(
            std::fabs(fp32_result.values[0]) < 0.1f &&
            std::fabs(mixed_result.values[0]) < 0.1 &&
            std::fabs(fp64_result.values[0]) > 0.99,
            "profile_storage_separation");
    }
    if (!failed) {
        failed = require_true(
            fp32_peak < mixed_peak && mixed_peak < fp64_peak,
            "profile_typed_memory_peak_separation");
    }
    return failed;
}
