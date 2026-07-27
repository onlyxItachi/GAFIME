#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

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

class ScopedEnvironmentOverride {
public:
    explicit ScopedEnvironmentOverride(const char* name) : name_(name) {
        const char* value = std::getenv(name_);
        had_value_ = value != nullptr;
        if (had_value_) {
            value_ = value;
        }
    }

    ~ScopedEnvironmentOverride() {
        if (had_value_) {
            static_cast<void>(set_value(value_.c_str()));
        } else {
            static_cast<void>(clear_value());
        }
    }

    bool set(const char* value) {
        return set_value(value) == 0;
    }

private:
    int set_value(const char* value) const {
#if defined(_WIN32)
        return _putenv_s(name_, value);
#else
        return setenv(name_, value, 1);
#endif
    }

    int clear_value() const {
#if defined(_WIN32)
        return _putenv_s(name_, "");
#else
        return unsetenv(name_);
#endif
    }

    const char* name_;
    bool had_value_ = false;
    std::string value_;
};

class DecisionPathStateCleanup {
public:
    explicit DecisionPathStateCleanup(uint32_t device_id) : device_id_(device_id) {}

    ~DecisionPathStateCleanup() {
        if (active_) {
            static_cast<void>(gafime_gpu_decision_path_release_device_state(device_id_));
        }
    }

    int release() {
        active_ = false;
        return gafime_gpu_decision_path_release_device_state(device_id_);
    }

private:
    uint32_t device_id_;
    bool active_ = true;
};

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
    int cuda_device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&cuda_device_count);
    if (cuda_status == cudaErrorNoDevice || cuda_status == cudaErrorInsufficientDriver ||
        cuda_device_count == 0) {
        return 77;
    }
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }
    GafimeGpuDeviceInfo info{};
    if (require_status(gafime_gpu_device_info(0, &info), "device_info")) {
        return 1;
    }
    if (info.backend_kind != GAFIME_BACKEND_CUDA || info.abi_version != GAFIME_ABI_VERSION) {
        std::fprintf(stderr, "device_info returned invalid ABI metadata\n");
        return 1;
    }
    const bool mi_accumulation_fp64 =
        (info.flags & GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64) != 0;
    if (mi_accumulation_fp64 != (GAFIME_EXPECT_MI_ACCUMULATION_FP64 != 0)) {
        std::fprintf(stderr, "CUDA payload reported the wrong MI accumulation policy\n");
        return 1;
    }
    if ((info.flags & GAFIME_GPU_DEVICE_FLAG_F64_STORAGE) != 0) {
        std::fprintf(stderr, "CUDA payload advertises unimplemented f64 storage\n");
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
        std::fprintf(stderr, "CUDA payload did not fail closed on f64 storage\n");
        return 1;
    }
    DecisionPathStateCleanup decision_path_state_cleanup(0u);
    const bool optix_rt = (info.flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) != 0;
    if (std::getenv("GAFIME_CUDA_EXPECT_NO_RT") != nullptr && optix_rt) {
        std::fprintf(stderr, "RT-disabled CUDA payload unexpectedly advertises OptiX RT\n");
        return 1;
    }
    if (std::getenv("GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP") != nullptr && !optix_rt) {
        std::fprintf(stderr, "RT-required CUDA payload does not advertise OptiX RT\n");
        return 1;
    }
    GafimeGpuGraphCapability graph_capability{};
    if (require_status(gafime_gpu_graph_capability(0, &graph_capability), "graph_capability")) {
        return 1;
    }
    if (graph_capability.graph_mode != GAFIME_GRAPH_STREAM_CAPTURE ||
        graph_capability.supports_device_ranking != 1) {
        std::fprintf(stderr, "unexpected graph capability metadata\n");
        return 1;
    }
    if (verify_immutable_descriptor_generation(GAFIME_BACKEND_CUDA)) {
        return 1;
    }
    if (verify_interaction_diagnostics()) {
        return 1;
    }
    if (gafime_gpu_test::verify_spearman_cache_boundaries(GAFIME_BACKEND_CUDA)) {
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

    const uint32_t combos[] = {0, 1, 2};
    const uint32_t metric_ids[] = {GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};
    GafimeArityChunk chunk{};
    chunk.arity = 1;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = 3;
    chunk.descriptor_offset = 0;
    chunk.descriptor_count = 3;

    GafimeLaunchProtocol protocol{};
    protocol.abi_version = GAFIME_ABI_VERSION;
    protocol.backend_kind = GAFIME_BACKEND_CUDA;
    protocol.max_arity = 1;
    protocol.n_samples = 4;
    protocol.n_features = 3;
    protocol.family_count = 1;
    protocol.combo_indices = {combos, 3};
    protocol.metric_ids = {metric_ids, 2};
    protocol.chunks = &chunk;
    protocol.chunk_count = 1;

    std::vector<uint32_t> result_combos(3, UINT32_MAX);
    std::vector<float> result_metrics(3 * 2, 0.0f);
    std::vector<uint32_t> ranks(3, 0);
    std::vector<uint32_t> families(3, 0);
    std::vector<uint64_t> candidate_ids(3, 0);
    std::vector<uint32_t> row_flags(3, 0);

    GafimeResultTable result{};
    result.abi_version = GAFIME_ABI_VERSION;
    result.max_arity = 1;
    result.metric_count = 2;
    result.capacity = 3;
    result.combo_indices = result_combos.data();
    result.metric_values = result_metrics.data();
    result.ranks = ranks.data();
    result.families = families.data();
    result.candidate_ids = candidate_ids.data();
    result.row_flags = row_flags.data();

    uint64_t initial_execution_peak = 0;
    if (require_status(
            gafime_gpu_execution_memory_peak(matrix, &protocol, &initial_execution_peak),
            "execution_memory_peak")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    uint64_t repeated_execution_peak = 0;
    if (require_status(
            gafime_gpu_execution_memory_peak(matrix, &protocol, &repeated_execution_peak),
            "execution_memory_peak_repeat") ||
        initial_execution_peak != repeated_execution_peak ||
        initial_execution_peak <= desc.bytes) {
        std::fprintf(stderr, "execution-memory preflight was unstable or incomplete\n");
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

    if (result.row_count != 3 || result_combos[0] != 0 || result_combos[1] != 1 || result_combos[2] != 2) {
        std::fprintf(stderr, "unexpected result table shape or combos\n");
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
    if (require_close(result_metrics[4], 0.0f, "feature2 pearson")) {
        return 1;
    }
    if (require_close(result_metrics[5], 0.0f, "feature2 r2")) {
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
    stable_protocol.backend_kind = GAFIME_BACKEND_CUDA;
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
    status = gafime_gpu_execute(matrix, &stable_protocol, &stable_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute_stable")) {
        return 1;
    }
    if (require_close(stable_result_metrics[0], 1.0f, "high-offset pearson")) {
        return 1;
    }
    if (require_close(stable_result_metrics[1], 1.0f, "high-offset r2")) {
        return 1;
    }

    GafimeLaunchProtocol topk_protocol = protocol;
    topk_protocol.rank.top_k = 2;
    topk_protocol.rank.primary_metric = GAFIME_METRIC_R2;
    topk_protocol.rank.descending = 1;

    std::vector<uint32_t> topk_result_combos(2, UINT32_MAX);
    std::vector<float> topk_result_metrics(2 * 2, 0.0f);
    std::vector<uint32_t> topk_ranks(2, 0);
    std::vector<uint32_t> topk_families(2, 0);
    std::vector<uint64_t> topk_candidate_ids(2, 0);
    std::vector<uint32_t> topk_row_flags(2, 0);
    GafimeResultTable topk_result{};
    topk_result.abi_version = GAFIME_ABI_VERSION;
    topk_result.max_arity = 1;
    topk_result.metric_count = 2;
    topk_result.capacity = 2;
    topk_result.combo_indices = topk_result_combos.data();
    topk_result.metric_values = topk_result_metrics.data();
    topk_result.ranks = topk_ranks.data();
    topk_result.families = topk_families.data();
    topk_result.candidate_ids = topk_candidate_ids.data();
    topk_result.row_flags = topk_row_flags.data();

    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc_topk")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3), "matrix_upload_topk")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    uint64_t topk_execution_peak = 0;
    if (require_status(
            gafime_gpu_execution_memory_peak(matrix, &topk_protocol, &topk_execution_peak),
            "execution_memory_peak_topk") ||
        topk_execution_peak <= initial_execution_peak) {
        std::fprintf(stderr, "top-k preflight omitted ranking storage\n");
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    status = gafime_gpu_execute(matrix, &topk_protocol, &topk_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute_topk")) {
        return 1;
    }
    if (topk_result.row_count != 2 || topk_result_combos[0] != 0 || topk_result_combos[1] != 1 ||
        topk_candidate_ids[0] != 0 || topk_candidate_ids[1] != 1) {
        std::fprintf(stderr, "unexpected top-k result rows\n");
        return 1;
    }
    if (require_close(topk_result_metrics[0], 1.0f, "topk feature0 pearson")) {
        return 1;
    }
    if (require_close(topk_result_metrics[2], -1.0f, "topk feature1 pearson")) {
        return 1;
    }

    GafimeLaunchProtocol permutation_protocol = protocol;
    permutation_protocol.flags = GAFIME_LAUNCH_FLAG_GRAPH;
    permutation_protocol.permutations.permutation_count = 3;
    permutation_protocol.permutations.seed = 99;

    std::vector<uint32_t> permutation_result_combos(3, UINT32_MAX);
    std::vector<float> permutation_result_metrics(3 * 2, 0.0f);
    std::vector<uint32_t> permutation_ranks(3, 0);
    std::vector<uint32_t> permutation_families(3, 0);
    std::vector<uint64_t> permutation_candidate_ids(3, 0);
    std::vector<uint32_t> permutation_row_flags(3, 0);
    GafimeResultTable permutation_result{};
    permutation_result.abi_version = GAFIME_ABI_VERSION;
    permutation_result.max_arity = 1;
    permutation_result.metric_count = 2;
    permutation_result.capacity = 3;
    permutation_result.combo_indices = permutation_result_combos.data();
    permutation_result.metric_values = permutation_result_metrics.data();
    permutation_result.ranks = permutation_ranks.data();
    permutation_result.families = permutation_families.data();
    permutation_result.candidate_ids = permutation_candidate_ids.data();
    permutation_result.row_flags = permutation_row_flags.data();

    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc_permutation")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3), "matrix_upload_permutation")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    status = gafime_gpu_execute(matrix, &permutation_protocol, &permutation_result);
    if (require_status(status, "gpu_execute_permutation")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    if ((permutation_result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) == 0 ||
        permutation_result.row_count != 3 ||
        permutation_result_combos[0] != 0 ||
        permutation_result_combos[1] != 1 ||
        permutation_result_combos[2] != 2) {
        std::fprintf(stderr, "unexpected permutation graph result rows\n");
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    if (require_close(permutation_result_metrics[0], 1.0f, "permutation feature0 pearson")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    if (require_close(permutation_result_metrics[2], -1.0f, "permutation feature1 pearson")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    uint64_t one_row_permutation_peak = 0;
    uint64_t all_rows_permutation_peak = 0;
    uint64_t repeated_permutation_peak = 0;
    if (require_status(
            gafime_gpu_permutation_memory_peak(
                matrix,
                &permutation_protocol,
                1,
                &one_row_permutation_peak
            ),
            "permutation_memory_peak_one") ||
        require_status(
            gafime_gpu_permutation_memory_peak(
                matrix,
                &permutation_protocol,
                3,
                &all_rows_permutation_peak
            ),
            "permutation_memory_peak_all") ||
        require_status(
            gafime_gpu_permutation_memory_peak(
                matrix,
                &permutation_protocol,
                3,
                &repeated_permutation_peak
            ),
            "permutation_memory_peak_repeat") ||
        all_rows_permutation_peak <= one_row_permutation_peak ||
        repeated_permutation_peak != all_rows_permutation_peak) {
        std::fprintf(stderr, "permutation-memory preflight was unstable or omitted selected rows\n");
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    std::vector<float> permutation_pvalues(3 * 2, 0.0f);
    GafimePermutationSignificanceTable significance{};
    significance.abi_version = GAFIME_ABI_VERSION;
    significance.metric_count = 2;
    significance.row_count = 3;
    significance.candidate_ids = permutation_candidate_ids.data();
    significance.observed_metric_values = permutation_result_metrics.data();
    significance.p_values = permutation_pvalues.data();
    status = gafime_gpu_permutation_pvalues(
        matrix,
        &permutation_protocol,
        &significance
    );
    uint64_t resident_permutation_peak = 0;
    if (status == GAFIME_STATUS_OK) {
        status = gafime_gpu_permutation_memory_peak(
            matrix,
            &permutation_protocol,
            3,
            &resident_permutation_peak
        );
    }
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_permutation_pvalues")) {
        return 1;
    }
    if (resident_permutation_peak > all_rows_permutation_peak ||
        !std::all_of(
            permutation_pvalues.begin(),
            permutation_pvalues.end(),
            [](float value) { return std::isfinite(value) && value > 0.0f && value <= 1.0f; }
        )) {
        std::fprintf(stderr, "permutation p-values exceeded their preflight or were invalid\n");
        return 1;
    }

    std::vector<float> mi_features;
    std::vector<float> mi_target;
    mi_features.reserve(128 * 2);
    mi_target.reserve(128);
    for (int row = 0; row < 128; ++row) {
        const float x0 = static_cast<float>(row % 16) / 15.0f;
        const float x1 = static_cast<float>((row * 7) % 23) / 22.0f;
        mi_features.push_back(x0);
        mi_features.push_back(x1);
        mi_target.push_back(x0 > 0.5f ? 1.0f : 0.0f);
    }
    GafimeMatrixDesc mi_desc = desc;
    mi_desc.rows = 128;
    mi_desc.cols = 2;
    mi_desc.row_stride = 2;
    mi_desc.bytes = 128 * 2 * sizeof(float);
    const uint32_t mi_combos[] = {0, 1};
    const uint32_t mi_metric_ids[] = {GAFIME_METRIC_MUTUAL_INFO};
    GafimeArityChunk mi_chunk{};
    mi_chunk.arity = 1;
    mi_chunk.family = GAFIME_FAMILY_CONTINUOUS;
    mi_chunk.combo_count = 2;
    mi_chunk.descriptor_offset = 0;
    mi_chunk.descriptor_count = 2;
    GafimeLaunchProtocol mi_protocol{};
    mi_protocol.abi_version = GAFIME_ABI_VERSION;
    mi_protocol.backend_kind = GAFIME_BACKEND_CUDA;
    mi_protocol.max_arity = 1;
    mi_protocol.n_samples = 128;
    mi_protocol.n_features = 2;
    mi_protocol.family_count = 1;
    mi_protocol.combo_indices = {mi_combos, 2};
    mi_protocol.metric_ids = {mi_metric_ids, 1};
    mi_protocol.chunks = &mi_chunk;
    mi_protocol.chunk_count = 1;
    std::vector<uint32_t> mi_result_combos(2, UINT32_MAX);
    std::vector<float> mi_result_metrics(2, 0.0f);
    std::vector<uint32_t> mi_ranks(2, 0);
    std::vector<uint32_t> mi_families(2, 0);
    std::vector<uint64_t> mi_candidate_ids(2, 0);
    std::vector<uint32_t> mi_row_flags(2, 0);
    GafimeResultTable mi_result{};
    mi_result.abi_version = GAFIME_ABI_VERSION;
    mi_result.max_arity = 1;
    mi_result.metric_count = 1;
    mi_result.capacity = 2;
    mi_result.combo_indices = mi_result_combos.data();
    mi_result.metric_values = mi_result_metrics.data();
    mi_result.ranks = mi_ranks.data();
    mi_result.families = mi_families.data();
    mi_result.candidate_ids = mi_candidate_ids.data();
    mi_result.row_flags = mi_row_flags.data();
    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &mi_desc, &matrix), "matrix_alloc_mi")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, mi_features.data(), mi_target.data(), 128, 2), "matrix_upload_mi")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    status = gafime_gpu_execute(matrix, &mi_protocol, &mi_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute_mi")) {
        return 1;
    }
    if (!(std::isfinite(mi_result_metrics[0]) && std::isfinite(mi_result_metrics[1]) &&
          mi_result_metrics[0] >= 0.0f && mi_result_metrics[0] > mi_result_metrics[1])) {
        std::fprintf(stderr, "unexpected MI metrics: %f %f\n", mi_result_metrics[0], mi_result_metrics[1]);
        return 1;
    }

    const uint32_t mixed_combos[] = {
        0, 1, 2,
        0, 1,
        0, 2,
        1, 2,
    };
    GafimeArityChunk mixed_chunks[2]{};
    mixed_chunks[0].arity = 1;
    mixed_chunks[0].family = GAFIME_FAMILY_CONTINUOUS;
    mixed_chunks[0].combo_count = 3;
    mixed_chunks[0].descriptor_offset = 0;
    mixed_chunks[0].descriptor_count = 3;
    mixed_chunks[1].arity = 2;
    mixed_chunks[1].family = GAFIME_FAMILY_CONTINUOUS;
    mixed_chunks[1].combo_count = 3;
    mixed_chunks[1].combo_row_offset = 3;
    mixed_chunks[1].descriptor_offset = 3;
    mixed_chunks[1].descriptor_count = 3;
    mixed_chunks[1].local_chunk_id = 1;

    GafimeLaunchProtocol mixed_protocol{};
    mixed_protocol.abi_version = GAFIME_ABI_VERSION;
    mixed_protocol.backend_kind = GAFIME_BACKEND_CUDA;
    mixed_protocol.max_arity = 2;
    mixed_protocol.n_samples = 4;
    mixed_protocol.n_features = 3;
    mixed_protocol.family_count = 1;
    mixed_protocol.combo_indices = {mixed_combos, 9};
    mixed_protocol.metric_ids = {metric_ids, 2};
    mixed_protocol.chunks = mixed_chunks;
    mixed_protocol.chunk_count = 2;

    std::vector<uint32_t> mixed_result_combos(6 * 2, UINT32_MAX);
    std::vector<float> mixed_result_metrics(6 * 2, 0.0f);
    std::vector<uint32_t> mixed_ranks(6, 0);
    std::vector<uint32_t> mixed_families(6, 0);
    std::vector<uint64_t> mixed_candidate_ids(6, 0);
    std::vector<uint32_t> mixed_row_flags(6, 0);
    GafimeResultTable mixed_result{};
    mixed_result.abi_version = GAFIME_ABI_VERSION;
    mixed_result.max_arity = 2;
    mixed_result.metric_count = 2;
    mixed_result.capacity = 6;
    mixed_result.combo_indices = mixed_result_combos.data();
    mixed_result.metric_values = mixed_result_metrics.data();
    mixed_result.ranks = mixed_ranks.data();
    mixed_result.families = mixed_families.data();
    mixed_result.candidate_ids = mixed_candidate_ids.data();
    mixed_result.row_flags = mixed_row_flags.data();

    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc_mixed")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3), "matrix_upload_mixed")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    status = gafime_gpu_execute(matrix, &mixed_protocol, &mixed_result);
    if (require_status(status, "gpu_execute_mixed_first")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    mixed_protocol.flags = GAFIME_LAUNCH_FLAG_GRAPH;
    mixed_result.flags = 0;
    status = gafime_gpu_execute(matrix, &mixed_protocol, &mixed_result);
    if (require_status(status, "gpu_execute_mixed_graph_first")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    status = gafime_gpu_execute(matrix, &mixed_protocol, &mixed_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute_mixed_graph_second")) {
        return 1;
    }
    if ((mixed_result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) == 0) {
        std::fprintf(stderr, "mixed graph execution did not set graph replay flag\n");
        return 1;
    }
    if (mixed_result.row_count != 6) {
        std::fprintf(stderr, "unexpected mixed result row count: %llu\n",
                     static_cast<unsigned long long>(mixed_result.row_count));
        return 1;
    }
    const uint32_t expected_mixed_combos[] = {
        0, UINT32_MAX,
        1, UINT32_MAX,
        2, UINT32_MAX,
        0, 1,
        0, 2,
        1, 2,
    };
    for (size_t idx = 0; idx < mixed_result_combos.size(); ++idx) {
        if (mixed_result_combos[idx] != expected_mixed_combos[idx]) {
            std::fprintf(stderr, "mixed combo mismatch at %zu: actual=%u expected=%u\n",
                         idx, mixed_result_combos[idx], expected_mixed_combos[idx]);
            return 1;
        }
    }
    if (require_close(mixed_result_metrics[6], 0.0f, "pair01 pearson")) {
        return 1;
    }
    if (require_close(mixed_result_metrics[7], 0.0f, "pair01 r2")) {
        return 1;
    }

    matrix = nullptr;
    if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc_decision_path")) {
        return 1;
    }
    if (require_status(gafime_gpu_matrix_upload(matrix, features, target, 4, 3), "matrix_upload_decision_path")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const GafimeDecisionPathTerm path_terms[] = {
        {0, GAFIME_DECISION_PATH_SIGN_LE, 2.0f, 0, {0, 0}},
        {0, GAFIME_DECISION_PATH_SIGN_GT, 2.0f, 0, {0, 0}},
        {1, GAFIME_DECISION_PATH_SIGN_GT, 2.0f, 0, {0, 0}},
    };
    const uint32_t path_offsets[] = {0, 1, 3};
    std::vector<float> membership(2 * 4, -1.0f);
    GafimeDecisionPathBatch path_batch{};
    path_batch.abi_version = GAFIME_ABI_VERSION;
    path_batch.path_count = 2;
    path_batch.term_count = 3;
    if (std::getenv("GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP") != nullptr) {
        path_batch.flags = GAFIME_DECISION_PATH_FLAG_REQUIRE_RT;
    }
    path_batch.terms = path_terms;
    path_batch.path_offsets = path_offsets;
    path_batch.membership_host = membership.data();
    status = gafime_gpu_decision_path_membership(matrix, &path_batch);
    if (require_status(status, "gpu_decision_path_membership")) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const float expected_membership[] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    for (size_t idx = 0; idx < membership.size(); ++idx) {
        if (require_close(membership[idx], expected_membership[idx], "decision_path membership")) {
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
    }
    const uint32_t path_score_metrics[] = {GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};
    std::vector<uint32_t> path_score_combos(2, UINT32_MAX);
    std::vector<float> path_score_values(2 * 2, 0.0f);
    std::vector<uint32_t> path_score_ranks(2, 0);
    std::vector<uint32_t> path_score_families(2, 0);
    std::vector<uint64_t> path_score_ids(2, 0);
    std::vector<uint32_t> path_score_flags(2, 0);
    GafimeResultTable path_score_result{};
    path_score_result.abi_version = GAFIME_ABI_VERSION;
    path_score_result.max_arity = 1;
    path_score_result.metric_count = 2;
    path_score_result.capacity = 2;
    path_score_result.combo_indices = path_score_combos.data();
    path_score_result.metric_values = path_score_values.data();
    path_score_result.ranks = path_score_ranks.data();
    path_score_result.families = path_score_families.data();
    path_score_result.candidate_ids = path_score_ids.data();
    path_score_result.row_flags = path_score_flags.data();
    GafimeDecisionPathScoreBatch path_score_batch{};
    path_score_batch.abi_version = GAFIME_ABI_VERSION;
    path_score_batch.path_count = 2;
    path_score_batch.term_count = 3;
    path_score_batch.terms = path_terms;
    path_score_batch.path_offsets = path_offsets;
    path_score_batch.metric_ids = path_score_metrics;
    path_score_batch.metric_count = 2;

    int disabled_firsthit_status = GAFIME_STATUS_DEVICE_ERROR;
    {
        ScopedEnvironmentOverride rt_mode("GAFIME_CUDA_DECISION_PATH_RT");
        ScopedEnvironmentOverride score_mode("GAFIME_CUDA_DECISION_PATH_RT_SCORE");
        if (!rt_mode.set("off") || !score_mode.set("firsthit")) {
            std::fprintf(stderr, "failed to configure disabled firsthit regression\n");
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
        disabled_firsthit_status =
            gafime_gpu_decision_path_score(matrix, &path_score_batch, &path_score_result);
    }
    if (disabled_firsthit_status != GAFIME_STATUS_UNSUPPORTED_BACKEND) {
        std::fprintf(
            stderr,
            "disabled firsthit returned status %d instead of %d\n",
            disabled_firsthit_status,
            GAFIME_STATUS_UNSUPPORTED_BACKEND
        );
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    status = gafime_gpu_decision_path_score(matrix, &path_score_batch, &path_score_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_decision_path_score")) {
        return 1;
    }
    if (path_score_result.row_count != 2 ||
        path_score_combos[0] != 0 ||
        path_score_combos[1] != 1 ||
        path_score_families[0] != GAFIME_FAMILY_DECISION_PATH ||
        path_score_families[1] != GAFIME_FAMILY_DECISION_PATH) {
        std::fprintf(stderr, "unexpected decision_path score metadata\n");
        return 1;
    }
    if (require_close(path_score_values[0], -0.8944272f, "decision_path score0 pearson") ||
        require_close(path_score_values[1], 0.8f, "decision_path score0 r2") ||
        require_close(path_score_values[2], 0.2581989f, "decision_path score1 pearson") ||
        require_close(path_score_values[3], 0.0666667f, "decision_path score1 r2")) {
        return 1;
    }
    if (require_status(
            decision_path_state_cleanup.release(),
            "gpu_decision_path_release_device_state"
        )) {
        return 1;
    }
    return 0;
}
