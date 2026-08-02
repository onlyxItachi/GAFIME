#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

#include "precision_kernels.cuh"

namespace {

constexpr uint32_t kAbi10 = GAFIME_ABI_VERSION;
constexpr uint32_t kAbi11 = GAFIME_PRECISION_ABI_VERSION;

bool check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "precision ABI smoke: %s\n", message);
    }
    return condition;
}

struct ProtocolFixture {
    std::vector<uint32_t> combos{0, 1};
    std::vector<uint32_t> metrics{
        GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_SPEARMAN,
        GAFIME_METRIC_MUTUAL_INFO,
        GAFIME_METRIC_R2,
    };
    GafimeArityChunk chunk{};
    GafimeLaunchProtocol base{};
    GafimePrecisionLaunchProtocol precision{};

    ProtocolFixture(uint64_t rows, uint32_t cols) {
        chunk.arity = 1;
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.combo_count = 2;
        chunk.descriptor_count = 2;
        base.abi_version = kAbi10;
        base.backend_kind = GAFIME_BACKEND_CUDA;
        base.max_arity = 1;
        base.n_samples = rows;
        base.n_features = cols;
        base.family_count = 1;
        base.combo_indices = {combos.data(), combos.size()};
        base.metric_ids = {metrics.data(), metrics.size()};
        base.chunks = &chunk;
        base.chunk_count = 1;
        precision.abi_version = kAbi11;
        precision.base = &base;
    }
};

struct FamilyCoverageFixture {
    std::vector<uint32_t> combos;
    std::vector<uint32_t> metrics{
        GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_SPEARMAN,
        GAFIME_METRIC_MUTUAL_INFO,
        GAFIME_METRIC_R2,
    };
    std::vector<GafimeArityChunk> chunks;
    GafimeShapeHint shape_hint{};
    GafimeLaunchProtocol base{};
    GafimePrecisionLaunchProtocol precision{};

    FamilyCoverageFixture(uint64_t rows, uint32_t cols) {
        constexpr uint32_t families[] = {
            GAFIME_FAMILY_CONTINUOUS,
            GAFIME_FAMILY_TIME_SERIES,
            GAFIME_FAMILY_DECISION_PATH,
            GAFIME_FAMILY_CONTINUOUS,
            GAFIME_FAMILY_TIME_SERIES,
        };
        uint64_t descriptor_offset = 0;
        for (uint32_t arity = 1; arity <= 5; ++arity) {
            GafimeArityChunk chunk{};
            chunk.arity = arity;
            chunk.family = families[arity - 1];
            chunk.shape_hint_index = 0;
            chunk.combo_row_offset = arity - 1;
            chunk.combo_count = 1;
            chunk.local_chunk_id = arity - 1;
            chunk.descriptor_offset = descriptor_offset;
            chunk.descriptor_count = 1;
            for (uint32_t feature = 0; feature < arity; ++feature) {
                combos.push_back(feature);
            }
            descriptor_offset += arity;
            chunks.push_back(chunk);
        }
        shape_hint.vendor_hint = 16;
        base.abi_version = kAbi10;
        base.backend_kind = GAFIME_BACKEND_CUDA;
        base.max_arity = 5;
        base.n_samples = rows;
        base.n_features = cols;
        base.family_count = 3;
        base.combo_indices = {combos.data(), combos.size()};
        base.metric_ids = {metrics.data(), metrics.size()};
        base.chunks = chunks.data();
        base.chunk_count = static_cast<uint32_t>(chunks.size());
        base.shape_hints = &shape_hint;
        base.shape_hint_count = 1;
        precision.abi_version = kAbi11;
        precision.base = &base;
    }
};

template <typename Result, typename ResultTable>
bool result_is_finite(const ResultTable& table) {
    for (uint64_t index = 0; index < table.row_count * table.metric_count; ++index) {
        if (!std::isfinite(static_cast<Result>(table.metric_values[index]))) return false;
    }
    return true;
}

template <typename ResultTable>
void init_result(
    ResultTable* result,
    uint32_t abi,
    uint32_t* combos,
    typename std::conditional_t<std::is_same_v<ResultTable, GafimeResultTable>, float, double>* values,
    uint32_t* ranks,
    uint32_t* families,
    uint64_t* ids,
    uint32_t* flags
) {
    result->abi_version = abi;
    result->max_arity = 1;
    result->metric_count = 4;
    result->capacity = 2;
    result->combo_indices = combos;
    result->metric_values = values;
    result->ranks = ranks;
    result->families = families;
    result->candidate_ids = ids;
    result->row_flags = flags;
}

bool precision_diagnostics_are_typed(GafimeGpuMatrix matrix) {
    // One unary row and one binary row prove that the shared ABI-1.0 batch
    // layout is dispatched through the ABI-1.1 matrix specialization instead
    // of being reinterpreted as the private legacy matrix layout.
    uint32_t combos[4] = {0, UINT32_MAX, 0, 1};
    uint64_t overflow_rows[2] = {UINT64_MAX, UINT64_MAX};
    uint32_t flags[2] = {UINT32_MAX, UINT32_MAX};
    GafimeInteractionDiagnosticBatch diagnostics{};
    diagnostics.abi_version = kAbi10;
    diagnostics.max_arity = 2;
    diagnostics.row_count = 2;
    diagnostics.combo_indices = combos;
    diagnostics.combo_index_count = 4;
    diagnostics.overflow_row_counts = overflow_rows;
    diagnostics.flags = flags;
    return check(
               gafime_gpu_interaction_diagnostics(matrix, &diagnostics) == GAFIME_STATUS_OK,
               "precision interaction diagnostics dispatch") &&
        check(
            overflow_rows[0] == 0 && overflow_rows[1] == 0 && flags[0] == 0 && flags[1] == 0,
            "precision interaction diagnostics values");
}

// Independent host oracle for the unary Pearson score.  It uses extended
// precision only to assess the CUDA lane; it is not shared with the payload
// implementation or its reduction order.
long double unary_pearson_oracle_f32_storage(
    const float* features,
    const float* target,
    uint64_t rows,
    uint32_t cols,
    uint32_t feature
) {
    long double sum_x = 0;
    long double sum_y = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        sum_x += static_cast<long double>(features[row * cols + feature]);
        sum_y += static_cast<long double>(target[row]);
    }
    const long double mean_x = sum_x / static_cast<long double>(rows);
    const long double mean_y = sum_y / static_cast<long double>(rows);
    long double sxx = 0;
    long double syy = 0;
    long double sxy = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        const long double dx = static_cast<long double>(features[row * cols + feature]) - mean_x;
        const long double dy = static_cast<long double>(target[row]) - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    return sxx == 0 || syy == 0 ? 0 : sxy / std::sqrt(sxx * syy);
}

bool run_fp32() {
    constexpr uint64_t rows = 6;
    constexpr uint32_t cols = 2;
    const float features[rows * cols] = {
        1, 1, 2, 2, 3, 4, 4, 2, 5, 3, 6, 1,
    };
    const float target[rows] = {1, 2, 3, 2, 5, 4};
    GafimePrecisionMatrixDesc desc{};
    desc.abi_version = kAbi11;
    desc.profile = GAFIME_PRECISION_FP32;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = rows;
    desc.cols = cols;
    desc.row_stride = cols;
    desc.bytes = sizeof(features);
    GafimeGpuMatrix matrix = nullptr;
    if (!check(gafime_gpu_matrix_alloc_v2(0, &desc, &matrix) == GAFIME_STATUS_OK, "fp32 alloc") ||
        !check(gafime_gpu_matrix_upload_f32_v2(matrix, features, target, rows, cols) == GAFIME_STATUS_OK,
            "fp32 upload") ||
        !check(gafime_gpu_matrix_upload_f64_v2(matrix, nullptr, nullptr, rows, cols) ==
                GAFIME_STATUS_UNSUPPORTED_BACKEND,
            "fp32 rejects f64 upload")) {
        gafime_gpu_matrix_free(matrix);
        return false;
    }
    ProtocolFixture fixture(rows, cols);
    fixture.precision.profile = GAFIME_PRECISION_FP32;
    fixture.base.flags = GAFIME_LAUNCH_FLAG_GRAPH;
    uint32_t combo_out[2]{};
    float value_out[8]{};
    uint32_t ranks[2]{};
    uint32_t families[2]{};
    uint64_t ids[2]{};
    uint32_t flags[2]{};
    GafimeResultTable result{};
    init_result(&result, kAbi10, combo_out, value_out, ranks, families, ids, flags);
    bool ok = precision_diagnostics_are_typed(matrix) &&
        check(gafime_gpu_execute_f32_v2(matrix, &fixture.precision, &result) == GAFIME_STATUS_OK,
        "fp32 graph execute") &&
        check(result.row_count == 2 && result_is_finite<float>(result), "fp32 all metrics") &&
        check(std::fabs(static_cast<double>(value_out[0]) - static_cast<double>(
                    unary_pearson_oracle_f32_storage(features, target, rows, cols, 0))) < 2e-5,
            "fp32 unary Pearson remains within honest fp32 tolerance") &&
        check((result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0, "fp32 graph replay");
    fixture.base.flags = 0;
    fixture.base.rank.top_k = 1;
    fixture.base.rank.primary_metric = GAFIME_METRIC_PEARSON;
    fixture.base.rank.descending = 1;
    result.flags = 0;
    ok = ok && check(gafime_gpu_execute_f32_v2(matrix, &fixture.precision, &result) == GAFIME_STATUS_OK,
        "fp32 ranking") && check(result.row_count == 1, "fp32 top-k result");
    fixture.base.rank = {};
    fixture.base.permutations.permutation_count = 2;
    uint64_t candidate_ids[1] = {0};
    float observed[4] = {value_out[0], value_out[1], value_out[2], value_out[3]};
    float p_values[4]{};
    GafimePermutationSignificanceTable significance{};
    significance.abi_version = kAbi10;
    significance.metric_count = 4;
    significance.row_count = 1;
    significance.candidate_ids = candidate_ids;
    significance.observed_metric_values = observed;
    significance.p_values = p_values;
    ok = ok && check(gafime_gpu_permutation_pvalues_f32_v2(matrix, &fixture.precision, &significance) ==
        GAFIME_STATUS_OK, "fp32 permutation") &&
        check(p_values[0] >= 0.0f && p_values[0] <= 1.0f, "fp32 p-value");
    gafime_gpu_matrix_free(matrix);
    return ok;
}

bool run_mixed() {
    constexpr uint64_t rows = 6;
    constexpr uint32_t cols = 2;
    const float features[rows * cols] = {
        1, 1, 2, 2, 3, 4, 4, 2, 5, 3, 6, 1,
    };
    const float target[rows] = {1, 2, 3, 2, 5, 4};
    GafimePrecisionMatrixDesc desc{};
    desc.abi_version = kAbi11;
    desc.profile = GAFIME_PRECISION_MIXED;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = rows;
    desc.cols = cols;
    desc.row_stride = cols;
    desc.bytes = sizeof(features);
    GafimeGpuMatrix matrix = nullptr;
    if (!check(gafime_gpu_matrix_alloc_v2(0, &desc, &matrix) == GAFIME_STATUS_OK, "mixed alloc") ||
        !check(gafime_gpu_matrix_upload_f32_v2(matrix, features, target, rows, cols) == GAFIME_STATUS_OK,
            "mixed upload") ||
        !check(gafime_gpu_matrix_upload_f64_v2(matrix, nullptr, nullptr, rows, cols) ==
                GAFIME_STATUS_UNSUPPORTED_BACKEND,
            "mixed rejects f64 upload")) {
        gafime_gpu_matrix_free(matrix);
        return false;
    }
    ProtocolFixture fixture(rows, cols);
    fixture.precision.profile = GAFIME_PRECISION_MIXED;
    uint32_t combo_out[2]{};
    double value_out[8]{};
    uint32_t ranks[2]{};
    uint32_t families[2]{};
    uint64_t ids[2]{};
    uint32_t flags[2]{};
    GafimeResultTableF64 result{};
    init_result(&result, kAbi11, combo_out, value_out, ranks, families, ids, flags);
    bool ok = precision_diagnostics_are_typed(matrix) &&
        check(gafime_gpu_execute_f64_v2(matrix, &fixture.precision, &result) == GAFIME_STATUS_OK,
        "mixed execute") && check(result.row_count == 2 && result_is_finite<double>(result),
        "mixed f64 public result") &&
        check(std::fabs(value_out[0] - static_cast<double>(
                    unary_pearson_oracle_f32_storage(features, target, rows, cols, 0))) < 1e-10,
            "mixed reductions match independent high-precision oracle") &&
        check(value_out[0] != static_cast<double>(static_cast<float>(value_out[0])),
            "mixed public score is not silently rounded back to fp32");
    fixture.base.permutations.permutation_count = 2;
    uint64_t candidate_ids[1] = {0};
    double observed[4] = {value_out[0], value_out[1], value_out[2], value_out[3]};
    double p_values[4]{};
    GafimePermutationSignificanceTableF64 significance{};
    significance.abi_version = kAbi11;
    significance.metric_count = 4;
    significance.row_count = 1;
    significance.candidate_ids = candidate_ids;
    significance.observed_metric_values = observed;
    significance.p_values = p_values;
    ok = ok && check(gafime_gpu_permutation_pvalues_f64_v2(matrix, &fixture.precision, &significance) ==
        GAFIME_STATUS_OK, "mixed permutation") &&
        check(p_values[0] >= 0.0 && p_values[0] <= 1.0, "mixed p-value");
    gafime_gpu_matrix_free(matrix);
    return ok;
}

bool run_fp64() {
    constexpr uint64_t rows = 6;
    constexpr uint32_t cols = 2;
    const double epsilon = std::ldexp(1.0, -30);
    const double features[rows * cols] = {
        1 + epsilon, 1, 2 + epsilon, 2, 3 + epsilon, 4,
        4 + epsilon, 2, 5 + epsilon, 3, 6 + epsilon, 1,
    };
    const double target[rows] = {1 + epsilon, 2 + epsilon, 3 + epsilon, 2 + epsilon, 5 + epsilon, 4 + epsilon};
    GafimePrecisionMatrixDesc desc{};
    desc.abi_version = kAbi11;
    desc.profile = GAFIME_PRECISION_FP64;
    desc.dtype = GAFIME_DTYPE_F64;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = rows;
    desc.cols = cols;
    desc.row_stride = cols;
    desc.bytes = sizeof(features);
    GafimeGpuMatrix matrix = nullptr;
    if (!check(gafime_gpu_matrix_alloc_v2(0, &desc, &matrix) == GAFIME_STATUS_OK, "fp64 alloc") ||
        !check(gafime_gpu_matrix_upload_f64_v2(matrix, features, target, rows, cols) == GAFIME_STATUS_OK,
            "fp64 upload") ||
        !check(gafime_gpu_matrix_upload_f32_v2(matrix, nullptr, nullptr, rows, cols) ==
                GAFIME_STATUS_UNSUPPORTED_BACKEND,
            "fp64 rejects f32 upload")) {
        gafime_gpu_matrix_free(matrix);
        return false;
    }
    ProtocolFixture fixture(rows, cols);
    fixture.precision.profile = GAFIME_PRECISION_FP64;
    uint32_t combo_out[2]{};
    double value_out[8]{};
    uint32_t ranks[2]{};
    uint32_t families[2]{};
    uint64_t ids[2]{};
    uint32_t flags[2]{};
    GafimeResultTableF64 result{};
    init_result(&result, kAbi11, combo_out, value_out, ranks, families, ids, flags);
    bool ok = precision_diagnostics_are_typed(matrix) &&
        check(gafime_gpu_execute_f64_v2(matrix, &fixture.precision, &result) == GAFIME_STATUS_OK,
        "fp64 execute") && check(result.row_count == 2 && result_is_finite<double>(result),
        "fp64 all metrics");
    gafime_gpu_matrix_free(matrix);
    return ok;
}

bool run_diagnostic_storage_width_contract() {
    constexpr uint64_t rows = 2;
    constexpr uint32_t cols = 2;
    const float features_f32[rows * cols] = {1.0e30f, 1.0e30f, -1.0e30f, -1.0e30f};
    const float target_f32[rows] = {1.0f, -1.0f};
    const double features_f64[rows * cols] = {1.0e30, 1.0e30, -1.0e30, -1.0e30};
    const double target_f64[rows] = {1.0, -1.0};

    const auto make_desc = [](GafimePrecisionProfile profile, uint32_t dtype, uint64_t bytes) {
        GafimePrecisionMatrixDesc desc{};
        desc.abi_version = kAbi11;
        desc.profile = profile;
        desc.dtype = dtype;
        desc.layout = GAFIME_MATRIX_ROW_MAJOR;
        desc.rows = rows;
        desc.cols = cols;
        desc.row_stride = cols;
        desc.bytes = bytes;
        return desc;
    };
    const auto fp32_desc = make_desc(GAFIME_PRECISION_FP32, GAFIME_DTYPE_F32, sizeof(features_f32));
    const auto mixed_desc = make_desc(GAFIME_PRECISION_MIXED, GAFIME_DTYPE_F32, sizeof(features_f32));
    const auto fp64_desc = make_desc(GAFIME_PRECISION_FP64, GAFIME_DTYPE_F64, sizeof(features_f64));
    GafimeGpuMatrix fp32 = nullptr;
    GafimeGpuMatrix mixed = nullptr;
    GafimeGpuMatrix fp64 = nullptr;
    const auto cleanup = [&] {
        gafime_gpu_matrix_free(fp32);
        gafime_gpu_matrix_free(mixed);
        gafime_gpu_matrix_free(fp64);
    };
    bool ok = check(gafime_gpu_matrix_alloc_v2(0, &fp32_desc, &fp32) == GAFIME_STATUS_OK,
            "diagnostics fp32 alloc") &&
        check(gafime_gpu_matrix_alloc_v2(0, &mixed_desc, &mixed) == GAFIME_STATUS_OK,
            "diagnostics mixed alloc") &&
        check(gafime_gpu_matrix_alloc_v2(0, &fp64_desc, &fp64) == GAFIME_STATUS_OK,
            "diagnostics fp64 alloc") &&
        check(gafime_gpu_matrix_upload_f32_v2(fp32, features_f32, target_f32, rows, cols) ==
                GAFIME_STATUS_OK, "diagnostics fp32 upload") &&
        check(gafime_gpu_matrix_upload_f32_v2(mixed, features_f32, target_f32, rows, cols) ==
                GAFIME_STATUS_OK, "diagnostics mixed upload") &&
        check(gafime_gpu_matrix_upload_f64_v2(fp64, features_f64, target_f64, rows, cols) ==
                GAFIME_STATUS_OK, "diagnostics fp64 upload");
    const auto overflow_rows = [&](GafimeGpuMatrix matrix, uint64_t* value_out) {
        uint32_t combo[2] = {0, 1};
        uint32_t flags = UINT32_MAX;
        GafimeInteractionDiagnosticBatch diagnostics{};
        diagnostics.abi_version = kAbi10;
        diagnostics.max_arity = 2;
        diagnostics.row_count = 1;
        diagnostics.combo_indices = combo;
        diagnostics.combo_index_count = 2;
        diagnostics.overflow_row_counts = value_out;
        diagnostics.flags = &flags;
        return gafime_gpu_interaction_diagnostics(matrix, &diagnostics) == GAFIME_STATUS_OK &&
            flags == 0;
    };
    uint64_t fp32_overflow = UINT64_MAX;
    uint64_t mixed_overflow = UINT64_MAX;
    uint64_t fp64_overflow = UINT64_MAX;
    ok = ok && check(overflow_rows(fp32, &fp32_overflow), "diagnostics fp32 execution") &&
        check(overflow_rows(mixed, &mixed_overflow), "diagnostics mixed execution") &&
        check(overflow_rows(fp64, &fp64_overflow), "diagnostics fp64 execution") &&
        check(fp32_overflow == rows && mixed_overflow == rows,
            "fp32 and mixed diagnose fp32 pointwise overflow") &&
        check(fp64_overflow == 0, "fp64 diagnostics preserve finite fp64 materialization");
    cleanup();
    return ok;
}

bool run_fp64_distinguishing_values() {
    constexpr uint64_t rows = 4;
    constexpr uint32_t cols = 1;
    const double epsilon = std::ldexp(1.0, -30);
    const double fp64_values[rows] = {1.0, 1.0 + epsilon, 1.0 + 2 * epsilon, 1.0 + 3 * epsilon};
    const float fp32_values[rows] = {
        static_cast<float>(fp64_values[0]), static_cast<float>(fp64_values[1]),
        static_cast<float>(fp64_values[2]), static_cast<float>(fp64_values[3]),
    };
    if (!check(fp32_values[0] == fp32_values[1] && fp32_values[1] == fp32_values[2] &&
            fp32_values[2] == fp32_values[3], "adversarial values collapse in fp32")) return false;
    uint32_t combo = 0;
    uint32_t metric = GAFIME_METRIC_PEARSON;
    GafimeArityChunk chunk{};
    chunk.arity = 1;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = 1;
    chunk.descriptor_count = 1;
    GafimeLaunchProtocol base{};
    base.abi_version = kAbi10;
    base.backend_kind = GAFIME_BACKEND_CUDA;
    base.max_arity = 1;
    base.n_samples = rows;
    base.n_features = cols;
    base.family_count = 1;
    base.combo_indices = {&combo, 1};
    base.metric_ids = {&metric, 1};
    base.chunks = &chunk;
    base.chunk_count = 1;

    GafimePrecisionMatrixDesc fp64_desc{};
    fp64_desc.abi_version = kAbi11;
    fp64_desc.profile = GAFIME_PRECISION_FP64;
    fp64_desc.dtype = GAFIME_DTYPE_F64;
    fp64_desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    fp64_desc.rows = rows;
    fp64_desc.cols = cols;
    fp64_desc.row_stride = cols;
    fp64_desc.bytes = sizeof(fp64_values);
    GafimeGpuMatrix fp64_matrix = nullptr;
    if (!check(gafime_gpu_matrix_alloc_v2(0, &fp64_desc, &fp64_matrix) == GAFIME_STATUS_OK,
            "fp64 distinguishing alloc") ||
        !check(gafime_gpu_matrix_upload_f64_v2(fp64_matrix, fp64_values, fp64_values, rows, cols) ==
                GAFIME_STATUS_OK, "fp64 distinguishing upload")) {
        gafime_gpu_matrix_free(fp64_matrix);
        return false;
    }
    GafimePrecisionLaunchProtocol fp64_protocol{};
    fp64_protocol.abi_version = kAbi11;
    fp64_protocol.profile = GAFIME_PRECISION_FP64;
    fp64_protocol.base = &base;
    uint32_t combo64 = 0;
    double values64 = 0;
    uint32_t rank64 = 0;
    uint32_t family64 = 0;
    uint64_t id64 = 0;
    uint32_t flag64 = 0;
    GafimeResultTableF64 result64{};
    result64.abi_version = kAbi11;
    result64.max_arity = 1;
    result64.metric_count = 1;
    result64.capacity = 1;
    result64.combo_indices = &combo64;
    result64.metric_values = &values64;
    result64.ranks = &rank64;
    result64.families = &family64;
    result64.candidate_ids = &id64;
    result64.row_flags = &flag64;
    bool ok = check(gafime_gpu_execute_f64_v2(fp64_matrix, &fp64_protocol, &result64) == GAFIME_STATUS_OK,
        "fp64 distinguishing execute") &&
        check(values64 > 0.999999, "fp64 preserves sub-fp32 distinctions");
    gafime_gpu_matrix_free(fp64_matrix);

    GafimePrecisionMatrixDesc fp32_desc{};
    fp32_desc.abi_version = kAbi11;
    fp32_desc.profile = GAFIME_PRECISION_FP32;
    fp32_desc.dtype = GAFIME_DTYPE_F32;
    fp32_desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    fp32_desc.rows = rows;
    fp32_desc.cols = cols;
    fp32_desc.row_stride = cols;
    fp32_desc.bytes = sizeof(fp32_values);
    GafimeGpuMatrix fp32_matrix = nullptr;
    if (!check(gafime_gpu_matrix_alloc_v2(0, &fp32_desc, &fp32_matrix) == GAFIME_STATUS_OK,
            "fp32 collapse alloc") ||
        !check(gafime_gpu_matrix_upload_f32_v2(fp32_matrix, fp32_values, fp32_values, rows, cols) ==
                GAFIME_STATUS_OK, "fp32 collapse upload")) {
        gafime_gpu_matrix_free(fp32_matrix);
        return false;
    }
    GafimePrecisionLaunchProtocol fp32_protocol{};
    fp32_protocol.abi_version = kAbi11;
    fp32_protocol.profile = GAFIME_PRECISION_FP32;
    fp32_protocol.base = &base;
    uint32_t combo32 = 0;
    float values32 = 0;
    uint32_t rank32 = 0;
    uint32_t family32 = 0;
    uint64_t id32 = 0;
    uint32_t flag32 = 0;
    GafimeResultTable result32{};
    result32.abi_version = kAbi10;
    result32.max_arity = 1;
    result32.metric_count = 1;
    result32.capacity = 1;
    result32.combo_indices = &combo32;
    result32.metric_values = &values32;
    result32.ranks = &rank32;
    result32.families = &family32;
    result32.candidate_ids = &id32;
    result32.row_flags = &flag32;
    ok = ok && check(gafime_gpu_execute_f32_v2(fp32_matrix, &fp32_protocol, &result32) == GAFIME_STATUS_OK,
        "fp32 collapse execute") && check(values32 == 0.0f, "fp32 honestly reports zero variance");
    gafime_gpu_matrix_free(fp32_matrix);
    return ok;
}

bool run_profile_identity_separation() {
    // All three matrices deliberately receive the same logical inputs and the
    // exact same immutable descriptor generation.  Their profile components
    // must still keep resident statistics, descriptor buffers, and graph
    // replay state distinct.
    constexpr uint64_t rows = 6;
    constexpr uint32_t cols = 2;
    const float features_f32[rows * cols] = {
        1, 1, 2, 2, 3, 4, 4, 2, 5, 3, 6, 1,
    };
    const float target_f32[rows] = {1, 2, 3, 2, 5, 4};
    std::vector<double> features_f64(rows * cols);
    std::vector<double> target_f64(rows);
    for (uint64_t index = 0; index < rows * cols; ++index) {
        features_f64[index] = static_cast<double>(features_f32[index]);
    }
    for (uint64_t index = 0; index < rows; ++index) {
        target_f64[index] = static_cast<double>(target_f32[index]);
    }

    const auto make_desc = [](GafimePrecisionProfile profile, uint32_t dtype, uint64_t bytes) {
        GafimePrecisionMatrixDesc desc{};
        desc.abi_version = kAbi11;
        desc.profile = profile;
        desc.dtype = dtype;
        desc.layout = GAFIME_MATRIX_ROW_MAJOR;
        desc.rows = rows;
        desc.cols = cols;
        desc.row_stride = cols;
        desc.bytes = bytes;
        return desc;
    };
    const GafimePrecisionMatrixDesc fp32_desc = make_desc(
        GAFIME_PRECISION_FP32, GAFIME_DTYPE_F32, sizeof(features_f32));
    const GafimePrecisionMatrixDesc mixed_desc = make_desc(
        GAFIME_PRECISION_MIXED, GAFIME_DTYPE_F32, sizeof(features_f32));
    const GafimePrecisionMatrixDesc fp64_desc = make_desc(
        GAFIME_PRECISION_FP64, GAFIME_DTYPE_F64,
        static_cast<uint64_t>(features_f64.size() * sizeof(double)));

    GafimeGpuMatrix fp32 = nullptr;
    GafimeGpuMatrix mixed = nullptr;
    GafimeGpuMatrix fp64 = nullptr;
    const auto cleanup = [&] {
        gafime_gpu_matrix_free(fp32);
        gafime_gpu_matrix_free(mixed);
        gafime_gpu_matrix_free(fp64);
    };
    bool ok = check(gafime_gpu_matrix_alloc_v2(0, &fp32_desc, &fp32) == GAFIME_STATUS_OK,
            "identity fp32 alloc") &&
        check(gafime_gpu_matrix_alloc_v2(0, &mixed_desc, &mixed) == GAFIME_STATUS_OK,
            "identity mixed alloc") &&
        check(gafime_gpu_matrix_alloc_v2(0, &fp64_desc, &fp64) == GAFIME_STATUS_OK,
            "identity fp64 alloc");
    if (!ok) {
        cleanup();
        return false;
    }
    ok = check(gafime_gpu_matrix_upload_f32_v2(fp32, features_f32, target_f32, rows, cols) ==
            GAFIME_STATUS_OK, "identity fp32 upload") &&
        check(gafime_gpu_matrix_upload_f32_v2(mixed, features_f32, target_f32, rows, cols) ==
            GAFIME_STATUS_OK, "identity mixed upload") &&
        check(gafime_gpu_matrix_upload_f64_v2(fp64, features_f64.data(), target_f64.data(), rows, cols) ==
            GAFIME_STATUS_OK, "identity fp64 upload");
    if (!ok) {
        cleanup();
        return false;
    }

    ProtocolFixture fixture(rows, cols);
    fixture.base.flags = GAFIME_LAUNCH_FLAG_GRAPH | GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    fixture.base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 0x250003ull;
    uint32_t combos32[2]{};
    float values32[8]{};
    uint32_t ranks32[2]{};
    uint32_t families32[2]{};
    uint64_t ids32[2]{};
    uint32_t flags32[2]{};
    GafimeResultTable result32{};
    init_result(&result32, kAbi10, combos32, values32, ranks32, families32, ids32, flags32);
    uint32_t combos_mixed[2]{};
    double values_mixed[8]{};
    uint32_t ranks_mixed[2]{};
    uint32_t families_mixed[2]{};
    uint64_t ids_mixed[2]{};
    uint32_t flags_mixed[2]{};
    GafimeResultTableF64 result_mixed{};
    init_result(
        &result_mixed, kAbi11, combos_mixed, values_mixed, ranks_mixed, families_mixed,
        ids_mixed, flags_mixed);
    uint32_t combos64[2]{};
    double values64[8]{};
    uint32_t ranks64[2]{};
    uint32_t families64[2]{};
    uint64_t ids64[2]{};
    uint32_t flags64[2]{};
    GafimeResultTableF64 result64{};
    init_result(&result64, kAbi11, combos64, values64, ranks64, families64, ids64, flags64);

    fixture.precision.profile = GAFIME_PRECISION_FP32;
    ok = check(gafime_gpu_execute_f32_v2(fp32, &fixture.precision, &result32) == GAFIME_STATUS_OK,
            "identity fp32 execute") &&
        check((result32.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "identity fp32 graph replay");
    fixture.precision.profile = GAFIME_PRECISION_MIXED;
    ok = ok && check(gafime_gpu_execute_f64_v2(mixed, &fixture.precision, &result_mixed) ==
            GAFIME_STATUS_OK, "identity mixed execute") &&
        check((result_mixed.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "identity mixed graph replay");
    fixture.precision.profile = GAFIME_PRECISION_FP64;
    ok = ok && check(gafime_gpu_execute_f64_v2(fp64, &fixture.precision, &result64) ==
            GAFIME_STATUS_OK, "identity fp64 execute") &&
        check((result64.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "identity fp64 graph replay");
    if (!ok) {
        cleanup();
        return false;
    }

    gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity fp32_id{};
    gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity mixed_id{};
    gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity fp64_id{};
    ok = check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(fp32, &fp32_id) ==
            GAFIME_STATUS_OK, "inspect fp32 identity") &&
        check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(mixed, &mixed_id) ==
            GAFIME_STATUS_OK, "inspect mixed identity") &&
        check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(fp64, &fp64_id) ==
            GAFIME_STATUS_OK, "inspect fp64 identity");
    const auto identity_matches = [&](const gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity& identity,
                                      uint32_t profile) {
        const uint32_t expected_storage = profile == GAFIME_PRECISION_FP64
            ? static_cast<uint32_t>(sizeof(double)) : static_cast<uint32_t>(sizeof(float));
        const uint32_t expected_reduction = profile == GAFIME_PRECISION_FP32
            ? static_cast<uint32_t>(sizeof(float)) : static_cast<uint32_t>(sizeof(double));
        return identity.profile == profile && identity.feature_stats_profile == profile &&
            identity.target_stats_profile == profile && identity.descriptor_profile == profile &&
            identity.descriptor_generation ==
                fixture.base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] &&
            identity.graph_profile == profile && identity.graph_valid == 1 &&
            identity.storage_bytes == expected_storage &&
            identity.accumulation_bytes == expected_reduction && identity.result_bytes == expected_reduction &&
            identity.resident_features != 0 && identity.resident_target != 0 &&
            identity.descriptor_combos != 0 && identity.graph_exec != 0;
    };
    ok = ok && check(identity_matches(fp32_id, GAFIME_PRECISION_FP32),
            "fp32 resident/stat/descriptor/graph profile identity") &&
        check(identity_matches(mixed_id, GAFIME_PRECISION_MIXED),
            "mixed resident/stat/descriptor/graph profile identity") &&
        check(identity_matches(fp64_id, GAFIME_PRECISION_FP64),
            "fp64 resident/stat/descriptor/graph profile identity") &&
        check(fp32_id.resident_features != mixed_id.resident_features &&
                fp32_id.resident_features != fp64_id.resident_features &&
                mixed_id.resident_features != fp64_id.resident_features &&
                fp32_id.resident_target != mixed_id.resident_target &&
                fp32_id.resident_target != fp64_id.resident_target &&
                mixed_id.resident_target != fp64_id.resident_target,
            "same descriptors retain distinct resident state per profile") &&
        check(fp32_id.descriptor_combos != mixed_id.descriptor_combos &&
                fp32_id.descriptor_combos != fp64_id.descriptor_combos &&
                mixed_id.descriptor_combos != fp64_id.descriptor_combos &&
                fp32_id.graph_exec != mixed_id.graph_exec &&
                fp32_id.graph_exec != fp64_id.graph_exec &&
                mixed_id.graph_exec != fp64_id.graph_exec,
            "same descriptors retain distinct descriptor and graph caches per profile") &&
        check(fp32_id.graph_metric_signature != mixed_id.graph_metric_signature &&
                fp32_id.graph_metric_signature != fp64_id.graph_metric_signature &&
                mixed_id.graph_metric_signature != fp64_id.graph_metric_signature,
            "graph replay signature includes profile");

    const float updated_target[rows] = {4, 5, 6, 2, 3, 1};
    ok = ok && check(gafime_gpu_matrix_update_target_f32_v2(fp32, updated_target, rows) ==
            GAFIME_STATUS_OK, "identity fp32 target update");
    gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity fp32_after_update{};
    gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity mixed_after_update{};
    gafime_cuda_v1::detail::PrecisionCudaMatrixIdentity fp64_after_update{};
    ok = ok && check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(fp32, &fp32_after_update) ==
            GAFIME_STATUS_OK, "inspect fp32 target update") &&
        check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(mixed, &mixed_after_update) ==
            GAFIME_STATUS_OK, "inspect mixed after fp32 target update") &&
        check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(fp64, &fp64_after_update) ==
            GAFIME_STATUS_OK, "inspect fp64 after fp32 target update") &&
        check(fp32_after_update.target_stats_profile == GAFIME_PRECISION_FP32 &&
                fp32_after_update.target_generation > fp32_id.target_generation &&
                fp32_after_update.descriptor_profile == 0 && fp32_after_update.graph_profile == 0 &&
                fp32_after_update.graph_valid == 0 && fp32_after_update.graph_metric_signature == 0,
            "target update invalidates only fp32 descriptor and graph state") &&
        check(mixed_after_update.target_generation == mixed_id.target_generation &&
                mixed_after_update.descriptor_profile == GAFIME_PRECISION_MIXED &&
                mixed_after_update.graph_profile == GAFIME_PRECISION_MIXED &&
                mixed_after_update.graph_valid == 1 &&
                mixed_after_update.graph_metric_signature == mixed_id.graph_metric_signature &&
                fp64_after_update.target_generation == fp64_id.target_generation &&
                fp64_after_update.descriptor_profile == GAFIME_PRECISION_FP64 &&
                fp64_after_update.graph_profile == GAFIME_PRECISION_FP64 &&
                fp64_after_update.graph_valid == 1 &&
                fp64_after_update.graph_metric_signature == fp64_id.graph_metric_signature,
            "target update cannot invalidate or reuse another profile cache");

    fixture.precision.profile = GAFIME_PRECISION_FP32;
    result32.flags = 0;
    ok = ok && check(gafime_gpu_execute_f32_v2(fp32, &fixture.precision, &result32) == GAFIME_STATUS_OK,
            "identity fp32 graph rebuild") &&
        check(gafime_cuda_v1::detail::inspect_precision_cuda_matrix(fp32, &fp32_after_update) ==
            GAFIME_STATUS_OK, "inspect fp32 graph rebuild") &&
        check(identity_matches(fp32_after_update, GAFIME_PRECISION_FP32),
            "fp32 graph rebuild retains fp32-only identity");
    cleanup();
    return ok;
}

bool run_family_and_arity_coverage() {
    constexpr uint64_t rows = 9;
    constexpr uint32_t cols = 5;
    std::vector<float> features_f32(rows * cols);
    std::vector<float> target_f32(rows);
    for (uint64_t row = 0; row < rows; ++row) {
        target_f32[row] = static_cast<float>(2 + row * 3 + (row % 2));
        for (uint32_t column = 0; column < cols; ++column) {
            features_f32[row * cols + column] = static_cast<float>(
                1 + (row + 1) * (column + 2) + (row % 3) * (column + 1));
        }
    }
    std::vector<double> features_f64(features_f32.begin(), features_f32.end());
    std::vector<double> target_f64(target_f32.begin(), target_f32.end());
    FamilyCoverageFixture fixture(rows, cols);
    fixture.base.flags = GAFIME_LAUNCH_FLAG_GRAPH | GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    fixture.base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 0x250004ull;

    const auto make_desc = [](GafimePrecisionProfile profile, uint32_t dtype, uint64_t bytes) {
        GafimePrecisionMatrixDesc desc{};
        desc.abi_version = kAbi11;
        desc.profile = profile;
        desc.dtype = dtype;
        desc.layout = GAFIME_MATRIX_ROW_MAJOR;
        desc.rows = rows;
        desc.cols = cols;
        desc.row_stride = cols;
        desc.bytes = bytes;
        return desc;
    };
    const GafimePrecisionMatrixDesc fp32_desc = make_desc(
        GAFIME_PRECISION_FP32, GAFIME_DTYPE_F32,
        static_cast<uint64_t>(features_f32.size() * sizeof(float)));
    const GafimePrecisionMatrixDesc mixed_desc = make_desc(
        GAFIME_PRECISION_MIXED, GAFIME_DTYPE_F32,
        static_cast<uint64_t>(features_f32.size() * sizeof(float)));
    const GafimePrecisionMatrixDesc fp64_desc = make_desc(
        GAFIME_PRECISION_FP64, GAFIME_DTYPE_F64,
        static_cast<uint64_t>(features_f64.size() * sizeof(double)));
    GafimeGpuMatrix fp32 = nullptr;
    GafimeGpuMatrix mixed = nullptr;
    GafimeGpuMatrix fp64 = nullptr;
    const auto cleanup = [&] {
        gafime_gpu_matrix_free(fp32);
        gafime_gpu_matrix_free(mixed);
        gafime_gpu_matrix_free(fp64);
    };
    bool ok = check(gafime_gpu_matrix_alloc_v2(0, &fp32_desc, &fp32) == GAFIME_STATUS_OK,
            "families fp32 alloc") &&
        check(gafime_gpu_matrix_alloc_v2(0, &mixed_desc, &mixed) == GAFIME_STATUS_OK,
            "families mixed alloc") &&
        check(gafime_gpu_matrix_alloc_v2(0, &fp64_desc, &fp64) == GAFIME_STATUS_OK,
            "families fp64 alloc") &&
        check(gafime_gpu_matrix_upload_f32_v2(
                fp32, features_f32.data(), target_f32.data(), rows, cols) == GAFIME_STATUS_OK,
            "families fp32 upload") &&
        check(gafime_gpu_matrix_upload_f32_v2(
                mixed, features_f32.data(), target_f32.data(), rows, cols) == GAFIME_STATUS_OK,
            "families mixed upload") &&
        check(gafime_gpu_matrix_upload_f64_v2(
                fp64, features_f64.data(), target_f64.data(), rows, cols) == GAFIME_STATUS_OK,
            "families fp64 upload");
    if (!ok) {
        cleanup();
        return false;
    }

    uint32_t combos32[25]{};
    float values32[20]{};
    uint32_t ranks32[5]{};
    uint32_t families32[5]{};
    uint64_t ids32[5]{};
    uint32_t flags32[5]{};
    GafimeResultTable result32{};
    init_result(&result32, kAbi10, combos32, values32, ranks32, families32, ids32, flags32);
    result32.max_arity = 5;
    result32.capacity = 5;

    uint32_t combos_mixed[25]{};
    double values_mixed[20]{};
    uint32_t ranks_mixed[5]{};
    uint32_t families_mixed[5]{};
    uint64_t ids_mixed[5]{};
    uint32_t flags_mixed[5]{};
    GafimeResultTableF64 result_mixed{};
    init_result(
        &result_mixed, kAbi11, combos_mixed, values_mixed, ranks_mixed, families_mixed,
        ids_mixed, flags_mixed);
    result_mixed.max_arity = 5;
    result_mixed.capacity = 5;

    uint32_t combos64[25]{};
    double values64[20]{};
    uint32_t ranks64[5]{};
    uint32_t families64[5]{};
    uint64_t ids64[5]{};
    uint32_t flags64[5]{};
    GafimeResultTableF64 result64{};
    init_result(&result64, kAbi11, combos64, values64, ranks64, families64, ids64, flags64);
    result64.max_arity = 5;
    result64.capacity = 5;

    fixture.precision.profile = GAFIME_PRECISION_FP32;
    ok = check(gafime_gpu_execute_f32_v2(fp32, &fixture.precision, &result32) == GAFIME_STATUS_OK,
            "families fp32 execute") &&
        check(result32.row_count == 5 && result_is_finite<float>(result32),
            "families fp32 all metrics") &&
        check((result32.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "families fp32 graph replay");
    fixture.precision.profile = GAFIME_PRECISION_MIXED;
    ok = ok && check(gafime_gpu_execute_f64_v2(mixed, &fixture.precision, &result_mixed) ==
            GAFIME_STATUS_OK, "families mixed execute") &&
        check(result_mixed.row_count == 5 && result_is_finite<double>(result_mixed),
            "families mixed f64 all metrics") &&
        check((result_mixed.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "families mixed graph replay");
    fixture.precision.profile = GAFIME_PRECISION_FP64;
    ok = ok && check(gafime_gpu_execute_f64_v2(fp64, &fixture.precision, &result64) ==
            GAFIME_STATUS_OK, "families fp64 execute") &&
        check(result64.row_count == 5 && result_is_finite<double>(result64),
            "families fp64 all metrics") &&
        check((result64.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0,
            "families fp64 graph replay");
    constexpr uint32_t expected_families[5] = {
        GAFIME_FAMILY_CONTINUOUS,
        GAFIME_FAMILY_TIME_SERIES,
        GAFIME_FAMILY_DECISION_PATH,
        GAFIME_FAMILY_CONTINUOUS,
        GAFIME_FAMILY_TIME_SERIES,
    };
    const auto full_result_has_families = [&](const uint32_t* families, const uint64_t* ids) {
        for (uint64_t row = 0; row < 5; ++row) {
            if (ids[row] != row || families[row] != expected_families[row]) return false;
        }
        return true;
    };
    ok = ok && check(full_result_has_families(families32, ids32),
            "fp32 preserves continuous/time-series/decision-path identities") &&
        check(full_result_has_families(families_mixed, ids_mixed),
            "mixed preserves continuous/time-series/decision-path identities") &&
        check(full_result_has_families(families64, ids64),
            "fp64 preserves continuous/time-series/decision-path identities");

    fixture.base.rank.top_k = 3;
    fixture.base.rank.primary_metric = GAFIME_METRIC_PEARSON;
    fixture.base.rank.descending = 1;
    const auto ranked_families_match_candidate_ids = [&](const uint32_t* families, const uint64_t* ids) {
        for (uint64_t row = 0; row < 3; ++row) {
            if (ids[row] >= 5 || families[row] != expected_families[ids[row]]) return false;
        }
        return true;
    };
    fixture.precision.profile = GAFIME_PRECISION_FP32;
    ok = ok && check(gafime_gpu_execute_f32_v2(fp32, &fixture.precision, &result32) == GAFIME_STATUS_OK,
            "families fp32 ranking") &&
        check(result32.row_count == 3 && ranked_families_match_candidate_ids(families32, ids32),
            "fp32 ranking retains family and candidate identity");
    fixture.precision.profile = GAFIME_PRECISION_MIXED;
    ok = ok && check(gafime_gpu_execute_f64_v2(mixed, &fixture.precision, &result_mixed) ==
            GAFIME_STATUS_OK, "families mixed ranking") &&
        check(result_mixed.row_count == 3 && ranked_families_match_candidate_ids(families_mixed, ids_mixed),
            "mixed ranking retains family and candidate identity");
    fixture.precision.profile = GAFIME_PRECISION_FP64;
    ok = ok && check(gafime_gpu_execute_f64_v2(fp64, &fixture.precision, &result64) ==
            GAFIME_STATUS_OK, "families fp64 ranking") &&
        check(result64.row_count == 3 && ranked_families_match_candidate_ids(families64, ids64),
            "fp64 ranking retains family and candidate identity");
    cleanup();
    return ok;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    GafimePrecisionCapabilities capabilities{};
    if (!check(gafime_gpu_precision_capabilities(0, &capabilities) == GAFIME_STATUS_OK,
            "precision capability query") ||
        !check(capabilities.profile_mask ==
                (GAFIME_PRECISION_PROFILE_MASK_FP32 | GAFIME_PRECISION_PROFILE_MASK_MIXED |
                    GAFIME_PRECISION_PROFILE_MASK_FP64),
            "all CUDA profiles advertised") ||
        !check(capabilities.storage_dtype_mask == (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64) &&
                capabilities.result_dtype_mask == (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64),
            "CUDA advertises typed storage and public results")) return 1;
    return run_fp32() && run_mixed() && run_fp64() && run_diagnostic_storage_width_contract() &&
            run_fp64_distinguishing_values() &&
            run_profile_identity_separation() && run_family_and_arity_coverage()
        ? 0
        : 1;
}
