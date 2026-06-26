#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../gpu/include/gafime_gpu_abi.h"

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

}  // namespace

int main() {
    GafimeGpuDeviceInfo info{};
    if (require_status(gafime_gpu_device_info(0, &info), "device_info")) {
        return 1;
    }
    if (info.backend_kind != GAFIME_BACKEND_CUDA || info.abi_version != GAFIME_ABI_VERSION) {
        std::fprintf(stderr, "device_info returned invalid ABI metadata\n");
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

    int status = gafime_gpu_execute(matrix, &protocol, &result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute")) {
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
    mixed_chunks[1].descriptor_offset = 3;
    mixed_chunks[1].descriptor_count = 3;

    GafimeLaunchProtocol mixed_protocol{};
    mixed_protocol.abi_version = GAFIME_ABI_VERSION;
    mixed_protocol.backend_kind = GAFIME_BACKEND_CUDA;
    mixed_protocol.max_arity = 2;
    mixed_protocol.n_samples = 4;
    mixed_protocol.n_features = 3;
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
    status = gafime_gpu_execute(matrix, &mixed_protocol, &mixed_result);
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute_mixed_second")) {
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
    return 0;
}
