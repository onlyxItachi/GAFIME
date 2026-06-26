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
    return 0;
}
