#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../src/common/gafime_gpu_abi.hpp"

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
    GafimeGpuGraphCapability graph_capability{};
    if (require_status(gafime_gpu_graph_capability(0, &graph_capability), "graph_capability")) {
        return 1;
    }
    if (graph_capability.graph_mode != GAFIME_GRAPH_STREAM_CAPTURE ||
        graph_capability.supports_device_ranking != 1) {
        std::fprintf(stderr, "unexpected graph capability metadata\n");
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
    gafime_gpu_matrix_free(matrix);
    if (require_status(status, "gpu_execute_permutation")) {
        return 1;
    }
    if ((permutation_result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) == 0 ||
        permutation_result.row_count != 3 ||
        permutation_result_combos[0] != 0 ||
        permutation_result_combos[1] != 1 ||
        permutation_result_combos[2] != 2) {
        std::fprintf(stderr, "unexpected permutation graph result rows\n");
        return 1;
    }
    if (require_close(permutation_result_metrics[0], 1.0f, "permutation feature0 pearson")) {
        return 1;
    }
    if (require_close(permutation_result_metrics[2], -1.0f, "permutation feature1 pearson")) {
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
    return 0;
}
