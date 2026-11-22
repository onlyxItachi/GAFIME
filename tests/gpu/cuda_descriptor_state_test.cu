// NVCC-owned host-only test of the actual launcher helpers; no CUDA calls.
#include "../../src/cuda/precision_launcher.cu"

#include <cstdlib>

// These checks must execute in Release/RelWithDebInfo builds too.
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d: %s failed\\n", __FILE__, __LINE__, #condition); \
    std::exit(1); \
} } while (false)

int main() {
    PrecisionCudaMatrix matrix{};
    matrix.rows = 4;
    matrix.cols = 2;
    matrix.feature_abs_exponents = {0, 120};
    matrix.target_abs_exponent = 0;
    uint32_t combos[] = {0, 1, 1, 1};
    uint32_t metric = GAFIME_METRIC_PEARSON;
    GafimeArityChunk chunks[2]{};
    chunks[0].arity = chunks[1].arity = 1;
    chunks[0].family = chunks[1].family = GAFIME_FAMILY_CONTINUOUS;
    chunks[0].combo_count = 4;
    GafimeLaunchProtocol base{};
    base.combo_indices = {combos, 4};
    base.metric_ids = {&metric, 1};
    base.chunks = chunks;
    base.chunk_count = 1;
    base.flags = GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL | GAFIME_LAUNCH_FLAG_GRAPH;
    base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 123;
    GafimePrecisionLaunchProtocol protocol{};
    protocol.profile = GAFIME_PRECISION_FP32;
    protocol.base = &base;
    matrix.descriptor_profile = protocol.profile;
    matrix.descriptor_generation = 123;
    matrix.descriptor_combo_len = 4;
    matrix.descriptor_metric_id_len = 1;
    CHECK(descriptors_resident(&matrix, &protocol));
    refresh_covariance_layout(&matrix, &base);
    CHECK(matrix.covariance_modes.size() == 1);
    const auto* same_storage = matrix.covariance_modes.data();
    refresh_covariance_layout(&matrix, &base);
    CHECK(matrix.covariance_modes.data() == same_storage);

    base.chunk_count = 2;
    chunks[0].combo_count = 1;
    chunks[1].descriptor_offset = 1;
    chunks[1].combo_count = 3;
    CHECK(descriptors_resident(&matrix, &protocol));
    CHECK(!covariance_layout_matches(&matrix, &base));
    // Fail closed even if an internal caller bypasses acquisition/refresh.
    CHECK(!graph_shape_matches(&matrix, &protocol, 4, false));
    CHECK(store_graph_shape(&matrix, &protocol, 4, false) == GAFIME_STATUS_DEVICE_ERROR);
    CHECK(launch_precision_score_kernels(&matrix, &base, nullptr) == GAFIME_STATUS_DEVICE_ERROR);
    refresh_covariance_layout(&matrix, &base);
    CHECK(matrix.covariance_modes == covariance_modes_for_protocol(&matrix, &base));
    const auto first_partition = matrix.covariance_modes;
    CHECK(first_partition[0] != first_partition[1]);

    chunks[0].combo_count = 3;
    chunks[1].descriptor_offset = 3;
    chunks[1].combo_count = 1;
    CHECK(!covariance_layout_matches(&matrix, &base));
    refresh_covariance_layout(&matrix, &base);
    CHECK(matrix.covariance_modes == covariance_modes_for_protocol(&matrix, &base));
    CHECK(matrix.covariance_modes != first_partition);
    chunks[0].family = GAFIME_FAMILY_TIME_SERIES;
    CHECK(covariance_layout_matches(&matrix, &base));

    // Both scoring and permutation graph bookkeeping share this guarded sink.
    CHECK(store_graph_shape(&matrix, &protocol, 4, false) == GAFIME_STATUS_OK);
    CHECK(store_graph_shape(&matrix, &protocol, 4, true) == GAFIME_STATUS_OK);
    CHECK(matrix.graph_chunk_shapes.size() == 2);
    invalidate_precision_descriptors(&matrix);
    CHECK(matrix.covariance_chunks.empty() && matrix.covariance_modes.empty());
    CHECK(!descriptors_resident(&matrix, &protocol));
    std::puts("CUDA host descriptor-state controls passed (no CUDA calls)");
}
