#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "../../src/common/gafime_gpu_abi.hpp"

namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();
}

int require_ok(int status, const char* operation) {
    if (status == GAFIME_STATUS_OK) {
        return 0;
    }
    std::fprintf(stderr, "%s failed with status %d\n", operation, status);
    return 1;
}

struct ResultStorage {
    explicit ResultStorage(uint32_t candidates)
        : combos(candidates, UINT32_MAX),
          metrics(candidates, 0.0f),
          ranks(candidates, 0),
          families(candidates, 0),
          candidate_ids(candidates, 0),
          flags(candidates, 0) {
        table.abi_version = GAFIME_ABI_VERSION;
        table.max_arity = 1;
        table.metric_count = 1;
        table.capacity = candidates;
        table.combo_indices = combos.data();
        table.metric_values = metrics.data();
        table.ranks = ranks.data();
        table.families = families.data();
        table.candidate_ids = candidate_ids.data();
        table.row_flags = flags.data();
    }

    std::vector<uint32_t> combos;
    std::vector<float> metrics;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> families;
    std::vector<uint64_t> candidate_ids;
    std::vector<uint32_t> flags;
    GafimeResultTable table{};
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s ROWS CANDIDATES WARMUPS REPEATS\n", argv[0]);
        return 2;
    }
    const uint64_t rows = std::strtoull(argv[1], nullptr, 10);
    const uint32_t candidates = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    const uint32_t warmups = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    const uint32_t repeats = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    if (rows == 0 || candidates == 0 || repeats == 0) {
        return 2;
    }

    std::vector<float> features(static_cast<size_t>(rows) * candidates);
    std::vector<float> target(static_cast<size_t>(rows));
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t y_code = (row * 104729u + row / 7u + 17u) % 100003u;
        target[static_cast<size_t>(row)] = static_cast<float>(y_code) / 100003.0f;
        for (uint32_t column = 0; column < candidates; ++column) {
            const uint64_t code =
                (row * (8191u + 2u * column) + row / (3u + column % 11u) +
                 97u * column) %
                100019u;
            features[static_cast<size_t>(row) * candidates + column] =
                static_cast<float>(code) / 100019.0f;
        }
    }

    GafimeMatrixDesc desc{};
    desc.abi_version = GAFIME_ABI_VERSION;
    desc.dtype = GAFIME_DTYPE_F32;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = rows;
    desc.cols = candidates;
    desc.row_stride = candidates;
    desc.bytes = rows * candidates * sizeof(float);

    GafimeGpuMatrix matrix = nullptr;
    if (require_ok(gafime_gpu_matrix_alloc(0, &desc, &matrix), "matrix_alloc")) {
        return 1;
    }
    if (require_ok(
            gafime_gpu_matrix_upload(matrix, features.data(), target.data(), rows, candidates),
            "matrix_upload"
        )) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }

    std::vector<uint32_t> combo_indices(candidates);
    std::iota(combo_indices.begin(), combo_indices.end(), 0u);
    const uint32_t metric_id = GAFIME_METRIC_SPEARMAN;
    GafimeArityChunk chunk{};
    chunk.arity = 1;
    chunk.family = GAFIME_FAMILY_CONTINUOUS;
    chunk.combo_count = candidates;
    chunk.descriptor_count = candidates;

    GafimeLaunchProtocol protocol{};
    protocol.abi_version = GAFIME_ABI_VERSION;
    protocol.backend_kind = GAFIME_BACKEND_CUDA;
    protocol.flags = GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    protocol.max_arity = 1;
    protocol.n_samples = rows;
    protocol.n_features = candidates;
    protocol.family_count = 1;
    protocol.combo_indices = {combo_indices.data(), combo_indices.size()};
    protocol.metric_ids = {&metric_id, 1};
    protocol.chunks = &chunk;
    protocol.chunk_count = 1;
    protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 1;

    ResultStorage result(candidates);
    auto execute = [&]() {
        return require_ok(gafime_gpu_execute(matrix, &protocol, &result.table), "execute");
    };

    const auto first_start = std::chrono::steady_clock::now();
    if (execute()) {
        gafime_gpu_matrix_free(matrix);
        return 1;
    }
    const double first_ms = elapsed_ms(first_start);
    for (uint32_t warmup = 0; warmup < warmups; ++warmup) {
        if (execute()) {
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
    }

    std::vector<double> samples;
    samples.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        if (execute()) {
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
        samples.push_back(elapsed_ms(start));
    }
    std::sort(samples.begin(), samples.end());
    const double median_ms = samples[samples.size() / 2];
    double checksum = 0.0;
    for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
        checksum += static_cast<double>(candidate + 1) * result.metrics[candidate];
    }
    std::printf(
        "rows=%llu candidates=%u first_ms=%.6f warm_median_ms=%.6f "
        "warm_best_ms=%.6f candidate_sample_gevals_per_s=%.6f checksum=%.9f\n",
        static_cast<unsigned long long>(rows),
        candidates,
        first_ms,
        median_ms,
        samples.front(),
        (static_cast<double>(rows) * candidates) / (median_ms * 1.0e6),
        checksum
    );
    gafime_gpu_matrix_free(matrix);
    return result.table.row_count == candidates ? 0 : 1;
}
