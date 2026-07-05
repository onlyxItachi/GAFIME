#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <string>
#include <vector>

#include "../../src/common/gafime_gpu_abi.hpp"

namespace {

struct BenchCase {
    uint64_t rows;
    uint32_t paths;
};

struct Box2 {
    float lo0;
    float hi0;
    float lo1;
    float hi1;
};

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

int require_status(int status, const char* label) {
    if (status != GAFIME_STATUS_OK) {
        std::fprintf(stderr, "%s failed with status %d\n", label, status);
        return 1;
    }
    return 0;
}

uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float unit_float(uint32_t x) {
    return static_cast<float>(hash32(x) & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

void build_features(uint64_t rows, std::vector<float>& row_major, std::vector<float>& feature_major) {
    row_major.resize(static_cast<size_t>(rows) * 2u);
    feature_major.resize(static_cast<size_t>(rows) * 2u);
    for (uint64_t row = 0; row < rows; ++row) {
        const float f0 = unit_float(static_cast<uint32_t>(row) * 17u + 3u);
        const float f1 = unit_float(static_cast<uint32_t>(row) * 29u + 11u);
        row_major[static_cast<size_t>(row) * 2u + 0u] = f0;
        row_major[static_cast<size_t>(row) * 2u + 1u] = f1;
        feature_major[row] = f0;
        feature_major[rows + row] = f1;
    }
}

void build_target(uint64_t rows, const std::vector<float>& feature_major, std::vector<float>& target) {
    target.resize(rows);
    const float* f0 = feature_major.data();
    const float* f1 = feature_major.data() + rows;
    for (uint64_t row = 0; row < rows; ++row) {
        target[row] = 0.65f * f0[row] + 0.35f * f1[row];
    }
}

void build_boxes_and_terms(
    uint32_t path_count,
    std::vector<Box2>& boxes,
    std::vector<GafimeDecisionPathTerm>& terms,
    std::vector<uint32_t>& offsets
) {
    boxes.resize(path_count);
    terms.resize(static_cast<size_t>(path_count) * 4u);
    offsets.resize(static_cast<size_t>(path_count) + 1u);
    for (uint32_t path = 0; path < path_count; ++path) {
        const float cx = 0.10f + 0.80f * unit_float(path * 101u + 7u);
        const float cy = 0.10f + 0.80f * unit_float(path * 131u + 13u);
        const float wx = 0.025f + 0.075f * unit_float(path * 151u + 17u);
        const float wy = 0.025f + 0.075f * unit_float(path * 181u + 19u);
        const Box2 box{
            std::max(0.0f, cx - wx),
            std::min(1.0f, cx + wx),
            std::max(0.0f, cy - wy),
            std::min(1.0f, cy + wy),
        };
        boxes[path] = box;
        offsets[path] = path * 4u;
        GafimeDecisionPathTerm* out = terms.data() + static_cast<size_t>(path) * 4u;
        out[0] = {0u, GAFIME_DECISION_PATH_SIGN_GT, box.lo0, 0u, {0u, 0u}};
        out[1] = {0u, GAFIME_DECISION_PATH_SIGN_LE, box.hi0, 0u, {0u, 0u}};
        out[2] = {1u, GAFIME_DECISION_PATH_SIGN_GT, box.lo1, 0u, {0u, 0u}};
        out[3] = {1u, GAFIME_DECISION_PATH_SIGN_LE, box.hi1, 0u, {0u, 0u}};
    }
    offsets[path_count] = path_count * 4u;
}

void cpu_membership_scalar(
    const float* feature_major,
    uint64_t rows,
    const std::vector<Box2>& boxes,
    float* out
) {
    const float* f0 = feature_major;
    const float* f1 = feature_major + rows;
    for (size_t path = 0; path < boxes.size(); ++path) {
        const Box2 box = boxes[path];
        float* row_out = out + static_cast<uint64_t>(path) * rows;
        for (uint64_t row = 0; row < rows; ++row) {
            const bool inside =
                f0[row] > box.lo0 && f0[row] <= box.hi0 &&
                f1[row] > box.lo1 && f1[row] <= box.hi1;
            row_out[row] = inside ? 1.0f : 0.0f;
        }
    }
}

void cpu_membership_avx512(
    const float* feature_major,
    uint64_t rows,
    const std::vector<Box2>& boxes,
    float* out
) {
#if defined(__AVX512F__)
    const float* f0 = feature_major;
    const float* f1 = feature_major + rows;
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const uint64_t vector_rows = rows / 16u * 16u;
    for (size_t path = 0; path < boxes.size(); ++path) {
        const Box2 box = boxes[path];
        const __m512 lo0 = _mm512_set1_ps(box.lo0);
        const __m512 hi0 = _mm512_set1_ps(box.hi0);
        const __m512 lo1 = _mm512_set1_ps(box.lo1);
        const __m512 hi1 = _mm512_set1_ps(box.hi1);
        float* row_out = out + static_cast<uint64_t>(path) * rows;
        uint64_t row = 0;
        for (; row < vector_rows; row += 16u) {
            const __m512 x0 = _mm512_loadu_ps(f0 + row);
            const __m512 x1 = _mm512_loadu_ps(f1 + row);
            __mmask16 mask = _mm512_cmp_ps_mask(x0, lo0, _CMP_GT_OQ);
            mask &= _mm512_cmp_ps_mask(x0, hi0, _CMP_LE_OQ);
            mask &= _mm512_cmp_ps_mask(x1, lo1, _CMP_GT_OQ);
            mask &= _mm512_cmp_ps_mask(x1, hi1, _CMP_LE_OQ);
            _mm512_storeu_ps(row_out + row, _mm512_mask_blend_ps(mask, zero, one));
        }
        for (; row < rows; ++row) {
            const bool inside =
                f0[row] > box.lo0 && f0[row] <= box.hi0 &&
                f1[row] > box.lo1 && f1[row] <= box.hi1;
            row_out[row] = inside ? 1.0f : 0.0f;
        }
    }
#else
    cpu_membership_scalar(feature_major, rows, boxes, out);
#endif
}

double time_cpu_avx512(
    const float* feature_major,
    uint64_t rows,
    const std::vector<Box2>& boxes,
    std::vector<float>& out,
    int repeats
) {
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < repeats; ++rep) {
        const auto start = Clock::now();
        cpu_membership_avx512(feature_major, rows, boxes, out.data());
        const auto stop = Clock::now();
        best = std::min(best, elapsed_seconds(start, stop));
    }
    return best;
}

double time_cpu_score_boxes(
    const float* feature_major,
    const std::vector<float>& target,
    uint64_t rows,
    const std::vector<Box2>& boxes,
    std::vector<float>& scores,
    int repeats
) {
    double best = std::numeric_limits<double>::infinity();
    scores.resize(boxes.size() * 2u);
    const float* f0 = feature_major;
    const float* f1 = feature_major + rows;
    for (int rep = 0; rep < repeats; ++rep) {
        const auto start = Clock::now();
        for (size_t path = 0; path < boxes.size(); ++path) {
            const Box2 box = boxes[path];
            double n = 0.0;
            double sx = 0.0;
            double sy = 0.0;
            double syy = 0.0;
            double sxy = 0.0;
            for (uint64_t row = 0; row < rows; ++row) {
                const float y_value = target[row];
                if (!std::isfinite(y_value)) {
                    continue;
                }
                const bool inside =
                    f0[row] > box.lo0 && f0[row] <= box.hi0 &&
                    f1[row] > box.lo1 && f1[row] <= box.hi1;
                const double x_value = inside ? 1.0 : 0.0;
                n += 1.0;
                sx += x_value;
                sy += y_value;
                syy += static_cast<double>(y_value) * y_value;
                sxy += x_value * y_value;
            }
            double pearson = 0.0;
            if (n > 0.0) {
                const double sxx_centered = std::max(0.0, sx - sx * sx / n);
                const double syy_centered = std::max(0.0, syy - sy * sy / n);
                const double sxy_centered = sxy - sx * sy / n;
                const double denom = std::sqrt(std::max(0.0, sxx_centered * syy_centered));
                if (denom > 0.0) {
                    pearson = std::clamp(sxy_centered / denom, -1.0, 1.0);
                }
            }
            scores[path * 2u + 0u] = static_cast<float>(pearson);
            scores[path * 2u + 1u] = static_cast<float>(std::clamp(pearson * pearson, 0.0, 1.0));
        }
        const auto stop = Clock::now();
        best = std::min(best, elapsed_seconds(start, stop));
    }
    return best;
}

double time_gpu_membership(
    GafimeGpuMatrix matrix,
    const std::vector<GafimeDecisionPathTerm>& terms,
    const std::vector<uint32_t>& offsets,
    uint32_t flags,
    std::vector<float>& out,
    int repeats
) {
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < repeats; ++rep) {
        std::fill(out.begin(), out.end(), -1.0f);
        GafimeDecisionPathBatch batch{};
        batch.abi_version = GAFIME_ABI_VERSION;
        batch.path_count = static_cast<uint32_t>(offsets.size() - 1u);
        batch.term_count = static_cast<uint32_t>(terms.size());
        batch.flags = flags;
        batch.terms = terms.data();
        batch.path_offsets = offsets.data();
        batch.membership_host = out.data();
        const auto start = Clock::now();
        const int status = gafime_gpu_decision_path_membership(matrix, &batch);
        const auto stop = Clock::now();
        if (require_status(status, "gafime_gpu_decision_path_membership")) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        best = std::min(best, elapsed_seconds(start, stop));
    }
    return best;
}

struct ScoreResult {
    std::vector<uint32_t> combo_indices;
    std::vector<float> metric_values;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> families;
    std::vector<uint64_t> candidate_ids;
    std::vector<uint32_t> row_flags;
    GafimeResultTable table{};

    ScoreResult(uint32_t path_count, uint32_t metric_count) :
        combo_indices(path_count, UINT32_MAX),
        metric_values(static_cast<size_t>(path_count) * metric_count, 0.0f),
        ranks(path_count, 0),
        families(path_count, 0),
        candidate_ids(path_count, 0),
        row_flags(path_count, 0)
    {
        table.abi_version = GAFIME_ABI_VERSION;
        table.max_arity = 1;
        table.metric_count = metric_count;
        table.capacity = path_count;
        table.combo_indices = combo_indices.data();
        table.metric_values = metric_values.data();
        table.ranks = ranks.data();
        table.families = families.data();
        table.candidate_ids = candidate_ids.data();
        table.row_flags = row_flags.data();
    }

    void reset() {
        std::fill(combo_indices.begin(), combo_indices.end(), UINT32_MAX);
        std::fill(metric_values.begin(), metric_values.end(), 0.0f);
        std::fill(ranks.begin(), ranks.end(), 0);
        std::fill(families.begin(), families.end(), 0);
        std::fill(candidate_ids.begin(), candidate_ids.end(), 0);
        std::fill(row_flags.begin(), row_flags.end(), 0);
        table.row_count = 0;
    }
};

double time_gpu_score(
    GafimeGpuMatrix matrix,
    const std::vector<GafimeDecisionPathTerm>& terms,
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& metric_ids,
    uint32_t flags,
    ScoreResult& result,
    int repeats
) {
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < repeats; ++rep) {
        result.reset();
        GafimeDecisionPathScoreBatch batch{};
        batch.abi_version = GAFIME_ABI_VERSION;
        batch.path_count = static_cast<uint32_t>(offsets.size() - 1u);
        batch.term_count = static_cast<uint32_t>(terms.size());
        batch.flags = flags;
        batch.terms = terms.data();
        batch.path_offsets = offsets.data();
        batch.metric_ids = metric_ids.data();
        batch.metric_count = static_cast<uint32_t>(metric_ids.size());
        const auto start = Clock::now();
        const int status = gafime_gpu_decision_path_score(matrix, &batch, &result.table);
        const auto stop = Clock::now();
        if (require_status(status, "gafime_gpu_decision_path_score")) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        best = std::min(best, elapsed_seconds(start, stop));
    }
    return best;
}

uint64_t compare_exact(const std::vector<float>& left, const std::vector<float>& right) {
    uint64_t mismatches = 0;
    for (size_t idx = 0; idx < left.size(); ++idx) {
        if (left[idx] != right[idx]) {
            ++mismatches;
        }
    }
    return mismatches;
}

void score_membership_cpu(
    const std::vector<float>& membership,
    const std::vector<float>& target,
    uint64_t rows,
    uint32_t path_count,
    std::vector<float>& scores
) {
    scores.assign(static_cast<size_t>(path_count) * 2u, 0.0f);
    for (uint32_t path = 0; path < path_count; ++path) {
        const float* member = membership.data() + static_cast<uint64_t>(path) * rows;
        double n = 0.0;
        double sx = 0.0;
        double sy = 0.0;
        double syy = 0.0;
        double sxy = 0.0;
        for (uint64_t row = 0; row < rows; ++row) {
            const float x_value = member[row];
            const float y_value = target[row];
            if (std::isfinite(x_value) && std::isfinite(y_value)) {
                n += 1.0;
                sx += x_value;
                sy += y_value;
                syy += static_cast<double>(y_value) * y_value;
                sxy += static_cast<double>(x_value) * y_value;
            }
        }
        double pearson = 0.0;
        if (n > 0.0) {
            const double sxx_centered = std::max(0.0, sx - sx * sx / n);
            const double syy_centered = std::max(0.0, syy - sy * sy / n);
            const double sxy_centered = sxy - sx * sy / n;
            const double denom = std::sqrt(std::max(0.0, sxx_centered * syy_centered));
            if (denom > 0.0) {
                pearson = std::clamp(sxy_centered / denom, -1.0, 1.0);
            }
        }
        scores[static_cast<size_t>(path) * 2u + 0u] = static_cast<float>(pearson);
        scores[static_cast<size_t>(path) * 2u + 1u] = static_cast<float>(std::clamp(pearson * pearson, 0.0, 1.0));
    }
}

float max_abs_diff(const std::vector<float>& left, const std::vector<float>& right) {
    float max_diff = 0.0f;
    const size_t count = std::min(left.size(), right.size());
    for (size_t idx = 0; idx < count; ++idx) {
        max_diff = std::max(max_diff, std::fabs(left[idx] - right[idx]));
    }
    return max_diff;
}

void print_result(
    const char* label,
    uint64_t evals,
    double seconds,
    uint64_t output_bytes
) {
    const double gevals = static_cast<double>(evals) / seconds / 1.0e9;
    const double gib = static_cast<double>(output_bytes) / seconds / 1073741824.0;
    std::printf(
        "%-18s %8.3f ms  %8.3f G eval/s  %8.3f GiB/s output\n",
        label,
        seconds * 1.0e3,
        gevals,
        gib
    );
}

}  // namespace

int main(int argc, char** argv) {
    bool score_only = false;
    std::vector<BenchCase> cases = {
        {65536u, 256u},
        {262144u, 256u},
        {262144u, 512u},
    };
    if (argc > 1) {
        cases.clear();
        for (int arg = 1; arg < argc; ++arg) {
            const std::string spec(argv[arg]);
            if (spec == "--score-only") {
                score_only = true;
                continue;
            }
            const size_t x = spec.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr, "case must be rowsxpath or --score-only, got %s\n", spec.c_str());
                return 2;
            }
            cases.push_back({
                static_cast<uint64_t>(std::strtoull(spec.substr(0, x).c_str(), nullptr, 10)),
                static_cast<uint32_t>(std::strtoul(spec.substr(x + 1).c_str(), nullptr, 10)),
            });
        }
        if (cases.empty()) {
            std::fprintf(stderr, "at least one rowsxpath case is required\n");
            return 2;
        }
    }

    GafimeGpuDeviceInfo info{};
    if (require_status(gafime_gpu_device_info(0, &info), "gafime_gpu_device_info")) {
        return 1;
    }
    std::printf("device: %s sm_%u%u memory %.2f GiB\n",
        info.name,
        info.compute_major,
        info.compute_minor,
        static_cast<double>(info.total_global_mem_bytes) / 1073741824.0
    );
#if defined(__AVX512F__)
    std::printf("cpu path: AVX512 membership comparator compiled in\n");
#else
    std::printf("cpu path: scalar fallback; compile with -mavx512f for AVX512\n");
#endif
    const char* rt_score_mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT_SCORE");
    std::printf("score path: %s\n", rt_score_mode == nullptr ? "bitset" : rt_score_mode);

    for (const BenchCase& bench : cases) {
        std::printf("\ncase: rows=%llu paths=%u evals=%.3fM output=%.2f MiB\n",
            static_cast<unsigned long long>(bench.rows),
            bench.paths,
            static_cast<double>(bench.rows) * bench.paths / 1.0e6,
            static_cast<double>(bench.rows) * bench.paths * sizeof(float) / 1048576.0
        );

        std::vector<float> row_major;
        std::vector<float> feature_major;
        std::vector<float> target;
        build_features(bench.rows, row_major, feature_major);
        build_target(bench.rows, feature_major, target);

        std::vector<Box2> boxes;
        std::vector<GafimeDecisionPathTerm> terms;
        std::vector<uint32_t> offsets;
        build_boxes_and_terms(bench.paths, boxes, terms, offsets);

        GafimeMatrixDesc desc{};
        desc.abi_version = GAFIME_ABI_VERSION;
        desc.dtype = GAFIME_DTYPE_F32;
        desc.layout = GAFIME_MATRIX_ROW_MAJOR;
        desc.rows = bench.rows;
        desc.cols = 2;
        desc.row_stride = 2;
        desc.bytes = bench.rows * 2u * sizeof(float);

        GafimeGpuMatrix matrix = nullptr;
        if (require_status(gafime_gpu_matrix_alloc(0, &desc, &matrix), "gafime_gpu_matrix_alloc")) {
            return 1;
        }
        const auto upload_start = Clock::now();
        const int upload_status = gafime_gpu_matrix_upload(matrix, row_major.data(), target.data(), bench.rows, 2);
        const auto upload_stop = Clock::now();
        if (require_status(upload_status, "gafime_gpu_matrix_upload")) {
            gafime_gpu_matrix_free(matrix);
            return 1;
        }
        std::printf("resident upload     %8.3f ms\n", elapsed_seconds(upload_start, upload_stop) * 1.0e3);

        const uint64_t output_len = bench.rows * static_cast<uint64_t>(bench.paths);
        const uint64_t output_bytes = output_len * sizeof(float);
        std::vector<uint32_t> score_metrics = {GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};
        std::vector<float> cpu_scores;
        ScoreResult gpu_rt_scores(bench.paths, static_cast<uint32_t>(score_metrics.size()));
        ScoreResult gpu_sm_scores(bench.paths, static_cast<uint32_t>(score_metrics.size()));

        if (score_only) {
            const double cpu_score_seconds = time_cpu_score_boxes(
                feature_major.data(),
                target,
                bench.rows,
                boxes,
                cpu_scores,
                1
            );
            const double rt_score_seconds = time_gpu_score(
                matrix,
                terms,
                offsets,
                score_metrics,
                GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
                gpu_rt_scores,
                3
            );
            const float rt_score_diff = max_abs_diff(cpu_scores, gpu_rt_scores.metric_values);

            const char* old_rt_mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT");
            const std::string old_rt_mode_value = old_rt_mode == nullptr ? std::string() : std::string(old_rt_mode);
            setenv("GAFIME_CUDA_DECISION_PATH_RT", "off", 1);
            const double sm_score_seconds = time_gpu_score(matrix, terms, offsets, score_metrics, 0, gpu_sm_scores, 2);
            if (old_rt_mode == nullptr) {
                unsetenv("GAFIME_CUDA_DECISION_PATH_RT");
            } else {
                setenv("GAFIME_CUDA_DECISION_PATH_RT", old_rt_mode_value.c_str(), 1);
            }
            const float sm_score_diff = max_abs_diff(cpu_scores, gpu_sm_scores.metric_values);

            print_result("cpu_score_ref", output_len, cpu_score_seconds, bench.paths * score_metrics.size() * sizeof(float));
            print_result("gpu_rt_score", output_len, rt_score_seconds, bench.paths * score_metrics.size() * sizeof(float));
            print_result("gpu_sm_score", output_len, sm_score_seconds, bench.paths * score_metrics.size() * sizeof(float));
            std::printf("score parity      rt_max_abs=%.6g sm_max_abs=%.6g\n", rt_score_diff, sm_score_diff);

            gafime_gpu_matrix_free(matrix);
            if (rt_score_diff > 1.0e-4f || sm_score_diff > 1.0e-4f) {
                return 1;
            }
            continue;
        }

        std::vector<float> cpu(output_len, 0.0f);
        std::vector<float> gpu_rt(output_len, 0.0f);
        std::vector<float> gpu_sm(output_len, 0.0f);
        const double cpu_seconds = time_cpu_avx512(feature_major.data(), bench.rows, boxes, cpu, 3);
        score_membership_cpu(cpu, target, bench.rows, bench.paths, cpu_scores);
        const double rt_seconds = time_gpu_membership(
            matrix,
            terms,
            offsets,
            GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            gpu_rt,
            3
        );
        const uint64_t rt_mismatches = compare_exact(cpu, gpu_rt);
        const double rt_score_seconds = time_gpu_score(
            matrix,
            terms,
            offsets,
            score_metrics,
            GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            gpu_rt_scores,
            3
        );
        const float rt_score_diff = max_abs_diff(cpu_scores, gpu_rt_scores.metric_values);

        const char* old_rt_mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT");
        const std::string old_rt_mode_value = old_rt_mode == nullptr ? std::string() : std::string(old_rt_mode);
        setenv("GAFIME_CUDA_DECISION_PATH_RT", "off", 1);
        const double sm_seconds = time_gpu_membership(matrix, terms, offsets, 0, gpu_sm, 2);
        const double sm_score_seconds = time_gpu_score(matrix, terms, offsets, score_metrics, 0, gpu_sm_scores, 2);
        if (old_rt_mode == nullptr) {
            unsetenv("GAFIME_CUDA_DECISION_PATH_RT");
        } else {
            setenv("GAFIME_CUDA_DECISION_PATH_RT", old_rt_mode_value.c_str(), 1);
        }
        const uint64_t sm_mismatches = compare_exact(cpu, gpu_sm);
        const float sm_score_diff = max_abs_diff(cpu_scores, gpu_sm_scores.metric_values);

        print_result("cpu_avx512", output_len, cpu_seconds, output_bytes);
        print_result("gpu_rt_abi", output_len, rt_seconds, output_bytes);
        print_result("gpu_sm_abi", output_len, sm_seconds, output_bytes);
        print_result("gpu_rt_score", output_len, rt_score_seconds, bench.paths * score_metrics.size() * sizeof(float));
        print_result("gpu_sm_score", output_len, sm_score_seconds, bench.paths * score_metrics.size() * sizeof(float));
        std::printf("parity            rt_mismatches=%llu sm_mismatches=%llu\n",
            static_cast<unsigned long long>(rt_mismatches),
            static_cast<unsigned long long>(sm_mismatches)
        );
        std::printf("score parity      rt_max_abs=%.6g sm_max_abs=%.6g\n", rt_score_diff, sm_score_diff);

        gafime_gpu_matrix_free(matrix);
        if (rt_mismatches != 0 || sm_mismatches != 0 || rt_score_diff > 1.0e-4f || sm_score_diff > 1.0e-4f) {
            return 1;
        }
    }
    return 0;
}
