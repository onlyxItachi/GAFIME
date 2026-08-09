#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "../../src/common/gafime_gpu_abi.hpp"

#ifndef GAFIME_METAL_TIMING_DEFAULT_LIBRARY_PATH
#define GAFIME_METAL_TIMING_DEFAULT_LIBRARY_PATH ""
#endif

#ifndef GAFIME_METAL_TIMING_SHADER_SOURCE
#define GAFIME_METAL_TIMING_SHADER_SOURCE ""
#endif

namespace {

constexpr uint32_t kReduceWidth = 64;
constexpr uint32_t kMaxPartialBlocks = 4096;
constexpr uint32_t kPrecisionFp32 = 1;
constexpr uint32_t kMetricPearson = 1;
constexpr uint32_t kMetricSpearman = 2;
constexpr uint32_t kMetricMutualInfo = 3;
constexpr uint32_t kMetricR2 = 4;
constexpr uint32_t kDefaultRows = 1024;
constexpr uint32_t kDefaultCandidates = 12;
constexpr uint32_t kDefaultWarmups = 10;
constexpr uint32_t kDefaultRepeats = 30;
constexpr double kSampleRegionTargetUs = 5000.0;
constexpr double kSampleRegionCalibrationTargetUs = kSampleRegionTargetUs * 2.0;
constexpr uint32_t kMaxLoopCount = 1u << 20;
constexpr uint32_t kBootstrapResamples = 2000;
constexpr uint64_t kBootstrapSeed = 20260809ULL;

struct Options {
    uint32_t rows = kDefaultRows;
    uint32_t candidates = kDefaultCandidates;
    uint32_t mi_bins = 24;
    uint32_t top_k = 8;
    uint32_t warmups = kDefaultWarmups;
    uint32_t repeats = kDefaultRepeats;
    std::string json_path;
    std::string metallib_path = GAFIME_METAL_TIMING_DEFAULT_LIBRARY_PATH;
    std::string shader_source_path = GAFIME_METAL_TIMING_SHADER_SOURCE;
    std::string payload_path;
    std::string wheel_path;
    std::string source_commit;
};

// These layouts are a test-side mirror of shader.metal. Static assertions make
// host/shader drift visible during the Apple build instead of silently timing
// an incorrectly encoded workload.
struct MetalChunk {
    uint32_t arity;
    uint32_t mi_bins;
    uint32_t scaled_covariance;
    uint32_t reserved;
    uint64_t descriptor_offset;
    uint64_t combo_count;
    uint64_t global_row_offset;
};

struct MetalLaunchInfo {
    uint64_t rows;
    uint32_t cols;
    uint32_t metric_count;
    uint32_t chunk_count;
    uint32_t precision_profile;
};

struct MetalRankInfo {
    uint64_t row_count;
    uint32_t metric_count;
    uint32_t primary_metric_index;
    uint32_t top_k;
    uint32_t partial_block_count;
};

static_assert(sizeof(MetalChunk) == 40);
static_assert(offsetof(MetalChunk, descriptor_offset) == 16);
static_assert(sizeof(MetalLaunchInfo) == 24);
static_assert(offsetof(MetalLaunchInfo, precision_profile) == 20);
static_assert(sizeof(MetalRankInfo) == 24);
static_assert(offsetof(MetalRankInfo, partial_block_count) == 20);

struct TimingRecord {
    std::string operation;
    std::string metric;
    std::string clock;
    std::string synchronization;
    std::vector<double> samples_us;
    std::vector<double> raw_samples_us;
    std::vector<double> host_synchronized_samples_us;
    std::vector<double> raw_host_synchronized_samples_us;
    std::vector<uint32_t> loop_counts_per_sample;
    uint32_t loop_count_per_sample = 1;
    uint32_t gpu_timestamp_valid_samples = 0;
};

[[noreturn]] void fail(const std::string& message);

// The benchmark binary is deliberately not linked against the staged payload.
// It must load the exact dylib supplied by the wheel and call the canonical
// ABI 1.1 surface so the event/timing artifact cannot silently drift onto the
// helper's private shader-only implementation.
struct CanonicalPayloadApi {
    using RoutesFn = int (*)(
        uint32_t,
        uint32_t,
        uint32_t,
        GafimeNumericRoute*,
        uint32_t,
        uint32_t*
    );
    using MatrixAllocFn = int (*)(
        uint32_t,
        const GafimeNumericMatrixDesc*,
        GafimeGpuMatrix*
    );
    using MatrixUploadFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericRoute*,
        const GafimeConstBufferView*,
        const GafimeConstBufferView*,
        uint64_t,
        uint32_t
    );
    using ExecuteFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericLaunchProtocol*,
        GafimeNumericResultTable*
    );
    using MatrixFreeFn = int (*)(GafimeGpuMatrix);

    void* handle = nullptr;
    RoutesFn routes = nullptr;
    MatrixAllocFn matrix_alloc = nullptr;
    MatrixUploadFn matrix_upload = nullptr;
    ExecuteFn execute = nullptr;
    MatrixFreeFn matrix_free = nullptr;

    explicit CanonicalPayloadApi(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* detail = dlerror();
            fail(
                "failed to load exact Metal payload dylib: " +
                std::string(detail == nullptr ? "unknown dynamic-loader error" : detail));
        }
        routes = load<RoutesFn>("gafime_gpu_numeric_routes_v2");
        matrix_alloc = load<MatrixAllocFn>("gafime_gpu_matrix_alloc_v2");
        matrix_upload = load<MatrixUploadFn>("gafime_gpu_matrix_upload_v2");
        execute = load<ExecuteFn>("gafime_gpu_execute_v2");
        matrix_free = load<MatrixFreeFn>("gafime_gpu_matrix_free_v2");
    }

    CanonicalPayloadApi(const CanonicalPayloadApi&) = delete;
    CanonicalPayloadApi& operator=(const CanonicalPayloadApi&) = delete;

    ~CanonicalPayloadApi() {
        if (handle != nullptr) dlclose(handle);
    }

    template <typename Fn>
    Fn load(const char* name) {
        dlerror();
        void* symbol = dlsym(handle, name);
        const char* detail = dlerror();
        if (symbol == nullptr || detail != nullptr) {
            const std::string detail_text =
                detail == nullptr ? "symbol lookup failed" : detail;
            if (handle != nullptr) dlclose(handle);
            handle = nullptr;
            fail(
                "exact Metal payload is missing canonical ABI symbol " +
                std::string(name) + ": " +
                detail_text);
        }
        static_assert(sizeof(Fn) == sizeof(symbol));
        Fn function = nullptr;
        std::memcpy(&function, &symbol, sizeof(function));
        return function;
    }
};

class ScopedEnvironmentOverride {
public:
    ScopedEnvironmentOverride(const char* name, const std::string& value)
        : name_(name), had_previous_(std::getenv(name) != nullptr) {
        if (had_previous_) previous_ = std::getenv(name);
        if (setenv(name, value.c_str(), 1) != 0) {
            fail("failed to set " + name_ + " for exact Metal payload");
        }
    }

    ScopedEnvironmentOverride(const ScopedEnvironmentOverride&) = delete;
    ScopedEnvironmentOverride& operator=(const ScopedEnvironmentOverride&) = delete;

    ~ScopedEnvironmentOverride() {
        if (had_previous_) {
            (void)setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::string previous_;
    bool had_previous_;
};

struct CanonicalPayloadEvidence {
    bool validated = false;
    uint32_t route_count = 0;
    int route_query_status = GAFIME_STATUS_DEVICE_ERROR;
    int route_fill_status = GAFIME_STATUS_DEVICE_ERROR;
    int matrix_alloc_status = GAFIME_STATUS_DEVICE_ERROR;
    int matrix_upload_status = GAFIME_STATUS_DEVICE_ERROR;
    int matrix_free_status = GAFIME_STATUS_DEVICE_ERROR;
    bool mixed_route_rejected = false;
    std::vector<std::string> symbols;
};

struct Sha256 {
    static constexpr std::array<uint32_t, 64> kRoundConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    std::array<uint32_t, 8> state = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<uint8_t, 64> block{};
    size_t block_size = 0;
    uint64_t bit_count = 0;

    static uint32_t rotate_right(uint32_t value, uint32_t amount) {
        return (value >> amount) | (value << (32u - amount));
    }

    void transform() {
        std::array<uint32_t, 64> words{};
        for (uint32_t index = 0; index < 16; ++index) {
            const uint32_t offset = index * 4u;
            words[index] = (static_cast<uint32_t>(block[offset]) << 24u) |
                (static_cast<uint32_t>(block[offset + 1]) << 16u) |
                (static_cast<uint32_t>(block[offset + 2]) << 8u) |
                static_cast<uint32_t>(block[offset + 3]);
        }
        for (uint32_t index = 16; index < 64; ++index) {
            const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (uint32_t index = 0; index < 64; ++index) {
            const uint32_t sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sigma1 + choose + kRoundConstants[index] +
                words[index];
            const uint32_t sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    void update(const uint8_t* data, size_t size) {
        bit_count += static_cast<uint64_t>(size) * 8u;
        for (size_t index = 0; index < size; ++index) {
            block[block_size++] = data[index];
            if (block_size == block.size()) {
                transform();
                block_size = 0;
            }
        }
    }

    std::string finish() {
        const uint64_t original_bit_count = bit_count;
        block[block_size++] = 0x80u;
        if (block_size > 56) {
            while (block_size < block.size()) block[block_size++] = 0;
            transform();
            block_size = 0;
        }
        while (block_size < 56) block[block_size++] = 0;
        for (int index = 7; index >= 0; --index) {
            block[block_size++] =
                static_cast<uint8_t>(original_bit_count >> (index * 8));
        }
        transform();
        std::ostringstream digest;
        digest << std::hex << std::setfill('0');
        for (const uint32_t value : state) digest << std::setw(8) << value;
        return digest.str();
    }
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string canonical_path(const std::string& path) {
    if (path.empty()) fail("required artifact path is empty");
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(path);
    if (!std::filesystem::is_regular_file(resolved)) {
        fail("artifact is not a regular file: " + resolved.string());
    }
    return resolved.string();
}

std::string sha256_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot open file for SHA-256: " + path);
    Sha256 hash;
    std::array<uint8_t, 64 * 1024> buffer{};
    while (file) {
        file.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = file.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!file.eof()) fail("failed while hashing: " + path);
    return hash.finish();
}

std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char character : value) {
        switch (character) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20u) {
                std::ostringstream encoded;
                encoded << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(
                               static_cast<unsigned char>(character));
                escaped += encoded.str();
            } else {
                escaped += character;
            }
        }
    }
    return escaped;
}

uint32_t parse_u32(const char* text, const char* option) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
        fail(std::string("invalid ") + option + " value");
    }
    return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto value_for = [&](const char* option) -> const char* {
            if (index + 1 >= argc) fail(std::string("missing value for ") + option);
            return argv[++index];
        };
        if (argument == "--rows") {
            options.rows = parse_u32(value_for("--rows"), "--rows");
        } else if (argument == "--candidates") {
            options.candidates =
                parse_u32(value_for("--candidates"), "--candidates");
        } else if (argument == "--mi-bins") {
            options.mi_bins = parse_u32(value_for("--mi-bins"), "--mi-bins");
        } else if (argument == "--top-k") {
            options.top_k = parse_u32(value_for("--top-k"), "--top-k");
        } else if (argument == "--warmups") {
            options.warmups = parse_u32(value_for("--warmups"), "--warmups");
        } else if (argument == "--repeats") {
            options.repeats = parse_u32(value_for("--repeats"), "--repeats");
        } else if (argument == "--json") {
            options.json_path = value_for("--json");
        } else if (argument == "--metallib") {
            options.metallib_path = value_for("--metallib");
        } else if (argument == "--shader-source") {
            options.shader_source_path = value_for("--shader-source");
        } else if (argument == "--payload") {
            options.payload_path = value_for("--payload");
        } else if (argument == "--wheel") {
            options.wheel_path = value_for("--wheel");
        } else if (argument == "--source-commit") {
            options.source_commit = value_for("--source-commit");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: " << argv[0]
                << " --json PATH --metallib PATH --shader-source PATH"
                << " --payload PATH --wheel PATH --source-commit SHA"
                << " [--rows N] [--candidates N] [--mi-bins N] [--top-k N]"
                << " [--warmups N] [--repeats N]\n";
            std::exit(0);
        } else {
            fail("unknown option: " + std::string(argument));
        }
    }
    const bool commit_is_full_sha = options.source_commit.size() == 40 &&
        std::all_of(
            options.source_commit.begin(), options.source_commit.end(),
            [](unsigned char value) { return std::isxdigit(value) != 0; });
    if (options.rows < 128 || options.rows > 4096 || options.candidates < 2 ||
        options.mi_bins > 48 || options.top_k == 0 ||
        options.top_k > options.candidates || options.warmups < kDefaultWarmups ||
        options.repeats < kDefaultRepeats || options.json_path.empty() ||
        !commit_is_full_sha) {
        fail(
            "invalid dimensions/provenance: rows must be 128..4096, candidates must be >= 2, "
            "and top-k must be positive, "
            "MI bins must be 2..48, warmups >= 10, repeats >= 30, JSON output and "
            "a full source commit SHA are required");
    }
    options.metallib_path = canonical_path(options.metallib_path);
    options.shader_source_path = canonical_path(options.shader_source_path);
    options.payload_path = canonical_path(options.payload_path);
    options.wheel_path = canonical_path(options.wheel_path);
    options.json_path = std::filesystem::absolute(options.json_path).string();
    return options;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2u;
    if (values.size() % 2u == 0) {
        return (values[middle - 1] + values[middle]) * 0.5;
    }
    return values[middle];
}

double mean(const std::vector<double>& values) {
    return values.empty()
        ? 0.0
        : std::accumulate(values.begin(), values.end(), 0.0) /
              static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(values.size() - 1, lower + 1);
    const double weight = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * weight;
}

double median_absolute_deviation(const std::vector<double>& values) {
    const double center = median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) deviations.push_back(std::abs(value - center));
    return median(std::move(deviations));
}

uint64_t stable_seed(const TimingRecord& record) {
    uint64_t hash = kBootstrapSeed;
    const auto mix = [&](std::string_view value) {
        for (const unsigned char character : value) {
            hash ^= static_cast<uint64_t>(character);
            hash *= 1099511628211ULL;
        }
    };
    mix(record.operation);
    mix(record.metric);
    return hash;
}

std::array<double, 2> bootstrap_median_ci(
    const std::vector<double>& values,
    uint64_t seed
) {
    if (values.empty()) return {0.0, 0.0};
    std::mt19937_64 generator(seed);
    std::vector<double> bootstrap;
    bootstrap.reserve(kBootstrapResamples);
    std::vector<double> resample(values.size());
    for (uint32_t iteration = 0; iteration < kBootstrapResamples; ++iteration) {
        for (double& value : resample) {
            value = values[static_cast<size_t>(generator() % values.size())];
        }
        bootstrap.push_back(median(resample));
    }
    const double lower = percentile(bootstrap, 0.025);
    const double upper = percentile(std::move(bootstrap), 0.975);
    return {lower, upper};
}

template <typename Fn>
TimingRecord time_host(
    std::string operation,
    std::string metric,
    uint32_t warmups,
    uint32_t repeats,
    Fn&& operation_fn
) {
    for (uint32_t index = 0; index < warmups; ++index) operation_fn();
    TimingRecord record{
        std::move(operation),
        std::move(metric),
        "host_steady_clock",
        "not_applicable_host_only",
        {},
        {},
        {},
        {},
        {},
        1,
        0,
    };
    auto measure = [&](uint32_t loop_count) {
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t loop = 0; loop < loop_count; ++loop) operation_fn();
        const auto stop = std::chrono::steady_clock::now();
        return std::max(1.0e-6,
                        std::chrono::duration<double, std::micro>(stop - start).count());
    };
    uint32_t loop_count = 1;
    double calibration_us = measure(loop_count);
    while (calibration_us < kSampleRegionCalibrationTargetUs && loop_count < kMaxLoopCount) {
        loop_count = loop_count > kMaxLoopCount / 2 ? kMaxLoopCount : loop_count * 2;
        calibration_us = measure(loop_count);
    }
    record.loop_count_per_sample = loop_count;
    record.samples_us.reserve(repeats);
    record.raw_samples_us.reserve(repeats);
    record.loop_counts_per_sample.reserve(repeats);
    for (uint32_t index = 0; index < repeats; ++index) {
        uint32_t sample_loop_count = loop_count;
        double raw_us = measure(sample_loop_count);
        while (raw_us < kSampleRegionTargetUs && sample_loop_count < kMaxLoopCount) {
            sample_loop_count = sample_loop_count > kMaxLoopCount / 2
                ? kMaxLoopCount : sample_loop_count * 2;
            raw_us = measure(sample_loop_count);
        }
        loop_count = std::max(loop_count, sample_loop_count);
        record.loop_counts_per_sample.push_back(sample_loop_count);
        record.raw_samples_us.push_back(raw_us);
        record.samples_us.push_back(std::max(1.0e-6, raw_us / sample_loop_count));
    }
    record.loop_count_per_sample = loop_count;
    return record;
}

void require_completed(id<MTLCommandBuffer> command_buffer, const char* operation) {
    if (command_buffer == nil ||
        command_buffer.status != MTLCommandBufferStatusCompleted) {
        std::string detail = operation;
        detail += " command buffer did not complete";
        if (command_buffer != nil && command_buffer.error != nil) {
            detail += ": ";
            detail += command_buffer.error.localizedDescription.UTF8String;
        }
        fail(detail);
    }
}

template <typename Encode>
TimingRecord time_command_buffers(
    id<MTLCommandQueue> queue,
    std::string operation,
    std::string metric,
    uint32_t warmups,
    uint32_t repeats,
    Encode&& encode
) {
    auto submit = [&](uint32_t loop_count) -> std::pair<double, double> {
        @autoreleasepool {
            const auto host_start = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            if (command_buffer == nil) fail("failed to allocate Metal command buffer");
            for (uint32_t loop = 0; loop < loop_count; ++loop) encode(command_buffer);
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            const auto host_stop = std::chrono::steady_clock::now();
            require_completed(command_buffer, operation.c_str());
            double gpu_us = 0.0;
            if (@available(macOS 10.15, *)) {
                const CFTimeInterval start = command_buffer.GPUStartTime;
                const CFTimeInterval stop = command_buffer.GPUEndTime;
                if (std::isfinite(start) && std::isfinite(stop) && start > 0.0 &&
                    stop > start) {
                    gpu_us = (stop - start) * 1.0e6;
                }
            }
            const double host_us =
                std::chrono::duration<double, std::micro>(host_stop - host_start).count();
            return {gpu_us, host_us};
        }
    };
    for (uint32_t index = 0; index < warmups; ++index) (void)submit(1);
    uint32_t loop_count = 1;
    auto calibration = submit(loop_count);
    double calibration_us = calibration.first > 0.0 ? calibration.first : calibration.second;
    while (calibration_us < kSampleRegionCalibrationTargetUs &&
           loop_count < kMaxLoopCount) {
        loop_count = loop_count > kMaxLoopCount / 2 ? kMaxLoopCount : loop_count * 2;
        calibration = submit(loop_count);
        calibration_us = calibration.first > 0.0 ? calibration.first : calibration.second;
    }
    std::vector<double> gpu_samples;
    std::vector<double> raw_gpu_samples;
    std::vector<double> host_samples;
    std::vector<double> raw_host_samples;
    std::vector<uint32_t> loop_counts;
    gpu_samples.reserve(repeats);
    raw_gpu_samples.reserve(repeats);
    host_samples.reserve(repeats);
    raw_host_samples.reserve(repeats);
    loop_counts.reserve(repeats);
    uint32_t valid_gpu_samples = 0;
    for (uint32_t index = 0; index < repeats; ++index) {
        uint32_t sample_loop_count = loop_count;
        auto sample = submit(sample_loop_count);
        double sampled_region_us = sample.first > 0.0 ? sample.first : sample.second;
        while (sampled_region_us < kSampleRegionTargetUs &&
               sample_loop_count < kMaxLoopCount) {
            sample_loop_count = sample_loop_count > kMaxLoopCount / 2
                ? kMaxLoopCount : sample_loop_count * 2;
            sample = submit(sample_loop_count);
            sampled_region_us = sample.first > 0.0 ? sample.first : sample.second;
        }
        const auto [gpu_us, host_us] = sample;
        loop_count = std::max(loop_count, sample_loop_count);
        loop_counts.push_back(sample_loop_count);
        raw_gpu_samples.push_back(gpu_us);
        raw_host_samples.push_back(host_us);
        gpu_samples.push_back(std::max(1.0e-6, gpu_us / sample_loop_count));
        host_samples.push_back(std::max(1.0e-6, host_us / sample_loop_count));
        valid_gpu_samples += gpu_us > 0.0 ? 1u : 0u;
    }
    const bool complete_gpu_timestamps = valid_gpu_samples == repeats;
    return TimingRecord{
        std::move(operation),
        std::move(metric),
        complete_gpu_timestamps
            ? "metal_command_buffer_gpu_timestamps"
            : "host_steady_clock_command_buffer_sync_fallback",
        "commit_then_waitUntilCompleted_before_timestamp_read",
        complete_gpu_timestamps ? std::move(gpu_samples) : host_samples,
        complete_gpu_timestamps ? std::move(raw_gpu_samples) : raw_host_samples,
        std::move(host_samples),
        std::move(raw_host_samples),
        std::move(loop_counts),
        loop_count,
        valid_gpu_samples,
    };
}

template <typename Fn>
TimingRecord time_canonical_payload(
    std::string operation,
    std::string metric,
    uint32_t warmups,
    uint32_t repeats,
    Fn&& operation_fn
) {
    TimingRecord record = time_host(
        std::move(operation),
        std::move(metric),
        warmups,
        repeats,
        std::forward<Fn>(operation_fn));
    record.clock = "host_steady_clock_canonical_abi1_1";
    record.synchronization =
        "canonical_abi1_1_payload_call_returns_after_device_completion";
    return record;
}

id<MTLComputePipelineState> load_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) fail("missing Metal function: " + std::string(name.UTF8String));
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        fail(
            "failed to build Metal pipeline " + std::string(name.UTF8String) +
            ": " + (error == nil ? "unknown" : error.localizedDescription.UTF8String));
    }
    if (pipeline.maxTotalThreadsPerThreadgroup < kReduceWidth) {
        fail("Metal pipeline cannot dispatch the contracted 64-thread group");
    }
    return pipeline;
}

uint32_t partial_block_count(uint64_t rows, uint32_t top_k) {
    const uint64_t target_blocks = 1 + (rows - 1) / kReduceWidth;
    const uint64_t storage_blocks = 1 + (rows - 1) / top_k;
    return static_cast<uint32_t>(std::min<uint64_t>(
        std::min(target_blocks, storage_blocks), kMaxPartialBlocks));
}

template <typename T>
id<MTLBuffer> buffer_with_vector(
    id<MTLDevice> device,
    const std::vector<T>& values,
    MTLResourceOptions storage
) {
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:values.data()
        length:values.size() * sizeof(T)
        options:storage];
    if (buffer == nil) fail("Metal buffer allocation failed");
    return buffer;
}

id<MTLBuffer> buffer_with_bytes(
    id<MTLDevice> device,
    const void* values,
    size_t length,
    MTLResourceOptions storage
) {
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:values length:length options:storage];
    if (buffer == nil) fail("Metal buffer allocation failed");
    return buffer;
}

id<MTLBuffer> empty_buffer(
    id<MTLDevice> device,
    size_t length,
    MTLResourceOptions storage
) {
    id<MTLBuffer> buffer = [device newBufferWithLength:length options:storage];
    if (buffer == nil) fail("Metal buffer allocation failed");
    return buffer;
}

void mark_modified(id<MTLBuffer> buffer, bool managed) {
    if (managed) [buffer didModifyRange:NSMakeRange(0, buffer.length)];
}

void encode_metric(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> features,
    id<MTLBuffer> target,
    id<MTLBuffer> means,
    id<MTLBuffer> combos,
    id<MTLBuffer> metric_ids,
    id<MTLBuffer> chunks,
    id<MTLBuffer> output,
    id<MTLBuffer> info,
    uint32_t candidates
) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) fail("failed to allocate Metal compute encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:features offset:0 atIndex:0];
    [encoder setBuffer:target offset:0 atIndex:1];
    [encoder setBuffer:means offset:0 atIndex:2];
    [encoder setBuffer:combos offset:0 atIndex:3];
    [encoder setBuffer:metric_ids offset:0 atIndex:4];
    [encoder setBuffer:chunks offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    [encoder setBuffer:info offset:0 atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(candidates, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceWidth, 1, 1)];
    [encoder endEncoding];
}

void write_samples(std::ostream& output, const std::vector<double>& samples) {
    output << '[';
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) output << ", ";
        output << samples[index];
    }
    output << ']';
}

void write_file_identity(
    std::ostream& output,
    const char* label,
    const std::string& path,
    bool trailing_comma
) {
    output << "    \"" << label << "\": {\"path\": \""
           << json_escape(path) << "\", \"size_bytes\": "
           << std::filesystem::file_size(path) << ", \"sha256\": \""
           << sha256_file(path) << "\"}" << (trailing_comma ? "," : "") << "\n";
}

struct CanonicalProtocolFixture {
    std::vector<uint32_t> combos;
    std::vector<uint32_t> metrics;
    GafimeShapeHint shape_hint{};
    GafimeArityChunk chunk{};
    GafimeLaunchProtocol base{};
    GafimeNumericLaunchProtocol numeric{};

    CanonicalProtocolFixture(
        const GafimeNumericRoute& route,
        uint32_t rows,
        uint32_t candidates,
        uint32_t mi_bins,
        uint32_t metric,
        uint32_t top_k
    ) : combos(candidates), metrics{metric} {
        std::iota(combos.begin(), combos.end(), 0u);
        shape_hint.vendor_hint = mi_bins;
        chunk.arity = 1;
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.metric_mask = 0;
        chunk.shape_hint_index = 0;
        chunk.combo_row_offset = 0;
        chunk.combo_count = candidates;
        chunk.local_chunk_id = 0;
        chunk.flags = 0;
        chunk.descriptor_offset = 0;
        chunk.descriptor_count = candidates;
        base.abi_version = GAFIME_ABI_VERSION;
        base.backend_kind = GAFIME_BACKEND_METAL;
        base.max_arity = 1;
        base.n_samples = rows;
        base.n_features = candidates;
        base.family_count = 1;
        base.combo_indices = {combos.data(), combos.size()};
        base.metric_ids = {metrics.data(), metrics.size()};
        base.chunks = &chunk;
        base.chunk_count = 1;
        base.shape_hints = &shape_hint;
        base.shape_hint_count = 1;
        base.rank.top_k = top_k;
        base.rank.primary_metric = metric;
        base.rank.descending = 1;
        base.rank.include_ties = 0;
        numeric.abi_version = GAFIME_PRECISION_ABI_VERSION;
        numeric.struct_size = sizeof(numeric);
        numeric.route = route;
        numeric.base = &base;
    }
};

struct CanonicalResultFixture {
    std::vector<uint32_t> combo_indices;
    std::vector<float> metric_values;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> families;
    std::vector<uint64_t> candidate_ids;
    std::vector<uint32_t> row_flags;
    GafimeNumericResultTable table{};

    CanonicalResultFixture(uint32_t capacity, uint32_t metric_count)
        : combo_indices(capacity),
          metric_values(static_cast<size_t>(capacity) * metric_count),
          ranks(capacity),
          families(capacity),
          candidate_ids(capacity),
          row_flags(capacity) {
        table.abi_version = GAFIME_PRECISION_ABI_VERSION;
        table.struct_size = sizeof(table);
        table.max_arity = 1;
        table.metric_count = metric_count;
        table.capacity = capacity;
        table.combo_indices = combo_indices.data();
        table.ranks = ranks.data();
        table.families = families.data();
        table.candidate_ids = candidate_ids.data();
        table.row_flags = row_flags.data();
        table.metric_values.abi_version = GAFIME_PRECISION_ABI_VERSION;
        table.metric_values.struct_size = sizeof(table.metric_values);
        table.metric_values.dtype = GAFIME_DTYPE_F32;
        table.metric_values.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
        table.metric_values.data = metric_values.data();
        table.metric_values.element_capacity = metric_values.size();
        table.metric_values.byte_length =
            metric_values.size() * sizeof(float);
        table.metric_values.byte_stride = sizeof(float);
    }

    void reset() {
        table.flags = 0;
        table.row_count = 0;
    }
};

GafimeConstBufferView f32_const_buffer(const std::vector<float>& values) {
    GafimeConstBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = values.data();
    view.element_count = values.size();
    view.byte_length = values.size() * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

CanonicalPayloadEvidence run_canonical_payload(
    const Options& options,
    const std::vector<float>& host_features,
    const std::vector<float>& host_target,
    std::vector<TimingRecord>& records,
    double& result_checksum
) {
    CanonicalPayloadEvidence evidence;
    // The payload resolves its shader library at matrix allocation. Point that
    // lookup at the same wheel-extracted metallib passed to the supplemental
    // lane, then call only the installed dylib's ABI 1.1 symbols below.
    ScopedEnvironmentOverride metallib_override(
        "GAFIME_METAL_V1_METALLIB", options.metallib_path);
    CanonicalPayloadApi api(options.payload_path);
    evidence.symbols = {
        "gafime_gpu_numeric_routes_v2",
        "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_v2",
        "gafime_gpu_execute_v2",
        "gafime_gpu_matrix_free_v2",
    };

    uint32_t route_count = 0;
    evidence.route_query_status = api.routes(
        0,
        GAFIME_PRECISION_ABI_VERSION,
        sizeof(GafimeNumericRoute),
        nullptr,
        0,
        &route_count);
    evidence.route_count = route_count;
    if (evidence.route_query_status != GAFIME_STATUS_OK || route_count != 1) {
        fail(
            "exact Metal payload must advertise exactly one canonical fp32 route; "
            "status=" + std::to_string(evidence.route_query_status) +
            " count=" + std::to_string(route_count));
    }
    GafimeNumericRoute route{};
    evidence.route_fill_status = api.routes(
        0,
        GAFIME_PRECISION_ABI_VERSION,
        sizeof(GafimeNumericRoute),
        &route,
        1,
        &route_count);
    if (evidence.route_fill_status != GAFIME_STATUS_OK ||
        route_count != 1 || route.abi_version != GAFIME_PRECISION_ABI_VERSION ||
        route.struct_size != sizeof(route) || route.route_id != GAFIME_NUMERIC_ROUTE_FP32 ||
        route.profile != GAFIME_PRECISION_FP32 ||
        route.storage_dtype != GAFIME_DTYPE_F32 ||
        route.pointwise_dtype != GAFIME_DTYPE_F32 ||
        route.reduction_dtype != GAFIME_DTYPE_F32 ||
        route.result_dtype != GAFIME_DTYPE_F32) {
        fail("exact Metal payload advertised a non-canonical fp32 route");
    }

    GafimeNumericRoute mixed_route = route;
    mixed_route.route_id = GAFIME_NUMERIC_ROUTE_MIXED;
    mixed_route.profile = GAFIME_PRECISION_MIXED;
    mixed_route.reduction_dtype = GAFIME_DTYPE_F64;
    mixed_route.result_dtype = GAFIME_DTYPE_F64;
    GafimeNumericMatrixDesc rejection_desc{};
    rejection_desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
    rejection_desc.struct_size = sizeof(rejection_desc);
    rejection_desc.route = mixed_route;
    rejection_desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    rejection_desc.rows = options.rows;
    rejection_desc.cols = options.candidates;
    rejection_desc.row_stride = options.candidates;
    rejection_desc.bytes = host_features.size() * sizeof(float);
    GafimeGpuMatrix rejected_matrix = nullptr;
    const int mixed_status = api.matrix_alloc(0, &rejection_desc, &rejected_matrix);
    if (mixed_status == GAFIME_STATUS_OK && rejected_matrix != nullptr) {
        (void)api.matrix_free(rejected_matrix);
    }
    evidence.mixed_route_rejected = mixed_status == GAFIME_STATUS_UNSUPPORTED_BACKEND;
    if (!evidence.mixed_route_rejected) {
        fail(
            "Metal payload did not fail closed for the unsupported mixed route: " +
            std::to_string(mixed_status));
    }

    GafimeNumericMatrixDesc matrix_desc{};
    matrix_desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
    matrix_desc.struct_size = sizeof(matrix_desc);
    matrix_desc.route = route;
    matrix_desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    matrix_desc.rows = options.rows;
    matrix_desc.cols = options.candidates;
    matrix_desc.row_stride = options.candidates;
    matrix_desc.bytes = host_features.size() * sizeof(float);

    auto allocate_and_free = [&]() {
        GafimeGpuMatrix temporary = nullptr;
        const int status = api.matrix_alloc(0, &matrix_desc, &temporary);
        if (status != GAFIME_STATUS_OK || temporary == nullptr) {
            fail("canonical Metal matrix allocation failed: " + std::to_string(status));
        }
        const int free_status = api.matrix_free(temporary);
        if (free_status != GAFIME_STATUS_OK) {
            fail("canonical Metal temporary matrix free failed: " +
                 std::to_string(free_status));
        }
    };
    records.push_back(time_canonical_payload(
        "matrix_allocation",
        "none",
        options.warmups,
        options.repeats,
        allocate_and_free));

    GafimeGpuMatrix matrix = nullptr;
    try {
        evidence.matrix_alloc_status = api.matrix_alloc(0, &matrix_desc, &matrix);
        if (evidence.matrix_alloc_status != GAFIME_STATUS_OK || matrix == nullptr) {
            fail("canonical Metal matrix allocation failed: " +
                 std::to_string(evidence.matrix_alloc_status));
        }
        const GafimeConstBufferView features_view = f32_const_buffer(host_features);
        const GafimeConstBufferView target_view = f32_const_buffer(host_target);
        evidence.matrix_upload_status = api.matrix_upload(
            matrix,
            &route,
            &features_view,
            &target_view,
            options.rows,
            options.candidates);
        if (evidence.matrix_upload_status != GAFIME_STATUS_OK) {
            fail("canonical Metal matrix upload failed: " +
                 std::to_string(evidence.matrix_upload_status));
        }
        records.push_back(time_canonical_payload(
            "h2d_upload_or_unified_write",
            "none",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = api.matrix_upload(
                    matrix,
                    &route,
                    &features_view,
                    &target_view,
                    options.rows,
                    options.candidates);
                if (status != GAFIME_STATUS_OK) {
                    fail("canonical Metal repeated upload failed: " +
                         std::to_string(status));
                }
            }));

        volatile uint64_t planning_checksum = 0;
        records.push_back(time_canonical_payload(
            "planning_and_descriptor_materialization",
            "none",
            options.warmups,
            options.repeats,
            [&]() {
                CanonicalProtocolFixture planned(
                    route,
                    options.rows,
                    options.candidates,
                    options.mi_bins,
                    kMetricPearson,
                    0);
                planning_checksum += planned.combos.back() + planned.chunk.combo_count;
            }));

        auto add_metric_record = [&](uint32_t metric, const char* name) {
            CanonicalProtocolFixture protocol(
                route,
                options.rows,
                options.candidates,
                options.mi_bins,
                metric,
                0);
            CanonicalResultFixture result(options.candidates, 1);
            records.push_back(time_canonical_payload(
                "metric_kernel",
                name,
                options.warmups,
                options.repeats,
                [&]() {
                    result.reset();
                    const int status = api.execute(matrix, &protocol.numeric, &result.table);
                    if (status != GAFIME_STATUS_OK ||
                        result.table.row_count != options.candidates) {
                        fail("canonical Metal metric execution failed: " +
                             std::to_string(status));
                    }
                }));
            for (float value : result.metric_values) {
                if (!std::isfinite(value)) fail("canonical Metal metric result is non-finite");
                result_checksum += value;
            }
        };

        add_metric_record(kMetricPearson, "pearson");
        add_metric_record(kMetricR2, "r2");
        add_metric_record(kMetricMutualInfo, "mutual_info");

        // The payload's canonical execute call builds and caches target ranks
        // before Spearman when this bounded workload is cache-eligible.
        CanonicalProtocolFixture target_rank_protocol(
            route,
            options.rows,
            options.candidates,
            options.mi_bins,
            kMetricSpearman,
            0);
        CanonicalResultFixture target_rank_result(options.candidates, 1);
        records.push_back(time_canonical_payload(
            "ranking_kernel",
            "spearman_target_ranks",
            options.warmups,
            options.repeats,
            [&]() {
                target_rank_result.reset();
                const int status = api.execute(
                    matrix, &target_rank_protocol.numeric, &target_rank_result.table);
                if (status != GAFIME_STATUS_OK ||
                    target_rank_result.table.row_count != options.candidates) {
                    fail("canonical Metal target-rank execution failed: " +
                         std::to_string(status));
                }
            }));
        for (float value : target_rank_result.metric_values) {
            if (!std::isfinite(value)) fail("canonical Metal Spearman result is non-finite");
            result_checksum += value;
        }

        add_metric_record(kMetricSpearman, "spearman");

        CanonicalProtocolFixture ranking_protocol(
            route,
            options.rows,
            options.candidates,
            options.mi_bins,
            kMetricSpearman,
            options.top_k);
        CanonicalResultFixture ranking_result(options.top_k, 1);
        records.push_back(time_canonical_payload(
            "ranking_topk_and_gather",
            "spearman",
            options.warmups,
            options.repeats,
            [&]() {
                ranking_result.reset();
                const int status = api.execute(
                    matrix, &ranking_protocol.numeric, &ranking_result.table);
                if (status != GAFIME_STATUS_OK ||
                    ranking_result.table.row_count != options.top_k) {
                    fail("canonical Metal ranked execution failed: " +
                         std::to_string(status));
                }
                for (uint32_t row = 0; row < options.top_k; ++row) {
                    if (ranking_result.combo_indices[row] >= options.candidates ||
                        !std::isfinite(ranking_result.metric_values[row])) {
                        fail("canonical Metal ranked output validation failed");
                    }
                }
            }));
        std::vector<uint32_t> host_selected_indices(options.top_k);
        std::vector<float> host_selected_metrics(options.top_k);
        records.push_back(time_canonical_payload(
            "d2h_result_readback",
            "results",
            options.warmups,
            options.repeats,
            [&]() {
                std::memcpy(
                    host_selected_indices.data(),
                    ranking_result.combo_indices.data(),
                    options.top_k * sizeof(uint32_t));
                std::memcpy(
                    host_selected_metrics.data(),
                    ranking_result.metric_values.data(),
                    options.top_k * sizeof(float));
            }));
        volatile double report_checksum = 0.0;
        records.push_back(time_canonical_payload(
            "report_construction",
            "results",
            options.warmups,
            options.repeats,
            [&]() {
                std::vector<std::pair<uint32_t, float>> report;
                report.reserve(options.top_k);
                for (uint32_t row = 0; row < options.top_k; ++row) {
                    report.emplace_back(host_selected_indices[row], host_selected_metrics[row]);
                }
                report_checksum += report.front().first + report.front().second;
            }));
        result_checksum += static_cast<double>(planning_checksum) + report_checksum;

        evidence.matrix_free_status = api.matrix_free(matrix);
        matrix = nullptr;
        if (evidence.matrix_free_status != GAFIME_STATUS_OK) {
            fail("canonical Metal matrix free failed: " +
                 std::to_string(evidence.matrix_free_status));
        }
    } catch (...) {
        if (matrix != nullptr) (void)api.matrix_free(matrix);
        throw;
    }
    evidence.validated = true;
    return evidence;
}

void write_timing_records(
    std::ostream& output,
    const std::vector<TimingRecord>& records
) {
    for (size_t index = 0; index < records.size(); ++index) {
        const TimingRecord& record = records[index];
        const auto ci = bootstrap_median_ci(record.samples_us, stable_seed(record));
        const auto& raw_samples = record.raw_samples_us.empty()
            ? record.samples_us : record.raw_samples_us;
        const auto& raw_host_samples = record.raw_host_synchronized_samples_us.empty()
            ? record.host_synchronized_samples_us : record.raw_host_synchronized_samples_us;
        output << "    {\"operation\": \"" << record.operation
               << "\", \"metric\": \"" << record.metric
               << "\", \"clock\": \"" << record.clock
               << "\", \"synchronization\": \"" << record.synchronization
               << "\", \"gpu_timestamp_valid_samples\": "
               << record.gpu_timestamp_valid_samples << ", \"samples_us\": ";
        write_samples(output, record.samples_us);
        output << ", \"host_synchronized_samples_us\": ";
        write_samples(output, record.host_synchronized_samples_us);
        output << ", \"raw_samples_us\": ";
        write_samples(output, raw_samples);
        output << ", \"raw_host_synchronized_samples_us\": ";
        write_samples(output, raw_host_samples);
        output << ", \"median_us\": " << median(record.samples_us)
               << ", \"mad_us\": " << median_absolute_deviation(record.samples_us)
               << ", \"p05_us\": " << percentile(record.samples_us, 0.05)
               << ", \"p95_us\": " << percentile(record.samples_us, 0.95)
               << ", \"bootstrap_median_95_ci_us\": [" << ci[0] << ", " << ci[1] << "]"
               << ", \"mean_us\": " << mean(record.samples_us)
               << ", \"min_us\": "
               << *std::min_element(record.samples_us.begin(), record.samples_us.end())
               << ", \"max_us\": "
               << *std::max_element(record.samples_us.begin(), record.samples_us.end())
               << ", \"sample_count\": " << record.samples_us.size()
               << ", \"loop_count_per_sample\": " << record.loop_count_per_sample
               << ", \"sample_region_target_us\": " << kSampleRegionTargetUs
               << ", \"sample_region_min_observed_us\": "
               << *std::min_element(raw_samples.begin(), raw_samples.end())
               << ", \"sample_region_target_met\": "
               << (*std::min_element(raw_samples.begin(), raw_samples.end()) >=
                       kSampleRegionTargetUs ? "true" : "false")
               << ", \"loop_count_per_sample\": " << record.loop_count_per_sample
               << ", \"loop_counts_per_sample\": [";
        for (size_t sample = 0; sample < record.loop_counts_per_sample.size(); ++sample) {
            if (sample != 0) output << ", ";
            output << record.loop_counts_per_sample[sample];
        }
        output << ']'
               << ", \"bootstrap_resamples\": " << kBootstrapResamples
               << ", \"bootstrap_seed\": " << stable_seed(record)
               << "}" << (index + 1 == records.size() ? "\n" : ",\n");
    }
}

void validate_records(const std::vector<TimingRecord>& records, uint32_t repeats) {
    if (records.empty()) fail("no Metal timing records produced");
    for (const TimingRecord& record : records) {
        if (record.samples_us.size() != repeats ||
            record.raw_samples_us.size() != repeats ||
            record.loop_counts_per_sample.size() != repeats) {
            fail("Metal timing record does not contain the requested raw sample count: " +
                 record.operation);
        }
        if ((!record.host_synchronized_samples_us.empty() &&
             record.host_synchronized_samples_us.size() != repeats) ||
            (!record.raw_host_synchronized_samples_us.empty() &&
             record.raw_host_synchronized_samples_us.size() != repeats)) {
            fail("Metal host timing record has an invalid synchronized sample count: " +
                 record.operation);
        }
        for (size_t index = 0; index < repeats; ++index) {
            if (!std::isfinite(record.samples_us[index]) || record.samples_us[index] <= 0.0 ||
                !std::isfinite(record.raw_samples_us[index]) ||
                record.raw_samples_us[index] < kSampleRegionTargetUs ||
                record.loop_counts_per_sample[index] == 0) {
                fail("Metal timing record has an invalid calibrated sample: " + record.operation);
            }
        }
    }
}

void write_json(
    const Options& options,
    int argc,
    char** argv,
    id<MTLDevice> device,
    const std::string& binary_path,
    const std::string& source_path,
    bool unified_memory,
    const std::vector<TimingRecord>& records,
    const std::vector<TimingRecord>& canonical_payload_records,
    const CanonicalPayloadEvidence& canonical_payload,
    double canonical_result_checksum,
    double result_checksum
) {
    const bool gpu_timing_supported = std::all_of(
        records.begin(), records.end(), [&](const TimingRecord& record) {
            return record.clock != "host_steady_clock_command_buffer_sync_fallback";
        });
    std::ostringstream output;
    output << std::setprecision(12);
    output << "{\n"
           << "  \"schema\": \"gafime.metal.native_timing.v1\",\n"
           << "  \"status\": \"pass\",\n"
           << "  \"backend\": \"metal\",\n"
           << "  \"profile\": \"fp32\",\n"
           << "  \"source_commit\": \"" << options.source_commit << "\",\n"
           << "  \"precision_domains\": {\"storage\": \"fp32\", "
              "\"pointwise\": \"fp32\", \"reduction\": \"fp32\", "
              "\"result\": \"fp32\"},\n"
           << "  \"gpu_timing_supported\": "
           << (gpu_timing_supported ? "true" : "false") << ",\n"
           << "  \"timing_contract\": \"GPUStartTime/GPUEndTime are read only "
              "after commit and waitUntilCompleted; synchronized host wall time is "
              "retained separately\",\n"
           << "  \"device\": {\"name\": \""
           << json_escape(device.name.UTF8String) << "\", \"registry_id\": "
           << static_cast<unsigned long long>(device.registryID)
           << ", \"has_unified_memory\": "
           << (unified_memory ? "true" : "false")
           << ", \"recommended_working_set_bytes\": "
           << static_cast<unsigned long long>(device.recommendedMaxWorkingSetSize)
           << ", \"max_threadgroup_memory_bytes\": "
           << static_cast<unsigned long long>(device.maxThreadgroupMemoryLength)
           << "},\n"
           << "  \"compiler\": {\"clang\": \"" << json_escape(__clang_version__)
           << "\", \"cplusplus\": " << __cplusplus << "},\n"
           << "  \"os_version\": \""
           << json_escape(NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String)
           << "\",\n"
           << "  \"command_line\": [";
    for (int index = 0; index < argc; ++index) {
        if (index != 0) output << ", ";
        output << '"' << json_escape(argv[index]) << '"';
    }
    output << "],\n"
           << "  \"workload\": {\"rows\": " << options.rows
           << ", \"candidates\": " << options.candidates
           << ", \"arity\": 1, \"mi_bins\": " << options.mi_bins
           << ", \"top_k\": " << options.top_k
           << ", \"input_elements\": "
           << static_cast<uint64_t>(options.rows) * options.candidates + options.rows
           << ", \"output_rows\": " << options.top_k << "},\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"repeats\": " << options.repeats << ",\n"
           << "  \"sample_region_target_us\": " << kSampleRegionTargetUs << ",\n"
           << "  \"sample_region_calibration_target_us\": "
           << kSampleRegionCalibrationTargetUs << ",\n"
           << "  \"bootstrap_resamples\": " << kBootstrapResamples << ",\n"
           << "  \"bootstrap_seed\": " << kBootstrapSeed << ",\n"
           << "  \"result_checksum\": " << result_checksum << ",\n"
           << "  \"execution_mode\": \"supplemental_internal_kernel\",\n"
           << "  \"decomposition_boundaries\": {\n"
           << "    \"ingest_conversion\": \"not present: native fp32 source policy\",\n"
           << "    \"candidate_materialization\": \"fused into each metric kernel\",\n"
           << "    \"planning\": \"host descriptor construction\",\n"
           << "    \"ranking\": \"partial selection plus merge plus gather in one "
              "synchronized command buffer\",\n"
           << "    \"d2h\": \"managed-resource synchronization when required, then "
              "host-visible result copy\"\n"
           << "  },\n"
           << "  \"provenance\": {\n";
    write_file_identity(output, "benchmark_source", source_path, true);
    write_file_identity(output, "shader_source", options.shader_source_path, true);
    write_file_identity(output, "benchmark_binary", binary_path, true);
    write_file_identity(output, "metallib", options.metallib_path, true);
    write_file_identity(output, "payload", options.payload_path, true);
    write_file_identity(output, "wheel", options.wheel_path, false);
    output << "  },\n"
           << "  \"records\": [\n";
    write_timing_records(output, records);
    output << "  ],\n"
           << "  \"canonical_payload_lifecycle\": {\n"
           << "    \"status\": \""
           << (canonical_payload.validated ? "validated" : "invalid") << "\",\n"
           << "    \"schema\": \"gafime.native-decomposition.v1\",\n"
           << "    \"execution_layer\": \"installed_payload_dylib\",\n"
           << "    \"abi\": \"canonical_1.1\",\n"
           << "    \"route_count\": " << canonical_payload.route_count << ",\n"
           << "    \"route_query_status\": " << canonical_payload.route_query_status << ",\n"
           << "    \"route_fill_status\": " << canonical_payload.route_fill_status << ",\n"
           << "    \"matrix_alloc_status\": " << canonical_payload.matrix_alloc_status << ",\n"
           << "    \"matrix_upload_status\": " << canonical_payload.matrix_upload_status << ",\n"
           << "    \"matrix_free_status\": " << canonical_payload.matrix_free_status << ",\n"
           << "    \"mixed_route_rejected\": "
           << (canonical_payload.mixed_route_rejected ? "true" : "false") << ",\n"
           << "    \"symbols\": [";
    for (size_t index = 0; index < canonical_payload.symbols.size(); ++index) {
        if (index != 0) output << ", ";
        output << '"' << json_escape(canonical_payload.symbols[index]) << '"';
    }
    output << "],\n"
           << "    \"records_field\": \"canonical_payload_records\",\n"
           << "    \"result_checksum\": " << canonical_result_checksum << "\n"
           << "  },\n"
           << "  \"canonical_payload_records\": [\n";
    write_timing_records(output, canonical_payload_records);
    output << "  ]\n}\n";
    std::filesystem::create_directories(
        std::filesystem::path(options.json_path).parent_path());
    std::ofstream file(options.json_path);
    if (!file) fail("cannot open JSON output: " + options.json_path);
    file << output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        @autoreleasepool {
            const Options options = parse_options(argc, argv);
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) fail("no default Metal device found");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) fail("failed to create Metal command queue");

            NSError* error = nil;
            NSURL* library_url = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:options.metallib_path.c_str()]];
            id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
            if (library == nil) {
                fail(
                    "failed to load metallib: " +
                    std::string(error == nil ? "unknown" : error.localizedDescription.UTF8String));
            }
            id<MTLComputePipelineState> continuous = load_pipeline(
                device, library, @"gafime_score_continuous");
            id<MTLComputePipelineState> mutual_info = load_pipeline(
                device, library, @"gafime_score_mutual_info");
            id<MTLComputePipelineState> target_ranks = load_pipeline(
                device, library, @"gafime_build_spearman_target_ranks");
            id<MTLComputePipelineState> spearman = load_pipeline(
                device, library, @"gafime_score_spearman");
            id<MTLComputePipelineState> partial_topk = load_pipeline(
                device, library, @"gafime_select_topk_partials_desc");
            id<MTLComputePipelineState> merge_topk = load_pipeline(
                device, library, @"gafime_merge_topk_partials_desc");
            id<MTLComputePipelineState> gather = load_pipeline(
                device, library, @"gafime_copy_selected_metric_rows");

            const bool unified_memory = device.hasUnifiedMemory;
            const MTLResourceOptions host_storage = unified_memory
                ? MTLResourceStorageModeShared
                : MTLResourceStorageModeManaged;
            std::vector<float> host_features(
                static_cast<size_t>(options.rows) * options.candidates);
            // The supplemental shader lane consumes feature-major storage,
            // while canonical ABI 1.1 matrix upload is intentionally row-major.
            std::vector<float> canonical_features(
                static_cast<size_t>(options.rows) * options.candidates);
            std::vector<float> host_target(options.rows);
            std::vector<float> host_means(options.candidates, 0.0f);
            std::vector<uint32_t> host_combos(options.candidates);
            std::iota(host_combos.begin(), host_combos.end(), 0u);
            for (uint32_t row = 0; row < options.rows; ++row) {
                host_target[row] = 0.25f + static_cast<float>(
                    (static_cast<uint64_t>(row) * 104729u + row / 7u + 17u) %
                    100003u) / 100003.0f;
                for (uint32_t column = 0; column < options.candidates; ++column) {
                    const float value = 0.1f + static_cast<float>(
                        (static_cast<uint64_t>(row) * (8191u + 2u * column) +
                         row / (3u + column % 11u) + 97u * column) %
                        100019u) / 100019.0f;
                    host_features[static_cast<size_t>(column) * options.rows + row] =
                        value;
                    canonical_features[
                        static_cast<size_t>(row) * options.candidates + column] = value;
                    host_means[column] += value;
                }
            }
            for (float& value : host_means) value /= static_cast<float>(options.rows);

            const MetalChunk chunk{
                1,
                options.mi_bins,
                0,
                0,
                0,
                options.candidates,
                0,
            };
            const MetalLaunchInfo info{
                options.rows,
                options.candidates,
                1,
                1,
                kPrecisionFp32,
            };
            const uint32_t rank_blocks =
                partial_block_count(options.candidates, options.top_k);
            const MetalRankInfo rank_info{
                options.candidates,
                1,
                0,
                options.top_k,
                rank_blocks,
            };

            id<MTLBuffer> features = buffer_with_vector(device, host_features, host_storage);
            id<MTLBuffer> target = buffer_with_vector(device, host_target, host_storage);
            id<MTLBuffer> means = buffer_with_vector(device, host_means, host_storage);
            id<MTLBuffer> combos = buffer_with_vector(device, host_combos, host_storage);
            id<MTLBuffer> chunks = buffer_with_bytes(
                device, &chunk, sizeof(chunk), host_storage);
            id<MTLBuffer> info_buffer = buffer_with_bytes(
                device, &info, sizeof(info), host_storage);
            id<MTLBuffer> rank_info_buffer = buffer_with_bytes(
                device, &rank_info, sizeof(rank_info), host_storage);
            const std::array<uint32_t, 4> metric_values = {
                kMetricPearson,
                kMetricR2,
                kMetricMutualInfo,
                kMetricSpearman,
            };
            std::array<id<MTLBuffer>, 4> metric_ids{};
            for (size_t index = 0; index < metric_ids.size(); ++index) {
                metric_ids[index] = buffer_with_bytes(
                    device, &metric_values[index], sizeof(uint32_t), host_storage);
            }
            id<MTLBuffer> output = empty_buffer(
                device, options.candidates * sizeof(float), host_storage);
            id<MTLBuffer> target_rank_buffer = empty_buffer(
                device, options.rows * sizeof(uint32_t), host_storage);
            id<MTLBuffer> partial_scores = empty_buffer(
                device,
                static_cast<size_t>(rank_blocks) * options.top_k * sizeof(float),
                MTLResourceStorageModePrivate);
            id<MTLBuffer> partial_indices = empty_buffer(
                device,
                static_cast<size_t>(rank_blocks) * options.top_k * sizeof(uint32_t),
                MTLResourceStorageModePrivate);
            id<MTLBuffer> selected_indices = empty_buffer(
                device, options.top_k * sizeof(uint32_t), host_storage);
            id<MTLBuffer> selected_metrics = empty_buffer(
                device, options.top_k * sizeof(float), host_storage);

            for (id<MTLBuffer> buffer in @[
                     features,
                     target,
                     means,
                     combos,
                     chunks,
                     info_buffer,
                     rank_info_buffer,
                     metric_ids[0],
                     metric_ids[1],
                     metric_ids[2],
                     metric_ids[3],
                 ]) {
                mark_modified(buffer, !unified_memory);
            }

            std::vector<TimingRecord> records;
            records.push_back(time_host(
                "matrix_allocation",
                "none",
                options.warmups,
                options.repeats,
                [&]() {
                    @autoreleasepool {
                        id<MTLBuffer> temporary_features = empty_buffer(
                            device, host_features.size() * sizeof(float), host_storage);
                        id<MTLBuffer> temporary_target = empty_buffer(
                            device, host_target.size() * sizeof(float), host_storage);
                        id<MTLBuffer> temporary_output = empty_buffer(
                            device, options.candidates * sizeof(float), host_storage);
                        (void)temporary_features;
                        (void)temporary_target;
                        (void)temporary_output;
                    }
                }));
            records.push_back(time_host(
                "h2d_upload_or_unified_write",
                "none",
                options.warmups,
                options.repeats,
                [&]() {
                    std::memcpy(
                        features.contents,
                        host_features.data(),
                        host_features.size() * sizeof(float));
                    std::memcpy(
                        target.contents,
                        host_target.data(),
                        host_target.size() * sizeof(float));
                    mark_modified(features, !unified_memory);
                    mark_modified(target, !unified_memory);
                }));
            volatile uint64_t planning_checksum = 0;
            records.push_back(time_host(
                "planning_and_descriptor_materialization",
                "none",
                options.warmups,
                options.repeats,
                [&]() {
                    std::vector<uint32_t> planned(options.candidates);
                    std::iota(planned.begin(), planned.end(), 0u);
                    MetalChunk planned_chunk = chunk;
                    planned_chunk.combo_count = planned.size();
                    planning_checksum = planning_checksum + planned.back() +
                        planned_chunk.combo_count;
                }));

            auto add_metric_record = [&](size_t metric_index, const char* metric_name,
                                         id<MTLComputePipelineState> pipeline) {
                records.push_back(time_command_buffers(
                    queue,
                    "metric_kernel",
                    metric_name,
                    options.warmups,
                    options.repeats,
                    [&](id<MTLCommandBuffer> command_buffer) {
                        encode_metric(
                            command_buffer,
                            pipeline,
                            features,
                            target,
                            means,
                            combos,
                            metric_ids[metric_index],
                            chunks,
                            output,
                            info_buffer,
                            options.candidates);
                    }));
            };
            add_metric_record(0, "pearson", continuous);
            add_metric_record(1, "r2", continuous);
            add_metric_record(2, "mutual_info", mutual_info);

            records.push_back(time_command_buffers(
                queue,
                "ranking_kernel",
                "spearman_target_ranks",
                options.warmups,
                options.repeats,
                [&](id<MTLCommandBuffer> command_buffer) {
                    id<MTLComputeCommandEncoder> encoder =
                        [command_buffer computeCommandEncoder];
                    if (encoder == nil) fail("failed to allocate rank encoder");
                    [encoder setComputePipelineState:target_ranks];
                    [encoder setBuffer:target offset:0 atIndex:0];
                    [encoder setBuffer:target_rank_buffer offset:0 atIndex:1];
                    [encoder setBuffer:info_buffer offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(
                        1 + (options.rows - 1) / kReduceWidth, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kReduceWidth, 1, 1)];
                    [encoder endEncoding];
                }));
            // Spearman's two extra buffers require one explicitly encoded,
            // cached-rank pass instead of the generic eight-buffer helper.
            records.push_back(time_command_buffers(
                queue,
                "metric_kernel",
                "spearman",
                options.warmups,
                options.repeats,
                [&](id<MTLCommandBuffer> command_buffer) {
                    id<MTLComputeCommandEncoder> encoder =
                        [command_buffer computeCommandEncoder];
                    if (encoder == nil) fail("failed to allocate Spearman encoder");
                    const uint32_t use_cached_ranks = 1;
                    [encoder setComputePipelineState:spearman];
                    [encoder setBuffer:features offset:0 atIndex:0];
                    [encoder setBuffer:target offset:0 atIndex:1];
                    [encoder setBuffer:means offset:0 atIndex:2];
                    [encoder setBuffer:combos offset:0 atIndex:3];
                    [encoder setBuffer:metric_ids[3] offset:0 atIndex:4];
                    [encoder setBuffer:chunks offset:0 atIndex:5];
                    [encoder setBuffer:output offset:0 atIndex:6];
                    [encoder setBuffer:info_buffer offset:0 atIndex:7];
                    [encoder setBuffer:target_rank_buffer offset:0 atIndex:8];
                    [encoder setBytes:&use_cached_ranks
                        length:sizeof(use_cached_ranks)
                        atIndex:9];
                    [encoder dispatchThreadgroups:MTLSizeMake(
                        options.candidates, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(kReduceWidth, 1, 1)];
                    [encoder endEncoding];
                }));

            records.push_back(time_command_buffers(
                queue,
                "ranking_topk_and_gather",
                "spearman",
                options.warmups,
                options.repeats,
                [&](id<MTLCommandBuffer> command_buffer) {
                    id<MTLComputeCommandEncoder> encoder =
                        [command_buffer computeCommandEncoder];
                    if (encoder == nil) fail("failed to allocate top-k encoder");
                    const MTLSize group = MTLSizeMake(kReduceWidth, 1, 1);
                    [encoder setComputePipelineState:partial_topk];
                    [encoder setBuffer:output offset:0 atIndex:0];
                    [encoder setBuffer:partial_scores offset:0 atIndex:1];
                    [encoder setBuffer:partial_indices offset:0 atIndex:2];
                    [encoder setBuffer:rank_info_buffer offset:0 atIndex:3];
                    [encoder dispatchThreadgroups:MTLSizeMake(rank_blocks, 1, 1)
                        threadsPerThreadgroup:group];
                    [encoder setComputePipelineState:merge_topk];
                    [encoder setBuffer:partial_scores offset:0 atIndex:0];
                    [encoder setBuffer:partial_indices offset:0 atIndex:1];
                    [encoder setBuffer:selected_indices offset:0 atIndex:2];
                    [encoder setBuffer:rank_info_buffer offset:0 atIndex:3];
                    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:group];
                    [encoder setComputePipelineState:gather];
                    [encoder setBuffer:output offset:0 atIndex:0];
                    [encoder setBuffer:selected_indices offset:0 atIndex:1];
                    [encoder setBuffer:selected_metrics offset:0 atIndex:2];
                    [encoder setBuffer:rank_info_buffer offset:0 atIndex:3];
                    [encoder dispatchThreadgroups:MTLSizeMake(
                        1 + (options.top_k - 1) / kReduceWidth, 1, 1)
                        threadsPerThreadgroup:group];
                    [encoder endEncoding];
                }));

            if (!unified_memory) {
                records.push_back(time_command_buffers(
                    queue,
                    "d2h_managed_resource_synchronize",
                    "results",
                    options.warmups,
                    options.repeats,
                    [&](id<MTLCommandBuffer> command_buffer) {
                        id<MTLBlitCommandEncoder> blit =
                            [command_buffer blitCommandEncoder];
                        if (blit == nil) fail("failed to allocate D2H blit encoder");
                        [blit synchronizeResource:selected_indices];
                        [blit synchronizeResource:selected_metrics];
                        [blit endEncoding];
                    }));
            }

            std::vector<uint32_t> host_selected_indices(options.top_k);
            std::vector<float> host_selected_metrics(options.top_k);
            records.push_back(time_host(
                "d2h_result_readback",
                "results",
                options.warmups,
                options.repeats,
                [&]() {
                    std::memcpy(
                        host_selected_indices.data(),
                        selected_indices.contents,
                        options.top_k * sizeof(uint32_t));
                    std::memcpy(
                        host_selected_metrics.data(),
                        selected_metrics.contents,
                        options.top_k * sizeof(float));
                }));
            volatile double report_checksum = 0.0;
            records.push_back(time_host(
                "report_construction",
                "results",
                options.warmups,
                options.repeats,
                [&]() {
                    std::vector<std::pair<uint32_t, float>> report;
                    report.reserve(options.top_k);
                    for (uint32_t index = 0; index < options.top_k; ++index) {
                        report.emplace_back(
                            host_selected_indices[index], host_selected_metrics[index]);
                    }
                    report_checksum = report_checksum + report.front().first +
                        report.front().second;
                }));
            for (uint32_t index = 0; index < options.top_k; ++index) {
                if (host_selected_indices[index] >= options.candidates ||
                    !std::isfinite(host_selected_metrics[index])) {
                    fail("Metal top-k output validation failed");
                }
            }

            std::vector<TimingRecord> canonical_payload_records;
            double canonical_result_checksum = 0.0;
            const CanonicalPayloadEvidence canonical_payload = run_canonical_payload(
                options,
                canonical_features,
                host_target,
                canonical_payload_records,
                canonical_result_checksum);
            validate_records(records, options.repeats);
            validate_records(canonical_payload_records, options.repeats);

            const std::string source_path = canonical_path(__FILE__);
            const std::string binary_path = canonical_path(argv[0]);
            write_json(
                options,
                argc,
                argv,
                device,
                binary_path,
                source_path,
                unified_memory,
                records,
                canonical_payload_records,
                canonical_payload,
                canonical_result_checksum,
                static_cast<double>(planning_checksum) + report_checksum);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metal_precision_native_timing: " << error.what() << '\n';
        return 1;
    }
}
