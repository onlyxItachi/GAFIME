#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
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
#include <unistd.h>

extern char** environ;

#include "../../src/common/gafime_gpu_abi.hpp"
#include "trusted_git.hpp"

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
constexpr uint32_t kCalibrationConfirmationSamples = 3;
constexpr uint32_t kMaxLoopCount = 1u << 24;
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
    std::string source_root;
    std::string source_commit;
    std::string harness_source_root;
    std::string harness_source_commit;
    std::string input_policy = "native";
    std::string variant;
    uint32_t ab_block = 0;
    bool ab_block_set = false;
    std::array<std::string, 2> variant_sequence;
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
    std::vector<double> gpu_timestamp_samples_us;
    std::vector<double> raw_gpu_timestamp_samples_us;
};

[[noreturn]] void fail(const std::string& message);

// Private mirrors of the pre-freeze typed ABI 1.1 prefix at the historical
// PR #70 baseline. The current public header intentionally contains only the
// generic numeric-route ABI; these layouts let one current helper test both
// payload generations without compiling a different harness for each side.
struct FrozenPrecisionMatrixDesc {
    uint32_t abi_version;
    uint32_t profile;
    uint32_t dtype;
    uint32_t layout;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
    uint64_t reserved[8];
};

struct FrozenPrecisionCapabilities {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t profile_mask;
    uint32_t storage_dtype_mask;
    uint32_t result_dtype_mask;
    uint32_t flags;
    uint64_t reserved[8];
};

struct FrozenPrecisionLaunchProtocol {
    uint32_t abi_version;
    uint32_t profile;
    const GafimeLaunchProtocol* base;
    uint64_t reserved[8];
};

static_assert(sizeof(FrozenPrecisionMatrixDesc) == 112);
static_assert(offsetof(FrozenPrecisionMatrixDesc, rows) == 24);
static_assert(offsetof(FrozenPrecisionMatrixDesc, bytes) == 40);
static_assert(sizeof(FrozenPrecisionCapabilities) == 88);
static_assert(offsetof(FrozenPrecisionCapabilities, reserved) == 24);
static_assert(sizeof(FrozenPrecisionLaunchProtocol) == 80);
static_assert(offsetof(FrozenPrecisionLaunchProtocol, base) == 8);

enum class PayloadAbiSurface {
    GenericNumericRouteV2,
    TypedPrecisionV1_1,
};

constexpr std::array<const char*, 10> kGenericPayloadSymbols = {
    "gafime_gpu_numeric_routes_v2",
    "gafime_gpu_matrix_alloc_v2",
    "gafime_gpu_matrix_upload_v2",
    "gafime_gpu_matrix_update_target_v2",
    "gafime_gpu_execute_v2",
    "gafime_gpu_execution_memory_peak_v2",
    "gafime_gpu_permutation_memory_peak_v2",
    "gafime_gpu_permutation_pvalues_v2",
    "gafime_gpu_interaction_diagnostics_v2",
    "gafime_gpu_matrix_free_v2",
};

constexpr std::array<const char*, 11> kTypedPayloadSymbols = {
    "gafime_gpu_precision_capabilities",
    "gafime_gpu_matrix_alloc_v2",
    "gafime_gpu_matrix_upload_f32_v2",
    "gafime_gpu_matrix_upload_f64_v2",
    "gafime_gpu_matrix_update_target_f32_v2",
    "gafime_gpu_matrix_update_target_f64_v2",
    "gafime_gpu_execute_f32_v2",
    "gafime_gpu_execute_f64_v2",
    "gafime_gpu_execution_memory_peak_v2",
    "gafime_gpu_interaction_diagnostics",
    "gafime_gpu_matrix_free",
};

// The benchmark binary is deliberately not linked against either payload. It
// detects one complete ABI surface and rejects partial or ambiguous ownership.
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
    using MatrixUpdateTargetFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericRoute*,
        const GafimeConstBufferView*,
        uint64_t
    );
    using ExecutionMemoryFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericLaunchProtocol*,
        uint64_t*
    );
    using PermutationMemoryFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericLaunchProtocol*,
        uint64_t,
        uint64_t*
    );
    using PermutationPvaluesFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericLaunchProtocol*,
        GafimeNumericSignificanceTable*
    );
    using DiagnosticsFn = int (*)(
        GafimeGpuMatrix,
        GafimeNumericInteractionDiagnosticBatch*
    );
    using ExecuteFn = int (*)(
        GafimeGpuMatrix,
        const GafimeNumericLaunchProtocol*,
        GafimeNumericResultTable*
    );
    using MatrixFreeFn = int (*)(GafimeGpuMatrix);
    using TypedCapabilitiesFn = int (*)(uint32_t, FrozenPrecisionCapabilities*);
    using TypedMatrixAllocFn = int (*)(
        uint32_t,
        const FrozenPrecisionMatrixDesc*,
        GafimeGpuMatrix*
    );
    using TypedUploadF32Fn = int (*)(
        GafimeGpuMatrix,
        const float*,
        const float*,
        uint64_t,
        uint32_t
    );
    using TypedUploadF64Fn = int (*)(
        GafimeGpuMatrix,
        const double*,
        const double*,
        uint64_t,
        uint32_t
    );
    using TypedUpdateF32Fn = int (*)(GafimeGpuMatrix, const float*, uint64_t);
    using TypedUpdateF64Fn = int (*)(GafimeGpuMatrix, const double*, uint64_t);
    using TypedExecuteF32Fn = int (*)(
        GafimeGpuMatrix,
        const FrozenPrecisionLaunchProtocol*,
        GafimeResultTable*
    );
    using TypedExecuteF64Fn = int (*)(
        GafimeGpuMatrix,
        const FrozenPrecisionLaunchProtocol*,
        void*
    );
    using TypedExecutionMemoryFn = int (*)(
        GafimeGpuMatrix,
        const FrozenPrecisionLaunchProtocol*,
        uint64_t*
    );
    using TypedPermutationMemoryFn = int (*)(
        GafimeGpuMatrix,
        const FrozenPrecisionLaunchProtocol*,
        uint64_t,
        uint64_t*
    );
    using TypedPermutationF32Fn = int (*)(
        GafimeGpuMatrix,
        const FrozenPrecisionLaunchProtocol*,
        GafimePermutationSignificanceTable*
    );
    using TypedPermutationF64Fn = int (*)(
        GafimeGpuMatrix,
        const FrozenPrecisionLaunchProtocol*,
        void*
    );
    using TypedDiagnosticsFn = int (*)(GafimeGpuMatrix, void*);
    using TypedMatrixFreeFn = void (*)(GafimeGpuMatrix);

    void* handle = nullptr;
    PayloadAbiSurface surface = PayloadAbiSurface::GenericNumericRouteV2;
    RoutesFn routes = nullptr;
    MatrixAllocFn matrix_alloc = nullptr;
    MatrixUploadFn matrix_upload = nullptr;
    MatrixUpdateTargetFn matrix_update_target = nullptr;
    ExecutionMemoryFn execution_memory = nullptr;
    PermutationMemoryFn permutation_memory = nullptr;
    PermutationPvaluesFn permutation_pvalues = nullptr;
    DiagnosticsFn diagnostics = nullptr;
    ExecuteFn execute = nullptr;
    MatrixFreeFn matrix_free = nullptr;
    TypedCapabilitiesFn typed_capabilities = nullptr;
    TypedMatrixAllocFn typed_matrix_alloc = nullptr;
    TypedUploadF32Fn typed_upload_f32 = nullptr;
    TypedUploadF64Fn typed_upload_f64 = nullptr;
    TypedUpdateF32Fn typed_update_f32 = nullptr;
    TypedUpdateF64Fn typed_update_f64 = nullptr;
    TypedExecuteF32Fn typed_execute_f32 = nullptr;
    TypedExecuteF64Fn typed_execute_f64 = nullptr;
    TypedExecutionMemoryFn typed_execution_memory = nullptr;
    TypedPermutationMemoryFn typed_permutation_memory = nullptr;
    TypedPermutationF32Fn typed_permutation_f32 = nullptr;
    TypedPermutationF64Fn typed_permutation_f64 = nullptr;
    TypedDiagnosticsFn typed_diagnostics = nullptr;
    TypedMatrixFreeFn typed_matrix_free = nullptr;
    std::vector<std::string> symbols;
    std::vector<std::string> optional_symbols;

    explicit CanonicalPayloadApi(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* detail = dlerror();
            fail(
                "failed to load exact Metal payload dylib: " +
                std::string(detail == nullptr ? "unknown dynamic-loader error" : detail));
        }
        routes = try_load<RoutesFn>("gafime_gpu_numeric_routes_v2");
        if (routes != nullptr) {
            surface = PayloadAbiSurface::GenericNumericRouteV2;
            for (const char* symbol : kGenericPayloadSymbols) {
                (void)load<void*>(symbol);
                symbols.emplace_back(symbol);
            }
            matrix_alloc = load<MatrixAllocFn>("gafime_gpu_matrix_alloc_v2");
            matrix_upload = load<MatrixUploadFn>("gafime_gpu_matrix_upload_v2");
            matrix_update_target = load<MatrixUpdateTargetFn>(
                "gafime_gpu_matrix_update_target_v2");
            execute = load<ExecuteFn>("gafime_gpu_execute_v2");
            execution_memory = load<ExecutionMemoryFn>(
                "gafime_gpu_execution_memory_peak_v2");
            permutation_memory = load<PermutationMemoryFn>(
                "gafime_gpu_permutation_memory_peak_v2");
            permutation_pvalues = load<PermutationPvaluesFn>(
                "gafime_gpu_permutation_pvalues_v2");
            diagnostics = load<DiagnosticsFn>(
                "gafime_gpu_interaction_diagnostics_v2");
            matrix_free = load<MatrixFreeFn>("gafime_gpu_matrix_free_v2");
            return;
        }
        if (try_load<void*>("gafime_gpu_precision_capabilities") == nullptr) {
            fail(
                "exact Metal payload exports neither the generic numeric-route "
                "nor historical pre-freeze typed precision ABI 1.1 surface");
        }
        surface = PayloadAbiSurface::TypedPrecisionV1_1;
        for (const char* symbol : kTypedPayloadSymbols) {
            (void)load<void*>(symbol);
            symbols.emplace_back(symbol);
        }
        typed_capabilities = load<TypedCapabilitiesFn>(
            "gafime_gpu_precision_capabilities");
        typed_matrix_alloc = load<TypedMatrixAllocFn>("gafime_gpu_matrix_alloc_v2");
        typed_upload_f32 = load<TypedUploadF32Fn>("gafime_gpu_matrix_upload_f32_v2");
        typed_upload_f64 = load<TypedUploadF64Fn>("gafime_gpu_matrix_upload_f64_v2");
        typed_update_f32 = load<TypedUpdateF32Fn>(
            "gafime_gpu_matrix_update_target_f32_v2");
        typed_update_f64 = load<TypedUpdateF64Fn>(
            "gafime_gpu_matrix_update_target_f64_v2");
        typed_execute_f32 = load<TypedExecuteF32Fn>("gafime_gpu_execute_f32_v2");
        typed_execute_f64 = load<TypedExecuteF64Fn>("gafime_gpu_execute_f64_v2");
        typed_execution_memory = load<TypedExecutionMemoryFn>(
            "gafime_gpu_execution_memory_peak_v2");
        typed_permutation_memory = try_load<TypedPermutationMemoryFn>(
            "gafime_gpu_permutation_memory_peak_v2");
        typed_permutation_f32 = try_load<TypedPermutationF32Fn>(
            "gafime_gpu_permutation_pvalues_f32_v2");
        typed_permutation_f64 = try_load<TypedPermutationF64Fn>(
            "gafime_gpu_permutation_pvalues_f64_v2");
        const bool any_typed_permutation = typed_permutation_memory != nullptr ||
            typed_permutation_f32 != nullptr || typed_permutation_f64 != nullptr;
        const bool complete_typed_permutation = typed_permutation_memory != nullptr &&
            typed_permutation_f32 != nullptr && typed_permutation_f64 != nullptr;
        if (any_typed_permutation && !complete_typed_permutation) {
            fail("historical typed Metal payload exposes an incomplete optional permutation family");
        }
        if (complete_typed_permutation) {
            optional_symbols.emplace_back("gafime_gpu_permutation_memory_peak_v2");
            optional_symbols.emplace_back("gafime_gpu_permutation_pvalues_f32_v2");
            optional_symbols.emplace_back("gafime_gpu_permutation_pvalues_f64_v2");
        }
        typed_diagnostics = load<TypedDiagnosticsFn>("gafime_gpu_interaction_diagnostics");
        typed_matrix_free = load<TypedMatrixFreeFn>("gafime_gpu_matrix_free");
    }

    CanonicalPayloadApi(const CanonicalPayloadApi&) = delete;
    CanonicalPayloadApi& operator=(const CanonicalPayloadApi&) = delete;

    ~CanonicalPayloadApi() {
        if (handle != nullptr) dlclose(handle);
    }

    bool typed() const {
        return surface == PayloadAbiSurface::TypedPrecisionV1_1;
    }

    const char* abi_surface_name() const {
        return typed() ? "precision-typed-v1.1" : "numeric-route-v2";
    }

    int free_matrix(GafimeGpuMatrix matrix) const {
        if (matrix == nullptr) return GAFIME_STATUS_OK;
        if (typed()) {
            typed_matrix_free(matrix);
            return GAFIME_STATUS_OK;
        }
        return matrix_free(matrix);
    }

    template <typename Fn>
    Fn load(const char* name) {
        Fn function = try_load<Fn>(name);
        if (function == nullptr) {
            const char* detail = dlerror();
            const std::string detail_text =
                detail == nullptr ? "symbol lookup failed" : detail;
            if (handle != nullptr) dlclose(handle);
            handle = nullptr;
            fail(
                "exact Metal payload is missing canonical ABI symbol " +
                std::string(name) + ": " +
                detail_text);
        }
        return function;
    }

    template <typename Fn>
    Fn try_load(const char* name) {
        dlerror();
        void* symbol = dlsym(handle, name);
        const char* detail = dlerror();
        if (symbol == nullptr || detail != nullptr) return nullptr;
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
    std::string abi_surface;
    uint32_t route_count = 0;
    uint32_t profile_mask = 0;
    uint32_t storage_dtype_mask = 0;
    uint32_t result_dtype_mask = 0;
    int route_query_status = GAFIME_STATUS_DEVICE_ERROR;
    int route_fill_status = GAFIME_STATUS_DEVICE_ERROR;
    int matrix_alloc_status = GAFIME_STATUS_DEVICE_ERROR;
    int matrix_upload_status = GAFIME_STATUS_DEVICE_ERROR;
    int matrix_update_target_status = GAFIME_STATUS_DEVICE_ERROR;
    int execute_status = GAFIME_STATUS_DEVICE_ERROR;
    int execution_memory_peak_status = GAFIME_STATUS_DEVICE_ERROR;
    uint64_t execution_memory_peak_bytes = 0;
    int permutation_memory_peak_status = GAFIME_STATUS_DEVICE_ERROR;
    uint64_t permutation_memory_peak_bytes = 0;
    int permutation_pvalues_status = GAFIME_STATUS_DEVICE_ERROR;
    int interaction_diagnostics_status = GAFIME_STATUS_DEVICE_ERROR;
    uint64_t permutation_pvalue_count = 0;
    uint64_t diagnostic_overflow_rows = 0;
    uint32_t diagnostic_flags = 0;
    bool permutation_supported = false;
    int matrix_free_status = GAFIME_STATUS_DEVICE_ERROR;
    bool mixed_route_rejected = false;
    bool fp64_route_rejected = false;
    std::vector<std::string> symbols;
    std::vector<std::string> optional_symbols;
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

std::string sha256_bytes(const void* data, size_t size) {
    Sha256 hash;
    if (size != 0) {
        hash.update(static_cast<const uint8_t*>(data), size);
    }
    return hash.finish();
}

struct MetalInputDataset {
    // All buffers consumed by the Metal kernels are explicitly float32.  The
    // common-f64 source is retained only long enough to authenticate the
    // deterministic source bytes before the host-side f64-to-f32 conversion.
    std::vector<float> host_features;
    std::vector<float> canonical_features;
    std::vector<float> host_target;
    std::vector<float> host_means;
    std::string source_matrix_sha256;
    std::string source_target_sha256;
    std::string execution_matrix_sha256;
    std::string execution_target_sha256;
    std::string source_dtype;
    std::string generator;
};

MetalInputDataset make_input_dataset(const Options& options) {
    MetalInputDataset dataset;
    const size_t matrix_size =
        static_cast<size_t>(options.rows) * options.candidates;
    dataset.host_features.resize(matrix_size);
    dataset.canonical_features.resize(matrix_size);
    dataset.host_target.resize(options.rows);
    dataset.host_means.assign(options.candidates, 0.0f);

    if (options.input_policy == "common-f64") {
        std::vector<double> source_features(matrix_size);
        std::vector<double> source_target(options.rows);
        for (uint32_t row = 0; row < options.rows; ++row) {
            source_target[row] = 0.31 + static_cast<double>(
                (static_cast<uint64_t>(row) * 65537u + row / 5u + 29u) %
                99991u) / 99991.0;
            for (uint32_t column = 0; column < options.candidates; ++column) {
                source_features[
                    static_cast<size_t>(row) * options.candidates + column] =
                    0.07 + static_cast<double>(
                        (static_cast<uint64_t>(row) * (12289u + 3u * column) +
                         row / (5u + column % 13u) + 211u * column) %
                        100003u) / 100003.0;
            }
        }
        dataset.source_matrix_sha256 = sha256_bytes(
            source_features.data(), source_features.size() * sizeof(double));
        dataset.source_target_sha256 = sha256_bytes(
            source_target.data(), source_target.size() * sizeof(double));
        dataset.source_dtype = "float64";
        dataset.generator = "deterministic_integer_modulus.common_f64.v1";
        for (uint32_t row = 0; row < options.rows; ++row) {
            dataset.host_target[row] = static_cast<float>(source_target[row]);
            for (uint32_t column = 0; column < options.candidates; ++column) {
                const float value = static_cast<float>(
                    source_features[static_cast<size_t>(row) * options.candidates + column]);
                dataset.host_features[
                    static_cast<size_t>(column) * options.rows + row] = value;
                dataset.canonical_features[
                    static_cast<size_t>(row) * options.candidates + column] = value;
                dataset.host_means[column] += value;
            }
        }
    } else {
        std::vector<float> source_features(matrix_size);
        std::vector<float> source_target(options.rows);
        for (uint32_t row = 0; row < options.rows; ++row) {
            source_target[row] = 0.25f + static_cast<float>(
                (static_cast<uint64_t>(row) * 104729u + row / 7u + 17u) %
                100003u) / 100003.0f;
            for (uint32_t column = 0; column < options.candidates; ++column) {
                source_features[
                    static_cast<size_t>(row) * options.candidates + column] =
                    0.1f + static_cast<float>(
                        (static_cast<uint64_t>(row) * (8191u + 2u * column) +
                         row / (3u + column % 11u) + 97u * column) %
                        100019u) / 100019.0f;
            }
        }
        dataset.source_matrix_sha256 = sha256_bytes(
            source_features.data(), source_features.size() * sizeof(float));
        dataset.source_target_sha256 = sha256_bytes(
            source_target.data(), source_target.size() * sizeof(float));
        dataset.source_dtype = "float32";
        dataset.generator = "deterministic_integer_modulus.native_fp32.v1";
        for (uint32_t row = 0; row < options.rows; ++row) {
            dataset.host_target[row] = source_target[row];
            for (uint32_t column = 0; column < options.candidates; ++column) {
                const float value = source_features[
                    static_cast<size_t>(row) * options.candidates + column];
                dataset.host_features[
                    static_cast<size_t>(column) * options.rows + row] = value;
                dataset.canonical_features[
                    static_cast<size_t>(row) * options.candidates + column] = value;
                dataset.host_means[column] += value;
            }
        }
    }
    for (float& value : dataset.host_means) {
        value /= static_cast<float>(options.rows);
    }
    dataset.execution_matrix_sha256 = sha256_bytes(
        dataset.canonical_features.data(),
        dataset.canonical_features.size() * sizeof(float));
    dataset.execution_target_sha256 = sha256_bytes(
        dataset.host_target.data(), dataset.host_target.size() * sizeof(float));
    return dataset;
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

std::string shell_quote(std::string_view value) {
    std::string quoted("'");
    for (const char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += '\'';
    return quoted;
}

struct CommandResult {
    bool success = false;
    std::string output;
};

CommandResult run_command(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) return {};
    CommandResult result;
    char buffer[512]{};
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) result.output += buffer;
    const int status = pclose(pipe);
    result.success = status == 0;
    while (!result.output.empty() &&
           (result.output.back() == '\n' || result.output.back() == '\r')) {
        result.output.pop_back();
    }
    return result;
}

std::string command_output(const std::string& command) {
    CommandResult result = run_command(command);
    return result.success ? std::move(result.output) : std::string{};
}

struct GitProvenance {
    std::string executable;
    std::string sha256;
    std::string version;
    std::string trusted_path;
    std::vector<std::string> sanitized_environment_variables;
};

std::vector<std::string> inherited_git_environment_variables() {
    std::vector<std::string> variables;
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const std::string value(*entry);
        const size_t separator = value.find('=');
        if (separator != std::string::npos && value.compare(0, 4, "GIT_") == 0) {
            variables.push_back(value.substr(0, separator));
        }
    }
    std::sort(variables.begin(), variables.end());
    variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
    return variables;
}

std::string resolve_git_executable() {
    const std::array<const char*, 5> candidates = {
        "/usr/bin/git",
        "/bin/git",
        "/usr/local/bin/git",
        "/opt/homebrew/bin/git",
        "/opt/local/bin/git",
    };
    for (const char* candidate : candidates) {
        std::error_code error;
        const std::filesystem::path resolved =
            std::filesystem::canonical(candidate, error);
        if (!error && std::filesystem::is_regular_file(resolved, error) &&
            !error && access(resolved.c_str(), X_OK) == 0) {
            return resolved.string();
        }
    }
    fail(
        "could not resolve a trusted absolute Git executable from system locations; "
        "PATH lookup is deliberately rejected");
}

std::string git_trusted_path(const std::string& executable) {
    const std::filesystem::path parent =
        std::filesystem::path(executable).parent_path();
    return parent.string() + ":/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:/opt/local/bin";
}

std::string git_command(
    const GitProvenance& git,
    const std::string* root,
    const std::vector<std::string>& arguments
) {
    // /usr/bin/env -i removes every inherited variable, including all GIT_*
    // redirection/config/object controls.  The only Git configuration values
    // reintroduced here are fixed no-system/no-global paths for reproducibility.
    std::string command =
        "/usr/bin/env -i PATH=" + shell_quote(git.trusted_path) +
        " GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null "
        "GIT_CONFIG_SYSTEM=/dev/null " + shell_quote(git.executable);
    if (root != nullptr) {
        command += " -C " + shell_quote(*root);
    }
    for (const std::string& argument : arguments) {
        command += " " + shell_quote(argument);
    }
    command += " 2>/dev/null";
    return command;
}

std::string git_output(
    const GitProvenance& git,
    const std::string* root,
    std::initializer_list<std::string> arguments
) {
    return command_output(git_command(git, root, std::vector<std::string>(arguments)));
}

GitProvenance authenticate_git() {
    GitProvenance git;
    git.executable = resolve_git_executable();
    git.trusted_path = git_trusted_path(git.executable);
    git.sanitized_environment_variables = inherited_git_environment_variables();
    git.version = git_output(git, nullptr, {"--version"});
    if (git.version.rfind("git version ", 0) != 0) {
        fail("trusted Git executable returned an invalid version");
    }
    git.sha256 = sha256_file(git.executable);
    return git;
}

std::string observed_python_executable() {
    const char* virtual_env = std::getenv("VIRTUAL_ENV");
    if (virtual_env != nullptr && *virtual_env != '\0') {
        const std::filesystem::path candidate =
            std::filesystem::path(virtual_env) / "bin" / "python";
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error) && !error) {
            return std::filesystem::absolute(candidate, error).lexically_normal().string();
        }
    }
    std::string path = command_output("command -v python3");
    if (path.empty()) path = command_output("command -v python");
    return path;
}

struct CommandSnapshot {
    std::string command;
    std::string status = "unavailable";
    std::string output;
    std::string detail;
};

CommandSnapshot capture_command(const std::string& command) {
    CommandSnapshot snapshot;
    snapshot.command = command;
    const CommandResult result = run_command(command);
    snapshot.output = result.output;
    if (result.success && !result.output.empty()) {
        snapshot.status = "pass";
    } else if (!result.success) {
        snapshot.detail = "command was unavailable or returned a non-zero status";
    } else {
        snapshot.detail = "command returned no output";
    }
    return snapshot;
}

CommandSnapshot unavailable_snapshot(
    const std::string& command, const std::string& detail
) {
    CommandSnapshot snapshot;
    snapshot.command = command;
    snapshot.detail = detail;
    return snapshot;
}

struct ClockPowerState {
    // system_profiler is an Apple-provided, stable device/capability snapshot.
    // It is intentionally retained before and after the benchmark even though
    // it does not expose a live clock reading.
    CommandSnapshot system_profiler;
    // pmset reports the host power-management policy without requiring a
    // privileged powermetrics session. It is not mislabeled as a live CPU
    // frequency governor.
    CommandSnapshot cpu_power_management;
    CommandSnapshot cpu_governor;
    // Apple exposes command-buffer GPU timestamps (recorded per timed region),
    // but no portable public current Metal clock/power query. Keep that absence
    // explicit instead of inventing GPU telemetry.
    CommandSnapshot metal_gpu_clock_power;
};

ClockPowerState capture_clock_power_state() {
    ClockPowerState state;
    state.system_profiler = capture_command(
        "system_profiler SPDisplaysDataType -json 2>&1");
    state.cpu_power_management = capture_command("pmset -g custom 2>&1");
#if defined(__APPLE__)
    state.cpu_governor = unavailable_snapshot(
        "macOS CPU frequency governor API",
        "macOS does not expose a Linux-style CPU scaling governor through this "
        "public helper; pmset power-management policy is captured separately");
#else
    state.cpu_governor = unavailable_snapshot(
        "macOS CPU frequency governor API",
        "the Metal helper is intended for macOS and was built on a non-Apple host");
#endif
    state.metal_gpu_clock_power = unavailable_snapshot(
        "Apple Metal public current clock/power query",
        "Apple's public Metal API exposes per-command-buffer GPU timestamps, "
        "not a portable current clock or power reading; no dynamic GPU metric "
        "was fabricated");
    return state;
}

void append_command_snapshot(std::ostringstream& output, const CommandSnapshot& snapshot) {
    output << "{\"command\":\"" << json_escape(snapshot.command)
           << "\",\"status\":\"" << json_escape(snapshot.status)
           << "\",\"output\":\"" << json_escape(snapshot.output) << '"';
    if (!snapshot.detail.empty()) {
        output << ",\"detail\":\"" << json_escape(snapshot.detail) << '"';
    }
    output << '}';
}

void append_clock_power_state(std::ostringstream& output, const ClockPowerState& state) {
    output << "{\"system_profiler\":";
    append_command_snapshot(output, state.system_profiler);
    output << ",\"cpu_power_management\":";
    append_command_snapshot(output, state.cpu_power_management);
    output << ",\"cpu_governor\":";
    append_command_snapshot(output, state.cpu_governor);
    output << ",\"metal_gpu_clock_power\":";
    append_command_snapshot(output, state.metal_gpu_clock_power);
    output << '}';
}

void append_environment(std::ostringstream& output) {
    // Keep both semantic controls and variant-bound paths. The perf13
    // comparison normalizes the latter against wheel/payload identities.
    const std::array<const char*, 16> keys = {
        "GAFIME_METAL_V1_LIB",
        "GAFIME_METAL_V1_METALLIB",
        "GAFIME_WHEEL_PATH",
        "GAFIME_NATIVE_AFFINITY",
        "GAFIME_METAL_PARITY_TOLERANCE",
        "METAL_DEVICE_WRAPPER_TYPE",
        "MTL_DEBUG_LAYER",
        "MTL_CAPTURE_ENABLED",
        "MTL_DEBUG_LAYER_VALIDATE",
        "DYLD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "PATH",
    };
    output << '{';
    bool first = true;
    for (const char* key : keys) {
        const char* value = std::getenv(key);
        if (value == nullptr) continue;
        if (!first) output << ',';
        first = false;
        output << '"' << json_escape(key) << "\":\""
               << json_escape(value) << '"';
    }
    output << '}';
}

std::string canonical_directory(const std::string& path) {
    if (path.empty()) fail("source root is required");
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(path);
    if (!std::filesystem::is_directory(resolved)) {
        fail("source root is not a directory: " + resolved.string());
    }
    return resolved.string();
}

struct SourceTreeState {
    std::string status = "unavailable";
    std::vector<std::string> entries;
};

bool is_full_commit(const std::string& value) {
    return value.size() == 40 && std::all_of(
        value.begin(), value.end(),
        [](unsigned char character) { return std::isxdigit(character) != 0; });
}

SourceTreeState source_tree_state(
    const GitProvenance& git,
    const std::string& source_root
) {
    SourceTreeState state;
    if (git_output(git, &source_root, {"rev-parse", "--is-inside-work-tree"}) !=
        "true") {
        return state;
    }
    const CommandResult status = run_command(git_command(
        git,
        &source_root,
        {"status", "--porcelain=v1", "--untracked-files=all"}));
    if (!status.success) return state;
    std::istringstream lines(status.output);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty()) state.entries.push_back(line);
    }
    state.status = state.entries.empty() ? "clean" : "dirty";
    return state;
}

void write_source_tree_state(
    std::ostringstream& output,
    const SourceTreeState& state
) {
    output << "{\"status\": \"" << state.status << "\", \"entry_count\": "
           << state.entries.size() << ", \"entries\": [";
    for (size_t index = 0; index < state.entries.size(); ++index) {
        if (index != 0) output << ", ";
        output << '"' << json_escape(state.entries[index]) << '"';
    }
    output << "]}";
}

struct SourceBinding {
    std::string root;
    std::string commit;
    std::string git_dir;
    std::string git_common_dir;
    std::string relative_path;
    std::string source_sha256;
    std::string current_git_blob;
    std::string head_git_blob;
    SourceTreeState tree;
};

struct ProductSourceBinding {
    std::string root;
    std::string commit;
    std::string git_dir;
    std::string git_common_dir;
    SourceTreeState tree;
};

struct VerifiedGitDirectories {
    std::string git_dir;
    std::string git_common_dir;
};

VerifiedGitDirectories verified_git_directories(
    const GitProvenance& git,
    const std::string& root
) {
    const std::filesystem::path root_path(root);
    const std::string expected_text =
        gafime_native_trusted_git::expected_git_dir(root_path);
    if (expected_text.empty()) {
        fail("cannot resolve the source root's physical .git target: " + root);
    }
    const std::filesystem::path expected(expected_text);
    const std::string expected_common_text =
        gafime_native_trusted_git::expected_common_dir(expected_text);
    if (expected_common_text.empty()) {
        fail("cannot resolve the source root's physical Git common dir: " + root);
    }
    const std::filesystem::path expected_common(expected_common_text);
    std::error_code error;
    const std::string reported_text = git_output(git, &root, {"rev-parse", "--git-dir"});
    if (reported_text.empty()) fail("Git did not report a git-dir for source root: " + root);
    std::filesystem::path reported(reported_text);
    if (reported.is_relative()) reported = root_path / reported;
    reported = std::filesystem::canonical(reported, error);
    if (error || reported != expected) {
        fail(
            "Git reported a git-dir different from the source root's physical .git "
            "target: " + reported.string() + " != " + expected.string());
    }
    const std::string reported_common_text =
        git_output(git, &root, {"rev-parse", "--git-common-dir"});
    if (reported_common_text.empty()) {
        fail("Git did not report a git-common-dir for source root: " + root);
    }
    std::filesystem::path reported_common(reported_common_text);
    if (reported_common.is_relative()) reported_common = root_path / reported_common;
    reported_common = std::filesystem::canonical(reported_common, error);
    if (error || reported_common != expected_common) {
        fail(
            "Git reported a git-common-dir different from the source root's physical "
            "common dir: " + reported_common.string() + " != " + expected_common.string());
    }
    return {expected.string(), expected_common.string()};
}

void verify_git_top_level(
    const GitProvenance& git,
    const std::string& root
) {
    const std::string reported = git_output(git, &root, {"rev-parse", "--show-toplevel"});
    std::error_code error;
    const std::filesystem::path physical = std::filesystem::canonical(root, error);
    const std::filesystem::path reported_path =
        std::filesystem::canonical(reported, error);
    if (error || reported_path != physical) {
        fail(
            "Git reported a repository top-level different from the physical source root: " +
            reported + " != " + physical.string());
    }
}

ProductSourceBinding bind_product_source(
    const GitProvenance& git,
    const std::string& root_input,
    const std::string& expected_commit
) {
    ProductSourceBinding binding;
    binding.root = canonical_directory(root_input);
    verify_git_top_level(git, binding.root);
    const VerifiedGitDirectories git_directories =
        verified_git_directories(git, binding.root);
    binding.git_dir = git_directories.git_dir;
    binding.git_common_dir = git_directories.git_common_dir;
    binding.commit = git_output(git, &binding.root, {"rev-parse", "HEAD"});
    binding.tree = source_tree_state(git, binding.root);
    if (!is_full_commit(expected_commit) || binding.commit != expected_commit) {
        fail("product source HEAD does not match its declared commit");
    }
    if (binding.tree.status != "clean") {
        fail("product source root must be a clean Git work tree");
    }
    return binding;
}

SourceBinding bind_source(
    const GitProvenance& git,
    const std::string& root_input,
    const std::string& expected_commit,
    const std::string& source_input,
    const char* label
) {
    SourceBinding binding;
    binding.root = canonical_directory(root_input);
    verify_git_top_level(git, binding.root);
    const VerifiedGitDirectories git_directories =
        verified_git_directories(git, binding.root);
    binding.git_dir = git_directories.git_dir;
    binding.git_common_dir = git_directories.git_common_dir;
    binding.commit = git_output(git, &binding.root, {"rev-parse", "HEAD"});
    binding.tree = source_tree_state(git, binding.root);
    if (!is_full_commit(expected_commit) || binding.commit != expected_commit) {
        fail(std::string(label) + " source HEAD does not match its declared commit");
    }
    if (binding.tree.status != "clean") {
        fail(std::string(label) + " source root must be a clean Git work tree");
    }
    const std::filesystem::path source = canonical_path(source_input);
    const std::filesystem::path relative = source.lexically_relative(binding.root);
    if (relative.empty() || relative.is_absolute() ||
        *relative.begin() == std::filesystem::path("..")) {
        fail(std::string(label) + " source file is outside its declared source root");
    }
    binding.relative_path = relative.generic_string();
    binding.source_sha256 = sha256_file(source.string());
    binding.current_git_blob = git_output(
        git, &binding.root, {"hash-object", "--", binding.relative_path});
    binding.head_git_blob = git_output(
        git, &binding.root, {"rev-parse", "HEAD:" + binding.relative_path});
    if (!is_full_commit(binding.current_git_blob) ||
        binding.current_git_blob != binding.head_git_blob) {
        fail(std::string(label) + " source file does not match the declared clean HEAD blob");
    }
    return binding;
}

void write_source_binding(std::ostream& output, const SourceBinding& binding) {
    output << "{\"root\":\"" << json_escape(binding.root)
           << "\",\"commit\":\"" << json_escape(binding.commit)
           << "\",\"git_dir\":\"" << json_escape(binding.git_dir)
           << "\",\"git_common_dir\":\"" << json_escape(binding.git_common_dir)
           << "\",\"relative_path\":\"" << json_escape(binding.relative_path)
           << "\",\"sha256\":\"" << json_escape(binding.source_sha256)
           << "\",\"source_sha256\":\"" << json_escape(binding.source_sha256)
           << "\",\"current_git_blob\":\"" << json_escape(binding.current_git_blob)
           << "\",\"head_git_blob\":\"" << json_escape(binding.head_git_blob)
           << "\",\"tree_state\":";
    std::ostringstream tree;
    write_source_tree_state(tree, binding.tree);
    output << tree.str() << '}';
}

void write_product_source_binding(
    std::ostream& output, const ProductSourceBinding& binding
) {
    output << "{\"root\":\"" << json_escape(binding.root)
           << "\",\"commit\":\"" << json_escape(binding.commit)
           << "\",\"git_dir\":\"" << json_escape(binding.git_dir)
           << "\",\"git_common_dir\":\"" << json_escape(binding.git_common_dir)
           << "\",\"tree_state\":";
    std::ostringstream tree;
    write_source_tree_state(tree, binding.tree);
    output << tree.str() << '}';
}

void write_harness_source_blob(
    std::ostream& output, const SourceBinding& binding
) {
    output << "{\"relative_path\":\"" << json_escape(binding.relative_path)
           << "\",\"source_sha256\":\"" << json_escape(binding.source_sha256)
           << "\",\"current_git_blob\":\"" << json_escape(binding.current_git_blob)
           << "\",\"head_git_blob\":\"" << json_escape(binding.head_git_blob)
           << "\"}";
}

uint32_t parse_u32(const char* text, const char* option) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
        fail(std::string("invalid ") + option + " value");
    }
    return static_cast<uint32_t>(value);
}

Options parse_options(const GitProvenance& git, int argc, char** argv) {
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
        } else if (argument == "--source-root") {
            options.source_root = value_for("--source-root");
        } else if (argument == "--source-commit") {
            options.source_commit = value_for("--source-commit");
        } else if (argument == "--harness-source-root") {
            options.harness_source_root = value_for("--harness-source-root");
        } else if (argument == "--harness-source-commit") {
            options.harness_source_commit = value_for("--harness-source-commit");
        } else if (argument == "--input-policy") {
            options.input_policy = value_for("--input-policy");
        } else if (argument == "--variant") {
            options.variant = value_for("--variant");
        } else if (argument == "--ab-block") {
            options.ab_block = parse_u32(value_for("--ab-block"), "--ab-block");
            options.ab_block_set = true;
        } else if (argument == "--variant-sequence") {
            const std::string sequence = value_for("--variant-sequence");
            const size_t separator = sequence.find(',');
            if (separator == std::string::npos ||
                sequence.find(',', separator + 1) != std::string::npos) {
                fail("--variant-sequence must be baseline,candidate or candidate,baseline");
            }
            options.variant_sequence = {
                sequence.substr(0, separator), sequence.substr(separator + 1)};
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: " << argv[0]
                << " --json PATH --metallib PATH --shader-source PATH"
                << " --payload PATH --wheel PATH --source-root PATH"
                << " --source-commit SHA --harness-source-root PATH"
                << " --harness-source-commit SHA --variant baseline|candidate"
                << " --ab-block N --variant-sequence baseline,candidate|candidate,baseline"
                << " --input-policy common-f64|native"
                << " [--rows N] [--candidates N] [--mi-bins N] [--top-k N]"
                << " [--warmups N] [--repeats N]\n";
            std::exit(0);
        } else {
            fail("unknown option: " + std::string(argument));
        }
    }
    if (options.rows < 128 || options.rows > 4096 || options.candidates < 2 ||
        options.mi_bins < 2 || options.mi_bins > 48 || options.top_k == 0 ||
        options.top_k > options.candidates || options.warmups < kDefaultWarmups ||
        options.repeats < kDefaultRepeats || options.json_path.empty() ||
        options.source_root.empty() || !is_full_commit(options.source_commit) ||
        options.harness_source_root.empty() ||
        !is_full_commit(options.harness_source_commit) ||
        (options.input_policy != "common-f64" && options.input_policy != "native") ||
        (options.variant != "baseline" && options.variant != "candidate") ||
        !options.ab_block_set ||
        !((options.variant_sequence[0] == "baseline" &&
           options.variant_sequence[1] == "candidate") ||
          (options.variant_sequence[0] == "candidate" &&
           options.variant_sequence[1] == "baseline"))) {
        fail(
            "invalid dimensions/provenance: rows must be 128..4096, candidates must be >= 2, "
            "and top-k must be positive, "
            "MI bins must be 2..48, warmups >= 10, repeats >= 30, JSON output and "
            "clean product/harness roots, full product/harness commit SHAs, and "
            "input-policy=common-f64|native, "
            "variant=baseline|candidate, an A/B block, and a complete baseline/candidate "
            "variant sequence are required");
    }
    options.metallib_path = canonical_path(options.metallib_path);
    options.shader_source_path = canonical_path(options.shader_source_path);
    options.payload_path = canonical_path(options.payload_path);
    options.wheel_path = canonical_path(options.wheel_path);
    options.source_root = canonical_directory(options.source_root);
    options.harness_source_root = canonical_directory(options.harness_source_root);
    verify_git_top_level(git, options.source_root);
    (void)verified_git_directories(git, options.source_root);
    const SourceTreeState tree_state = source_tree_state(git, options.source_root);
    if (tree_state.status != "clean") {
        fail("source root must be a clean Git work tree");
    }
    const std::string observed_commit =
        git_output(git, &options.source_root, {"rev-parse", "HEAD"});
    if (observed_commit != options.source_commit) {
        fail("source root HEAD does not match --source-commit");
    }
    verify_git_top_level(git, options.harness_source_root);
    (void)verified_git_directories(git, options.harness_source_root);
    const SourceTreeState harness_tree_state =
        source_tree_state(git, options.harness_source_root);
    if (harness_tree_state.status != "clean") {
        fail("harness source root must be a clean Git work tree");
    }
    const std::string observed_harness_commit =
        git_output(git, &options.harness_source_root, {"rev-parse", "HEAD"});
    if (observed_harness_commit != options.harness_source_commit) {
        fail("harness source HEAD does not match --harness-source-commit");
    }
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

template <typename Measure>
uint32_t calibrate_loop_count(
    std::string_view label,
    Measure&& measure
) {
    uint32_t loop_count = 1;
    while (true) {
        bool stable = true;
        double minimum_us = std::numeric_limits<double>::infinity();
        for (uint32_t confirmation = 0;
             confirmation < kCalibrationConfirmationSamples;
             ++confirmation) {
            const double elapsed_us = measure(loop_count);
            minimum_us = std::min(minimum_us, elapsed_us);
            if (!std::isfinite(elapsed_us) ||
                elapsed_us < kSampleRegionCalibrationTargetUs) {
                stable = false;
                break;
            }
        }
        if (stable) return loop_count;
        if (loop_count >= kMaxLoopCount) {
            fail(std::string(label) +
                 " could not hold the fixed calibration target at the maximum loop "
                 "count; minimum observed " + std::to_string(minimum_us) + " us");
        }
        loop_count = loop_count > kMaxLoopCount / 2
            ? kMaxLoopCount
            : loop_count * 2;
    }
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
    record.samples_us.reserve(repeats);
    record.raw_samples_us.reserve(repeats);
    record.loop_counts_per_sample.reserve(repeats);
    const uint32_t calibrated_loop_count =
        calibrate_loop_count("Metal host timing", measure);
    for (uint32_t sample = 0; sample < repeats; ++sample) {
        const double raw_us = measure(calibrated_loop_count);
        if (raw_us < kSampleRegionTargetUs) {
            fail("Metal host timing calibration did not hold its fixed sample-region target: " +
                 std::to_string(raw_us) + " us");
        }
        record.loop_counts_per_sample.push_back(calibrated_loop_count);
        record.raw_samples_us.push_back(raw_us);
        record.samples_us.push_back(std::max(1.0e-6, raw_us / calibrated_loop_count));
    }
    record.loop_count_per_sample = calibrated_loop_count;
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
    auto measure_region = [&](uint32_t loop_count) {
        const auto calibration = submit(loop_count);
        return calibration.first > 0.0 ? calibration.first : calibration.second;
    };
    const uint32_t calibrated_loop_count =
        calibrate_loop_count("Metal command-buffer timing", measure_region);
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
    for (uint32_t sample_index = 0; sample_index < repeats; ++sample_index) {
        const auto sample = submit(calibrated_loop_count);
        const double sampled_region_us = sample.first > 0.0 ? sample.first : sample.second;
        if (sampled_region_us < kSampleRegionTargetUs) {
            fail("Metal command-buffer calibration did not hold its fixed sample-region target: " +
                 std::to_string(sampled_region_us) + " us");
        }
        const auto [gpu_us, host_us] = sample;
        loop_counts.push_back(calibrated_loop_count);
        raw_gpu_samples.push_back(gpu_us);
        raw_host_samples.push_back(host_us);
        gpu_samples.push_back(std::max(1.0e-6, gpu_us / calibrated_loop_count));
        host_samples.push_back(std::max(1.0e-6, host_us / calibrated_loop_count));
        valid_gpu_samples += gpu_us > 0.0 ? 1u : 0u;
    }
    const bool complete_gpu_timestamps = valid_gpu_samples == repeats;
    std::vector<double> selected_samples = complete_gpu_timestamps
        ? gpu_samples : host_samples;
    std::vector<double> selected_raw_samples = complete_gpu_timestamps
        ? raw_gpu_samples : raw_host_samples;
    return TimingRecord{
        std::move(operation),
        std::move(metric),
        complete_gpu_timestamps
            ? "metal_command_buffer_gpu_timestamps"
            : "host_steady_clock_command_buffer_sync_fallback",
        "commit_then_waitUntilCompleted_before_timestamp_read",
        std::move(selected_samples),
        std::move(selected_raw_samples),
        std::move(host_samples),
        std::move(raw_host_samples),
        std::move(loop_counts),
        calibrated_loop_count,
        valid_gpu_samples,
        std::move(gpu_samples),
        std::move(raw_gpu_samples),
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
    // C++ max_digits10 is sufficient to round-trip every finite binary64
    // value through JSON parsing.
    output << '[' << std::setprecision(std::numeric_limits<double>::max_digits10);
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
    FrozenPrecisionLaunchProtocol typed{};

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
        typed.abi_version = GAFIME_PRECISION_ABI_VERSION;
        typed.profile = route.profile;
        typed.base = &base;
    }
};

struct TypedResultFixture {
    std::vector<uint32_t> combo_indices;
    std::vector<float> metric_values;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> families;
    std::vector<uint64_t> candidate_ids;
    std::vector<uint32_t> row_flags;
    GafimeResultTable table{};

    TypedResultFixture(uint32_t capacity, uint32_t metric_count)
        : combo_indices(capacity),
          metric_values(static_cast<size_t>(capacity) * metric_count),
          ranks(capacity),
          families(capacity),
          candidate_ids(capacity),
          row_flags(capacity) {
        // Historical ABI 1.1's f32 entry point is a thin adapter over the
        // frozen ABI 1.0 result layout and therefore validates the ABI 1.0
        // result prefix.  The typed f64 result uses the ABI 1.1 extension,
        // but this fixture is intentionally only the f32 baseline surface.
        table.abi_version = GAFIME_ABI_VERSION;
        table.max_arity = 1;
        table.metric_count = metric_count;
        table.capacity = capacity;
        table.combo_indices = combo_indices.data();
        table.metric_values = metric_values.data();
        table.ranks = ranks.data();
        table.families = families.data();
        table.candidate_ids = candidate_ids.data();
        table.row_flags = row_flags.data();
    }

    void reset() {
        table.flags = 0;
        table.row_count = 0;
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

GafimeMutableBufferView f32_mutable_buffer(std::vector<float>& values) {
    GafimeMutableBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = values.data();
    view.element_capacity = values.size();
    view.byte_length = values.size() * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

struct TypedSignificanceFixture {
    uint64_t candidate_id = 0;
    float observed = 0.0f;
    float p_value = 0.0f;
    GafimePermutationSignificanceTable table{};

    TypedSignificanceFixture(uint64_t id, float value)
        : candidate_id(id), observed(value) {
        table.abi_version = GAFIME_ABI_VERSION;
        table.metric_count = 1;
        table.row_count = 1;
        table.candidate_ids = &candidate_id;
        table.observed_metric_values = &observed;
        table.p_values = &p_value;
    }
};

struct CanonicalSignificanceFixture {
    std::vector<uint64_t> candidate_ids;
    std::vector<float> observed;
    std::vector<float> p_values;
    GafimeNumericSignificanceTable table{};

    CanonicalSignificanceFixture(uint64_t id, float value)
        : candidate_ids{id}, observed{value}, p_values(1, 0.0f) {
        table.abi_version = GAFIME_PRECISION_ABI_VERSION;
        table.struct_size = sizeof(table);
        table.metric_count = 1;
        table.row_count = 1;
        table.candidate_ids = candidate_ids.data();
        table.observed_metric_values = f32_const_buffer(observed);
        table.p_values = f32_mutable_buffer(p_values);
    }
};

struct TypedDiagnosticsFixture {
    uint32_t combo_index = 0;
    uint64_t overflow_row_count = UINT64_MAX;
    uint32_t flags = UINT32_MAX;
    GafimeInteractionDiagnosticBatch table{};

    TypedDiagnosticsFixture() {
        table.abi_version = GAFIME_ABI_VERSION;
        table.max_arity = 1;
        table.row_count = 1;
        table.combo_indices = &combo_index;
        table.combo_index_count = 1;
        table.overflow_row_counts = &overflow_row_count;
        table.flags = &flags;
    }
};

struct CanonicalDiagnosticsFixture {
    uint32_t combo_index = 0;
    uint64_t overflow_row_count = UINT64_MAX;
    uint32_t flags = UINT32_MAX;
    GafimeNumericInteractionDiagnosticBatch table{};

    explicit CanonicalDiagnosticsFixture(const GafimeNumericRoute& route) {
        table.abi_version = GAFIME_PRECISION_ABI_VERSION;
        table.struct_size = sizeof(table);
        table.route = route;
        table.max_arity = 1;
        table.row_count = 1;
        table.combo_indices = &combo_index;
        table.combo_index_count = 1;
        table.overflow_row_counts = &overflow_row_count;
        table.row_flags = &flags;
    }
};

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
    evidence.abi_surface = api.abi_surface_name();
    evidence.symbols = api.symbols;
    evidence.optional_symbols = api.optional_symbols;

    GafimeNumericRoute route{};
    if (!api.typed()) {
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
        evidence.route_fill_status = api.routes(
            0,
            GAFIME_PRECISION_ABI_VERSION,
            sizeof(GafimeNumericRoute),
            &route,
            1,
            &route_count);
        if (evidence.route_fill_status != GAFIME_STATUS_OK ||
            route_count != 1 || route.abi_version != GAFIME_PRECISION_ABI_VERSION ||
            route.struct_size != sizeof(route) ||
            route.route_id != GAFIME_NUMERIC_ROUTE_FP32 ||
            route.profile != GAFIME_PRECISION_FP32 ||
            route.storage_dtype != GAFIME_DTYPE_F32 ||
            route.pointwise_dtype != GAFIME_DTYPE_F32 ||
            route.reduction_dtype != GAFIME_DTYPE_F32 ||
            route.result_dtype != GAFIME_DTYPE_F32) {
            fail("exact Metal payload advertised a non-canonical fp32 route");
        }
        evidence.profile_mask = GAFIME_PRECISION_PROFILE_MASK_FP32;
        evidence.storage_dtype_mask = GAFIME_DTYPE_MASK_F32;
        evidence.result_dtype_mask = GAFIME_DTYPE_MASK_F32;
    } else {
        FrozenPrecisionCapabilities capabilities{};
        evidence.route_query_status = api.typed_capabilities(0, &capabilities);
        evidence.route_fill_status = evidence.route_query_status;
        evidence.route_count = 1;
        evidence.profile_mask = capabilities.profile_mask;
        evidence.storage_dtype_mask = capabilities.storage_dtype_mask;
        evidence.result_dtype_mask = capabilities.result_dtype_mask;
        if (evidence.route_query_status != GAFIME_STATUS_OK ||
            capabilities.abi_version != GAFIME_PRECISION_ABI_VERSION ||
            capabilities.backend_kind != GAFIME_BACKEND_METAL ||
            capabilities.profile_mask != GAFIME_PRECISION_PROFILE_MASK_FP32 ||
            capabilities.storage_dtype_mask != GAFIME_DTYPE_MASK_F32 ||
            capabilities.result_dtype_mask != GAFIME_DTYPE_MASK_F32) {
            fail("typed Metal payload capability masks are not exactly fp32-only");
        }
        route.abi_version = GAFIME_PRECISION_ABI_VERSION;
        route.struct_size = sizeof(route);
        route.route_id = GAFIME_NUMERIC_ROUTE_FP32;
        route.profile = GAFIME_PRECISION_FP32;
        route.storage_dtype = GAFIME_DTYPE_F32;
        route.pointwise_dtype = GAFIME_DTYPE_F32;
        route.reduction_dtype = GAFIME_DTYPE_F32;
        route.result_dtype = GAFIME_DTYPE_F32;
        route.overflow_policy = GAFIME_OVERFLOW_IEEE;
    }

    auto generic_desc = [&](const GafimeNumericRoute& requested) {
        GafimeNumericMatrixDesc desc{};
        desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
        desc.struct_size = sizeof(desc);
        desc.route = requested;
        desc.layout = GAFIME_MATRIX_ROW_MAJOR;
        desc.rows = options.rows;
        desc.cols = options.candidates;
        desc.row_stride = options.candidates;
        desc.bytes = host_features.size() *
            (requested.storage_dtype == GAFIME_DTYPE_F64 ? sizeof(double) : sizeof(float));
        return desc;
    };
    auto typed_desc = [&](const GafimeNumericRoute& requested) {
        FrozenPrecisionMatrixDesc desc{};
        desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
        desc.profile = requested.profile;
        desc.dtype = requested.storage_dtype;
        desc.layout = GAFIME_MATRIX_ROW_MAJOR;
        desc.rows = options.rows;
        desc.cols = options.candidates;
        desc.row_stride = options.candidates;
        desc.bytes = host_features.size() *
            (requested.storage_dtype == GAFIME_DTYPE_F64 ? sizeof(double) : sizeof(float));
        return desc;
    };
    auto allocate = [&](const GafimeNumericRoute& requested, GafimeGpuMatrix* output) {
        if (api.typed()) {
            const FrozenPrecisionMatrixDesc desc = typed_desc(requested);
            return api.typed_matrix_alloc(0, &desc, output);
        }
        const GafimeNumericMatrixDesc desc = generic_desc(requested);
        return api.matrix_alloc(0, &desc, output);
    };
    auto require_rejection = [&](const GafimeNumericRoute& requested, const char* name) {
        GafimeGpuMatrix rejected = nullptr;
        const int status = allocate(requested, &rejected);
        if (rejected != nullptr) (void)api.free_matrix(rejected);
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || rejected != nullptr) {
            fail(
                std::string("Metal payload did not fail closed before allocation for ") +
                name + ": " + std::to_string(status));
        }
    };

    GafimeNumericRoute mixed_route = route;
    mixed_route.route_id = GAFIME_NUMERIC_ROUTE_MIXED;
    mixed_route.profile = GAFIME_PRECISION_MIXED;
    mixed_route.reduction_dtype = GAFIME_DTYPE_F64;
    mixed_route.result_dtype = GAFIME_DTYPE_F64;
    require_rejection(mixed_route, "mixed");
    evidence.mixed_route_rejected = true;
    GafimeNumericRoute fp64_route = route;
    fp64_route.route_id = GAFIME_NUMERIC_ROUTE_FP64;
    fp64_route.profile = GAFIME_PRECISION_FP64;
    fp64_route.storage_dtype = GAFIME_DTYPE_F64;
    fp64_route.pointwise_dtype = GAFIME_DTYPE_F64;
    fp64_route.reduction_dtype = GAFIME_DTYPE_F64;
    fp64_route.result_dtype = GAFIME_DTYPE_F64;
    require_rejection(fp64_route, "fp64");
    evidence.fp64_route_rejected = true;

    auto allocate_and_free = [&]() {
        GafimeGpuMatrix temporary = nullptr;
        const int status = allocate(route, &temporary);
        if (status != GAFIME_STATUS_OK || temporary == nullptr) {
            fail("canonical Metal matrix allocation failed: " + std::to_string(status));
        }
        const int free_status = api.free_matrix(temporary);
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
        evidence.matrix_alloc_status = allocate(route, &matrix);
        if (evidence.matrix_alloc_status != GAFIME_STATUS_OK || matrix == nullptr) {
            fail("canonical Metal matrix allocation failed: " +
                 std::to_string(evidence.matrix_alloc_status));
        }
        const GafimeConstBufferView features_view = f32_const_buffer(host_features);
        const GafimeConstBufferView target_view = f32_const_buffer(host_target);
        auto upload_matrix = [&]() {
            if (api.typed()) {
                return api.typed_upload_f32(
                    matrix,
                    host_features.data(),
                    host_target.data(),
                    options.rows,
                    options.candidates);
            }
            return api.matrix_upload(
                matrix,
                &route,
                &features_view,
                &target_view,
                options.rows,
                options.candidates);
        };
        evidence.matrix_upload_status = upload_matrix();
        if (evidence.matrix_upload_status != GAFIME_STATUS_OK) {
            fail("canonical Metal matrix upload failed: " +
                 std::to_string(evidence.matrix_upload_status));
        }
        auto update_target = [&]() {
            if (api.typed()) {
                return api.typed_update_f32(
                    matrix, host_target.data(), options.rows);
            }
            return api.matrix_update_target(
                matrix, &route, &target_view, options.rows);
        };
        evidence.matrix_update_target_status = update_target();
        if (evidence.matrix_update_target_status != GAFIME_STATUS_OK) {
            fail("canonical Metal target update failed: " +
                 std::to_string(evidence.matrix_update_target_status));
        }
        records.push_back(time_canonical_payload(
            "matrix_update_target",
            "none",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = update_target();
                if (status != GAFIME_STATUS_OK) {
                    fail("canonical Metal repeated target update failed: " +
                         std::to_string(status));
                }
            }));
        records.push_back(time_canonical_payload(
            "h2d_upload_or_unified_write",
            "none",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = upload_matrix();
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

        auto execute_payload = [&api, matrix](
            CanonicalProtocolFixture& protocol,
            CanonicalResultFixture& generic_result,
            TypedResultFixture& typed_result
        ) {
            if (api.typed()) {
                typed_result.reset();
                return api.typed_execute_f32(
                    matrix, &protocol.typed, &typed_result.table);
            }
            generic_result.reset();
            return api.execute(matrix, &protocol.numeric, &generic_result.table);
        };

        auto add_metric_record = [&](uint32_t metric, const char* name) {
            CanonicalProtocolFixture protocol(
                route,
                options.rows,
                options.candidates,
                options.mi_bins,
                metric,
                0);
            CanonicalResultFixture result(options.candidates, 1);
            TypedResultFixture typed_result(options.candidates, 1);
            records.push_back(time_canonical_payload(
                "metric_kernel",
                name,
                options.warmups,
                options.repeats,
                [&]() {
                    const int status = execute_payload(protocol, result, typed_result);
                    const uint64_t row_count = api.typed()
                        ? typed_result.table.row_count : result.table.row_count;
                    if (status != GAFIME_STATUS_OK || row_count != options.candidates) {
                        fail("canonical Metal metric execution failed: " +
                             std::to_string(status));
                    }
                }));
            const std::vector<float>& values = api.typed()
                ? typed_result.metric_values : result.metric_values;
            for (float value : values) {
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
        auto execution_memory_peak = [&]() {
            if (api.typed()) {
                return api.typed_execution_memory(
                    matrix,
                    &target_rank_protocol.typed,
                    &evidence.execution_memory_peak_bytes);
            }
            return api.execution_memory(
                matrix,
                &target_rank_protocol.numeric,
                &evidence.execution_memory_peak_bytes);
        };
        evidence.execution_memory_peak_status = execution_memory_peak();
        if (evidence.execution_memory_peak_status != GAFIME_STATUS_OK ||
            evidence.execution_memory_peak_bytes == 0) {
            fail("canonical Metal execution-memory forecast failed: " +
                 std::to_string(evidence.execution_memory_peak_status));
        }
        records.push_back(time_canonical_payload(
            "execution_memory_peak",
            "none",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = execution_memory_peak();
                if (status != GAFIME_STATUS_OK ||
                    evidence.execution_memory_peak_bytes == 0) {
                    fail("canonical Metal repeated execution-memory forecast failed: " +
                         std::to_string(status));
                }
            }));
        CanonicalResultFixture target_rank_result(options.candidates, 1);
        TypedResultFixture typed_target_rank_result(options.candidates, 1);
        records.push_back(time_canonical_payload(
            "ranking_kernel",
            "spearman_target_ranks",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = execute_payload(
                    target_rank_protocol, target_rank_result, typed_target_rank_result);
                const uint64_t row_count = api.typed()
                    ? typed_target_rank_result.table.row_count
                    : target_rank_result.table.row_count;
                if (status != GAFIME_STATUS_OK || row_count != options.candidates) {
                    fail("canonical Metal target-rank execution failed: " +
                         std::to_string(status));
                }
            }));
        const std::vector<float>& target_rank_values = api.typed()
            ? typed_target_rank_result.metric_values : target_rank_result.metric_values;
        for (float value : target_rank_values) {
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
        TypedResultFixture typed_ranking_result(options.top_k, 1);
        records.push_back(time_canonical_payload(
            "ranking_topk_and_gather",
            "spearman",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = execute_payload(
                    ranking_protocol, ranking_result, typed_ranking_result);
                const uint64_t row_count = api.typed()
                    ? typed_ranking_result.table.row_count
                    : ranking_result.table.row_count;
                if (status != GAFIME_STATUS_OK || row_count != options.top_k) {
                    fail("canonical Metal ranked execution failed: " +
                         std::to_string(status));
                }
                const std::vector<uint32_t>& combo_indices = api.typed()
                    ? typed_ranking_result.combo_indices : ranking_result.combo_indices;
                const std::vector<float>& metric_values = api.typed()
                    ? typed_ranking_result.metric_values : ranking_result.metric_values;
                for (uint32_t row = 0; row < options.top_k; ++row) {
                    if (combo_indices[row] >= options.candidates ||
                        !std::isfinite(metric_values[row])) {
                        fail("canonical Metal ranked output validation failed");
                    }
                }
            }));

        // Exercise the canonical significance surface against a real result
        // row.  The historical typed ABI made permutation symbols optional;
        // when that frozen payload omits the optional family we record the
        // explicit unsupported status instead of claiming an invocation.
        CanonicalProtocolFixture permutation_protocol(
            route,
            options.rows,
            options.candidates,
            options.mi_bins,
            kMetricPearson,
            0);
        CanonicalResultFixture permutation_result(options.candidates, 1);
        TypedResultFixture typed_permutation_result(options.candidates, 1);
        const int permutation_execute_status = execute_payload(
            permutation_protocol,
            permutation_result,
            typed_permutation_result);
        const uint64_t permutation_rows = api.typed()
            ? typed_permutation_result.table.row_count
            : permutation_result.table.row_count;
        if (permutation_execute_status != GAFIME_STATUS_OK || permutation_rows == 0) {
            fail("canonical Metal significance fixture execution failed: " +
                 std::to_string(permutation_execute_status));
        }
        // Every metric, rank, top-k, and significance preparation execute
        // above has already failed closed on a non-OK ABI status.  Record the
        // aggregate lifecycle status only after that complete sequence.
        evidence.execute_status = GAFIME_STATUS_OK;
        const uint64_t observed_candidate_id = api.typed()
            ? typed_permutation_result.candidate_ids[0]
            : permutation_result.candidate_ids[0];
        const float observed_metric = api.typed()
            ? typed_permutation_result.metric_values[0]
            : permutation_result.metric_values[0];
        if (!std::isfinite(observed_metric)) {
            fail("canonical Metal significance observed metric is non-finite");
        }
        permutation_protocol.base.permutations.permutation_count = 2;
        permutation_protocol.base.permutations.seed = 0x12345678u;

        const bool typed_permutation_available = api.typed() &&
            api.typed_permutation_memory != nullptr &&
            api.typed_permutation_f32 != nullptr;
        evidence.permutation_supported = !api.typed() || typed_permutation_available;
        if (!evidence.permutation_supported) {
            evidence.permutation_memory_peak_status = GAFIME_STATUS_UNSUPPORTED_BACKEND;
            evidence.permutation_pvalues_status = GAFIME_STATUS_UNSUPPORTED_BACKEND;
        } else {
            auto permutation_memory_peak = [&]() {
                if (api.typed()) {
                    return api.typed_permutation_memory(
                        matrix,
                        &permutation_protocol.typed,
                        1,
                        &evidence.permutation_memory_peak_bytes);
                }
                return api.permutation_memory(
                    matrix,
                    &permutation_protocol.numeric,
                    1,
                    &evidence.permutation_memory_peak_bytes);
            };
            evidence.permutation_memory_peak_status = permutation_memory_peak();
            if (evidence.permutation_memory_peak_status != GAFIME_STATUS_OK ||
                evidence.permutation_memory_peak_bytes == 0) {
                fail("canonical Metal permutation-memory forecast failed: " +
                     std::to_string(evidence.permutation_memory_peak_status));
            }
            records.push_back(time_canonical_payload(
                "permutation_memory_peak",
                "none",
                options.warmups,
                options.repeats,
                [&]() {
                    const int status = permutation_memory_peak();
                    if (status != GAFIME_STATUS_OK ||
                        evidence.permutation_memory_peak_bytes == 0) {
                        fail("canonical Metal repeated permutation-memory forecast failed: " +
                             std::to_string(status));
                    }
                }));

            TypedSignificanceFixture typed_significance(
                observed_candidate_id, observed_metric);
            CanonicalSignificanceFixture canonical_significance(
                observed_candidate_id, observed_metric);
            auto permutation_pvalues = [&]() {
                if (api.typed()) {
                    typed_significance.p_value = 0.0f;
                    return api.typed_permutation_f32(
                        matrix,
                        &permutation_protocol.typed,
                        &typed_significance.table);
                }
                canonical_significance.p_values[0] = 0.0f;
                return api.permutation_pvalues(
                    matrix,
                    &permutation_protocol.numeric,
                    &canonical_significance.table);
            };
            evidence.permutation_pvalues_status = permutation_pvalues();
            const float initial_p_value = api.typed()
                ? typed_significance.p_value
                : canonical_significance.p_values[0];
            if (evidence.permutation_pvalues_status != GAFIME_STATUS_OK ||
                !std::isfinite(initial_p_value) || initial_p_value < 0.0f ||
                initial_p_value > 1.0f) {
                fail("canonical Metal permutation p-value execution failed: " +
                     std::to_string(evidence.permutation_pvalues_status));
            }
            evidence.permutation_pvalue_count = 1;
            records.push_back(time_canonical_payload(
                "permutation_pvalues",
                "pearson",
                options.warmups,
                options.repeats,
                [&]() {
                    const int status = permutation_pvalues();
                    const float p_value = api.typed()
                        ? typed_significance.p_value
                        : canonical_significance.p_values[0];
                    if (status != GAFIME_STATUS_OK || !std::isfinite(p_value) ||
                        p_value < 0.0f || p_value > 1.0f) {
                        fail("canonical Metal repeated permutation p-value execution failed: " +
                             std::to_string(status));
                    }
                }));
        }

        TypedDiagnosticsFixture typed_diagnostics;
        CanonicalDiagnosticsFixture canonical_diagnostics(route);
        auto interaction_diagnostics = [&]() {
            if (api.typed()) {
                typed_diagnostics.overflow_row_count = UINT64_MAX;
                typed_diagnostics.flags = UINT32_MAX;
                return api.typed_diagnostics(matrix, &typed_diagnostics.table);
            }
            canonical_diagnostics.overflow_row_count = UINT64_MAX;
            canonical_diagnostics.flags = UINT32_MAX;
            return api.diagnostics(matrix, &canonical_diagnostics.table);
        };
        evidence.interaction_diagnostics_status = interaction_diagnostics();
        const uint64_t diagnostic_overflow = api.typed()
            ? typed_diagnostics.overflow_row_count
            : canonical_diagnostics.overflow_row_count;
        const uint32_t diagnostic_flags = api.typed()
            ? typed_diagnostics.flags
            : canonical_diagnostics.flags;
        if (evidence.interaction_diagnostics_status != GAFIME_STATUS_OK ||
            diagnostic_overflow != 0 || diagnostic_flags != 0) {
            fail("canonical Metal interaction diagnostics failed: " +
                 std::to_string(evidence.interaction_diagnostics_status));
        }
        evidence.diagnostic_overflow_rows = diagnostic_overflow;
        evidence.diagnostic_flags = diagnostic_flags;
        records.push_back(time_canonical_payload(
            "interaction_diagnostics",
            "none",
            options.warmups,
            options.repeats,
            [&]() {
                const int status = interaction_diagnostics();
                const uint64_t overflow = api.typed()
                    ? typed_diagnostics.overflow_row_count
                    : canonical_diagnostics.overflow_row_count;
                const uint32_t flags = api.typed()
                    ? typed_diagnostics.flags
                    : canonical_diagnostics.flags;
                if (status != GAFIME_STATUS_OK || overflow != 0 || flags != 0) {
                    fail("canonical Metal repeated interaction diagnostics failed: " +
                         std::to_string(status));
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
                const std::vector<uint32_t>& combo_indices = api.typed()
                    ? typed_ranking_result.combo_indices : ranking_result.combo_indices;
                const std::vector<float>& metric_values = api.typed()
                    ? typed_ranking_result.metric_values : ranking_result.metric_values;
                std::memcpy(
                    host_selected_indices.data(),
                    combo_indices.data(),
                    options.top_k * sizeof(uint32_t));
                std::memcpy(
                    host_selected_metrics.data(),
                    metric_values.data(),
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

        evidence.matrix_free_status = api.free_matrix(matrix);
        matrix = nullptr;
        if (evidence.matrix_free_status != GAFIME_STATUS_OK) {
            fail("canonical Metal matrix free failed: " +
                 std::to_string(evidence.matrix_free_status));
        }
    } catch (...) {
        if (matrix != nullptr) (void)api.free_matrix(matrix);
        throw;
    }
    evidence.validated = true;
    return evidence;
}

void write_timing_records(
    std::ostream& output,
    const std::vector<TimingRecord>& records
) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
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
        output << ", \"gpu_timestamp_samples_us\": ";
        write_samples(output, record.gpu_timestamp_samples_us);
        output << ", \"raw_gpu_timestamp_samples_us\": ";
        write_samples(output, record.raw_gpu_timestamp_samples_us);
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
        if ((!record.gpu_timestamp_samples_us.empty() &&
             record.gpu_timestamp_samples_us.size() != repeats) ||
            (!record.raw_gpu_timestamp_samples_us.empty() &&
             record.raw_gpu_timestamp_samples_us.size() != repeats) ||
            record.gpu_timestamp_samples_us.size() !=
                record.raw_gpu_timestamp_samples_us.size()) {
            fail("Metal GPU timestamp diagnostic lane has an invalid sample count: " +
                 record.operation);
        }
        if (record.loop_count_per_sample == 0 ||
            std::any_of(
                record.loop_counts_per_sample.begin(),
                record.loop_counts_per_sample.end(),
                [&](uint32_t value) { return value != record.loop_count_per_sample; })) {
            fail("Metal timing record changed its fixed calibration loop count: " +
                 record.operation);
        }
        const auto validate_lane = [&](const std::vector<double>& normalized,
                                       const std::vector<double>& raw,
                                       const char* lane,
                                       bool require_sample_floor) {
            if (normalized.size() != repeats || raw.size() != repeats) {
                fail(std::string("Metal ") + lane +
                     " timing lane has an invalid sample count: " + record.operation);
            }
            for (size_t index = 0; index < repeats; ++index) {
                const double normalized_value = normalized[index];
                const double raw_value = raw[index];
                if (!std::isfinite(normalized_value) || normalized_value <= 0.0 ||
                    !std::isfinite(raw_value) || raw_value <= 0.0 ||
                    (require_sample_floor && raw_value < kSampleRegionTargetUs)) {
                    fail(std::string("Metal timing record has an invalid ") + lane +
                         " calibrated sample: " + record.operation);
                }
                const double expected = raw_value /
                    static_cast<double>(record.loop_count_per_sample);
                const double ulp = std::nextafter(
                    expected, std::numeric_limits<double>::infinity()) - expected;
                if (std::abs(normalized_value - expected) >
                    std::max(1.0e-9, 4.0 * std::abs(ulp))) {
                    fail(std::string("Metal ") + lane +
                         " timing record raw/normalized duration mismatch: " +
                         record.operation);
                }
            }
        };
        validate_lane(record.samples_us, record.raw_samples_us, "selected", true);
        if (!record.host_synchronized_samples_us.empty() ||
            !record.raw_host_synchronized_samples_us.empty()) {
            validate_lane(
                record.host_synchronized_samples_us,
                record.raw_host_synchronized_samples_us,
                "host",
                false);
        }
        if (!record.gpu_timestamp_samples_us.empty()) {
            uint32_t valid_gpu_samples = 0;
            for (size_t index = 0; index < repeats; ++index) {
                const double normalized = record.gpu_timestamp_samples_us[index];
                const double raw = record.raw_gpu_timestamp_samples_us[index];
                if (!std::isfinite(normalized) || normalized <= 0.0 ||
                    !std::isfinite(raw) || raw < 0.0) {
                    fail("Metal GPU timestamp diagnostic lane is non-finite: " +
                         record.operation);
                }
                if (raw > 0.0) {
                    ++valid_gpu_samples;
                    const double expected = raw /
                        static_cast<double>(record.loop_count_per_sample);
                    const double ulp = std::nextafter(
                        expected, std::numeric_limits<double>::infinity()) - expected;
                    if (std::abs(normalized - expected) >
                        std::max(1.0e-9, 4.0 * std::abs(ulp))) {
                        fail("Metal GPU timestamp diagnostic raw/normalized mismatch: " +
                             record.operation);
                    }
                } else if (normalized != 1.0e-6) {
                    fail("Metal missing GPU timestamp sentinel is invalid: " +
                         record.operation);
                }
            }
            if (valid_gpu_samples != record.gpu_timestamp_valid_samples) {
                fail("Metal GPU timestamp diagnostic count does not match the samples: " +
                     record.operation);
            }
        } else if (record.gpu_timestamp_valid_samples != 0) {
            fail("Metal GPU timestamp diagnostic samples are missing: " + record.operation);
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
    const GitProvenance& git,
    const ProductSourceBinding& product_source,
    const SourceBinding& harness_source,
    const MetalInputDataset& input_dataset,
    bool unified_memory,
    const std::vector<TimingRecord>& records,
    const std::vector<TimingRecord>& canonical_payload_records,
    const CanonicalPayloadEvidence& canonical_payload,
    double canonical_result_checksum,
    double result_checksum,
    const ClockPowerState& clock_power_before,
    const ClockPowerState& clock_power_after
) {
    const bool gpu_timing_supported = std::all_of(
        records.begin(), records.end(), [&](const TimingRecord& record) {
            return record.clock != "host_steady_clock_command_buffer_sync_fallback";
        });
    const std::string python_executable = observed_python_executable();
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\n"
           << "  \"schema\": \"gafime.metal.native_timing.v1\",\n"
           << "  \"status\": \"pass\",\n"
           << "  \"backend\": \"metal\",\n"
           << "  \"profile\": \"fp32\",\n"
           << "  \"variant\": \"" << json_escape(options.variant) << "\",\n"
           << "  \"ab_block\": " << options.ab_block << ",\n"
           << "  \"variant_sequence\": [\""
           << json_escape(options.variant_sequence[0]) << "\", \""
           << json_escape(options.variant_sequence[1]) << "\"],\n"
           << "  \"process_isolation\": "
              "\"fresh_helper_process_per_variant_trial\",\n"
           << "  \"source_commit\": \"" << product_source.commit << "\",\n"
           << "  \"source_root\": \"" << json_escape(product_source.root) << "\",\n"
           << "  \"git_provenance\": {\"executable\": \""
           << json_escape(git.executable) << "\", \"sha256\": \""
           << json_escape(git.sha256) << "\", \"version\": \""
           << json_escape(git.version) << "\", \"trusted_path\": \""
           << json_escape(git.trusted_path)
           << "\", \"path_lookup_ignored\": true, \"sanitized_environment_variables\": [";
    for (size_t index = 0; index < git.sanitized_environment_variables.size(); ++index) {
        if (index != 0) output << ", ";
        output << "\"" << json_escape(git.sanitized_environment_variables[index]) << "\"";
    }
    output << "], \"removed_environment\": [";
    for (size_t index = 0; index < git.sanitized_environment_variables.size(); ++index) {
        if (index != 0) output << ", ";
        output << "\"" << json_escape(git.sanitized_environment_variables[index]) << "\"";
    }
    output << "], \"controlled_environment_variables\": [\"GIT_CONFIG_NOSYSTEM\", "
              "\"GIT_CONFIG_GLOBAL\", \"GIT_CONFIG_SYSTEM\"]},\n"
           << "  \"git\": {\"path\": \"" << json_escape(git.executable)
           << "\", \"sha256\": \"" << json_escape(git.sha256)
           << "\", \"version\": \"" << json_escape(git.version)
           << "\", \"removed_environment\": [";
    for (size_t index = 0; index < git.sanitized_environment_variables.size(); ++index) {
        if (index != 0) output << ", ";
        output << "\"" << json_escape(git.sanitized_environment_variables[index]) << "\"";
    }
    output << "]},\n"
           << "  \"source_tree_state\": ";
    write_source_tree_state(output, product_source.tree);
    output << ",\n  \"product_source_root\": \""
           << json_escape(product_source.root) << "\",\n"
           << "  \"product_source_tree_state\": ";
    write_source_tree_state(output, product_source.tree);
    output << ",\n  \"product_source_commit\": \""
           << json_escape(product_source.commit) << "\",\n"
           << "  \"product_source_binding\": ";
    write_product_source_binding(output, product_source);
    output << ",\n  \"harness_source_commit\": \""
           << json_escape(harness_source.commit) << "\",\n"
           << "  \"harness_source_root\": \""
           << json_escape(harness_source.root) << "\",\n"
           << "  \"harness_source_tree_state\": ";
    write_source_tree_state(output, harness_source.tree);
    output << ",\n  \"harness_source_binding\": ";
    write_source_binding(output, harness_source);
    output << ",\n  \"harness_source_blob\": ";
    write_harness_source_blob(output, harness_source);
    output << ",\n"
           << "  \"input_policy\": \""
           << json_escape(options.input_policy) << "\",\n"
           << "  \"input_identity\": {\"algorithm\": "
              "\"gafime.metal.native_timing.dataset.v2\", "
              "\"input_policy\": \""
           << json_escape(options.input_policy)
           << "\", \"policy_detail\": \""
           << json_escape(options.input_policy == "common-f64"
                              ? "deterministic float64 source converted to float32 before Metal execution"
                              : "deterministic native float32 source executed without cross-dtype conversion")
           << "\", \"generator\": \""
           << json_escape(input_dataset.generator)
           << "\", \"source_dtype\": \""
           << json_escape(input_dataset.source_dtype)
           << "\", \"matrix_sha256\": \""
           << input_dataset.source_matrix_sha256
           << "\", \"target_sha256\": \""
           << input_dataset.source_target_sha256
           << "\", \"execution_matrix_sha256\": \""
           << input_dataset.execution_matrix_sha256
           << "\", \"execution_target_sha256\": \""
           << input_dataset.execution_target_sha256
           << "\", \"matrix_shape\": [" << options.rows << ", "
           << options.candidates << "], \"target_shape\": [" << options.rows
           << "], \"matrix_dtype\": \"" << input_dataset.source_dtype
           << "\", \"target_dtype\": \"" << input_dataset.source_dtype
           << "\", \"execution_dtype\": \"float32\", "
              "\"execution_matrix_dtype\": \"float32\", "
              "\"execution_target_dtype\": \"float32\", "
              "\"layout\": \"row_major\"},\n"
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
           << "  \"environment\": ";
    append_environment(output);
    output << ",\n"
           << "  \"clock_and_power_capture_point\": \"before and after all timed "
              "benchmark regions\",\n"
           << "  \"clock_and_power_state\": {\"before\": ";
    append_clock_power_state(output, clock_power_before);
    output << ", \"after\": ";
    append_clock_power_state(output, clock_power_after);
    output << "},\n"
           << "  \"clock\": {\"host\": \"std::chrono::steady_clock\", "
              "\"device\": \"MTLCommandBuffer GPUStartTime/GPUEndTime "
              "after synchronized waitUntilCompleted\"},\n"
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
           << "    \"ingest_conversion\": \""
           << json_escape(options.input_policy == "common-f64"
                              ? "host-only common-f64 source conversion to float32 before all Metal execution; excluded from GPU timed records"
                              : "not present: native fp32 source policy")
           << "\",\n"
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
    write_file_identity(output, "wheel", options.wheel_path, true);
    write_file_identity(output, "python_executable", python_executable, true);
    write_file_identity(output, "harness_source", source_path, true);
    output << "    \"product_source_binding\": ";
    write_product_source_binding(output, product_source);
    output << ",\n    \"harness_source_binding\": ";
    write_source_binding(output, harness_source);
    output << '\n';
    output << "  },\n"
           << "  \"records\": [\n";
    write_timing_records(output, records);
    output << "  ],\n"
           << "  \"canonical_payload_lifecycle\": {\n"
           << "    \"status\": \""
           << (canonical_payload.validated ? "validated" : "invalid") << "\",\n"
           << "    \"schema\": \"gafime.native-decomposition.v1\",\n"
           << "    \"execution_layer\": \"installed_payload_dylib\",\n"
           << "    \"source_commit\": \""
           << json_escape(product_source.commit) << "\",\n"
           << "    \"source_root\": \""
           << json_escape(product_source.root) << "\",\n"
           << "    \"source_tree_state\": ";
    write_source_tree_state(output, product_source.tree);
    output << ",\n"
           << "    \"product_source_commit\": \""
           << json_escape(product_source.commit) << "\",\n"
           << "    \"product_source_root\": \""
           << json_escape(product_source.root) << "\",\n"
           << "    \"product_source_tree_state\": ";
    write_source_tree_state(output, product_source.tree);
    output << ",\n"
           << "    \"product_source_binding\": ";
    write_product_source_binding(output, product_source);
    output << ",\n"
           << "    \"harness_source_commit\": \""
           << json_escape(harness_source.commit) << "\",\n"
           << "    \"harness_source_root\": \""
           << json_escape(harness_source.root) << "\",\n"
           << "    \"harness_source_tree_state\": ";
    write_source_tree_state(output, harness_source.tree);
    output << ",\n"
           << "    \"harness_source_binding\": ";
    write_source_binding(output, harness_source);
    output << ",\n"
           << "    \"harness_source_blob\": ";
    write_harness_source_blob(output, harness_source);
    output << ",\n"
           << "    \"abi\": \"1.1\",\n"
           << "    \"abi_surface\": \""
           << json_escape(canonical_payload.abi_surface) << "\",\n"
           << "    \"profiles\": [\"fp32\"],\n"
           << "    \"operations\": [";
    if (canonical_payload.abi_surface == "numeric-route-v2") {
        output << "\"numeric_routes\", \"matrix_alloc\", \"matrix_upload\", "
                  "\"matrix_update_target\", \"execute\", "
                  "\"execution_memory_peak\", \"permutation_memory_peak\", "
                  "\"permutation_pvalues\", \"interaction_diagnostics\", "
                  "\"matrix_free\"";
    } else {
        output << "\"precision_capabilities\", \"matrix_alloc\", "
                  "\"matrix_upload\", \"matrix_update_target\", \"execute\", "
                  "\"execution_memory_peak\", \"interaction_diagnostics\", "
                  "\"matrix_free\"";
    }
    output << "],\n"
           << "    \"route_count\": " << canonical_payload.route_count << ",\n"
           << "    \"profile_mask\": " << canonical_payload.profile_mask << ",\n"
           << "    \"storage_dtype_mask\": "
           << canonical_payload.storage_dtype_mask << ",\n"
           << "    \"result_dtype_mask\": "
           << canonical_payload.result_dtype_mask << ",\n"
           << "    \"route_query_status\": " << canonical_payload.route_query_status << ",\n"
           << "    \"route_fill_status\": " << canonical_payload.route_fill_status << ",\n"
           << "    \"matrix_alloc_status\": " << canonical_payload.matrix_alloc_status << ",\n"
           << "    \"matrix_upload_status\": " << canonical_payload.matrix_upload_status << ",\n"
           << "    \"execute_status\": " << canonical_payload.execute_status << ",\n"
           << "    \"operation_status\": {\n"
           << "      \"matrix_update_target\": {\"status\": \""
           << (canonical_payload.matrix_update_target_status == GAFIME_STATUS_OK
                   ? "pass" : "error")
           << "\", \"abi_status\": "
           << canonical_payload.matrix_update_target_status << "},\n"
           << "      \"execution_memory_peak\": {\"status\": \""
           << (canonical_payload.execution_memory_peak_status == GAFIME_STATUS_OK
                   ? "pass" : "error")
           << "\", \"abi_status\": "
           << canonical_payload.execution_memory_peak_status
           << ", \"bytes\": " << canonical_payload.execution_memory_peak_bytes << "},\n"
           << "      \"permutation_memory_peak\": {\"status\": \""
           << (canonical_payload.permutation_supported
                   ? (canonical_payload.permutation_memory_peak_status == GAFIME_STATUS_OK
                          ? "pass" : "error")
                   : "unsupported")
           << "\", \"abi_status\": "
           << canonical_payload.permutation_memory_peak_status
           << ", \"bytes\": " << canonical_payload.permutation_memory_peak_bytes << "},\n"
           << "      \"permutation_pvalues\": {\"status\": \""
           << (canonical_payload.permutation_supported
                   ? (canonical_payload.permutation_pvalues_status == GAFIME_STATUS_OK
                          ? "pass" : "error")
                   : "unsupported")
           << "\", \"abi_status\": "
           << canonical_payload.permutation_pvalues_status
           << ", \"row_count\": " << canonical_payload.permutation_pvalue_count << "},\n"
           << "      \"interaction_diagnostics\": {\"status\": \""
           << (canonical_payload.interaction_diagnostics_status == GAFIME_STATUS_OK
                   ? "pass" : "error")
           << "\", \"abi_status\": "
           << canonical_payload.interaction_diagnostics_status
           << ", \"overflow_rows\": " << canonical_payload.diagnostic_overflow_rows
           << ", \"row_flags\": " << canonical_payload.diagnostic_flags << "}\n"
           << "    },\n"
           << "    \"matrix_free_status\": " << canonical_payload.matrix_free_status << ",\n"
           << "    \"permutation_supported\": "
           << (canonical_payload.permutation_supported ? "true" : "false") << ",\n"
           << "    \"mixed_route_rejected\": "
           << (canonical_payload.mixed_route_rejected ? "true" : "false") << ",\n"
           << "    \"fp64_route_rejected\": "
           << (canonical_payload.fp64_route_rejected ? "true" : "false") << ",\n"
           << "    \"symbols\": [";
    for (size_t index = 0; index < canonical_payload.symbols.size(); ++index) {
        if (index != 0) output << ", ";
        output << '"' << json_escape(canonical_payload.symbols[index]) << '"';
    }
    output << "],\n"
           << "    \"optional_symbols\": [";
    for (size_t index = 0; index < canonical_payload.optional_symbols.size(); ++index) {
        if (index != 0) output << ", ";
        output << '"' << json_escape(canonical_payload.optional_symbols[index]) << '"';
    }
    output << "],\n"
           << "    \"records_field\": \"canonical_payload_records\",\n"
           << "    \"result_checksum\": " << canonical_result_checksum << ",\n"
           << "    \"wheel_member\": \"gafime/_metal/libgafime_metal_v1.dylib\",\n"
           << "    \"wheel_member_sha256\": \""
           << sha256_file(options.payload_path) << "\",\n"
           << "    \"provenance\": {\n";
    write_file_identity(output, "payload", options.payload_path, true);
    write_file_identity(output, "wheel", options.wheel_path, true);
    write_file_identity(output, "harness_source", source_path, false);
    output << "    }\n"
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
            const GitProvenance git = authenticate_git();
            const Options options = parse_options(git, argc, argv);
            const std::string source_path = canonical_path(__FILE__);
            const std::string binary_path = canonical_path(argv[0]);
            const ProductSourceBinding product_source = bind_product_source(
                git, options.source_root, options.source_commit);
            const SourceBinding harness_source = bind_source(
                git,
                options.harness_source_root,
                options.harness_source_commit,
                source_path,
                "harness");
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
            const MetalInputDataset input_dataset = make_input_dataset(options);
            const std::vector<float>& host_features = input_dataset.host_features;
            // The supplemental shader lane consumes feature-major storage,
            // while canonical ABI 1.1 matrix upload is intentionally row-major.
            const std::vector<float>& canonical_features = input_dataset.canonical_features;
            const std::vector<float>& host_target = input_dataset.host_target;
            const std::vector<float>& host_means = input_dataset.host_means;
            std::vector<uint32_t> host_combos(options.candidates);
            std::iota(host_combos.begin(), host_combos.end(), 0u);

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

            // Capture host/device state outside every timed region. The
            // per-command-buffer GPU timestamps below remain the authoritative
            // arithmetic timing clock; these snapshots are provenance only.
            const ClockPowerState clock_power_before = capture_clock_power_state();
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
            const ClockPowerState clock_power_after = capture_clock_power_state();

            write_json(
                options,
                argc,
                argv,
                device,
                binary_path,
                source_path,
                git,
                product_source,
                harness_source,
                input_dataset,
                unified_memory,
                records,
                canonical_payload_records,
                canonical_payload,
                canonical_result_checksum,
                static_cast<double>(planning_checksum) + report_checksum,
                clock_power_before,
                clock_power_after);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metal_precision_native_timing: " << error.what() << '\n';
        return 1;
    }
}
