/*
 * ROCm native timing evidence for ABI 1.1 precision routes, including the
 * historical pre-freeze typed baseline.
 *
 * This executable deliberately loads the payload named by --payload instead
 * of linking against the build-tree target.  The resulting artifact is a
 * machine-readable decomposition of the canonical host lifecycle and the HIP
 * event time of each metric/ranking route.  It is a benchmark helper, not a
 * second product execution path.
 */

#include <hip/hip_runtime.h>

#include "gafime_gpu_abi.hpp"
#include "native_loop_plan_parser.hpp"
#include "rocm_native_direct_lane.hpp"
#include "trusted_git.hpp"

#ifndef GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE
#define GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE 1
#endif
#ifndef GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE_NAME
#define GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE_NAME "supplemental_internal_kernel"
#endif
#ifndef GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_ROOT
#define GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_ROOT ""
#endif
#ifndef GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_COMMIT
#define GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_COMMIT ""
#endif
#ifndef GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_SHA256
#define GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_SHA256 ""
#endif
#ifndef GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_HEADER_SHA256
#define GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_HEADER_SHA256 ""
#endif
#ifndef GAFIME_ROCM_NATIVE_TIMING_LINKED_DIRECT_SOURCE_SHA256
#define GAFIME_ROCM_NATIVE_TIMING_LINKED_DIRECT_SOURCE_SHA256 ""
#endif

#if GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE < 1 || \
    GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE > 3
#error "GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE must be 1, 2, or 3"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sched.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kBackend = GAFIME_BACKEND_ROCM;
constexpr uint32_t kAbi11 = GAFIME_PRECISION_ABI_VERSION;
constexpr std::array<uint32_t, 3> kProfileIds = {
    GAFIME_PRECISION_FP32,
    GAFIME_PRECISION_MIXED,
    GAFIME_PRECISION_FP64,
};
constexpr std::array<std::string_view, 3> kProfileNames = {
    "fp32", "mixed", "fp64",
};
constexpr std::array<std::string_view, 3> kEvidenceLanes = {
    "canonical_payload_api",
    "supplemental_internal_kernel",
    "supplemental_host_phase",
};
constexpr std::array<uint32_t, 4> kMetricIds = {
    GAFIME_METRIC_PEARSON,
    GAFIME_METRIC_R2,
    GAFIME_METRIC_MUTUAL_INFO,
    GAFIME_METRIC_SPEARMAN,
};

// Keep each sampled region long enough that the device/host clocks are not
// dominated by timer quantisation or launch overhead.  The value is the
// elapsed region (not the normalized per-call result) and is deliberately
// independent of the public benchmark's release gate.
constexpr double kSampleRegionTargetUs = 5000.0;
// Calibrate to a 2x guard band so normal clock/launch variance does not leave
// a recorded region just below the public 5 ms methodology floor.
constexpr double kSampleRegionCalibrationTargetUs = kSampleRegionTargetUs * 2.0;
constexpr uint32_t kPerRecordUntimedSameCellPreconditions = 10;
constexpr double kPerRecordUntimedPreconditionMinUs = 100000.0;
constexpr double kPreconditionDeviceBatchTargetUs = 1000.0;
constexpr uint32_t kMaxPreconditionBatchIterations = 4096;
constexpr uint32_t kMaxDevicePreconditionIterations = 1u << 20;
constexpr uint32_t kMaxHostPreconditionIterations = 1u << 24;
constexpr uint32_t kMaxLoopCount = 1u << 20;
constexpr uint32_t kBootstrapResamples = 2000;
constexpr uint64_t kBootstrapSeed = 20260809ULL;
constexpr uint32_t kMinimumOrderRepetitions = 30;

struct BenchmarkError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string payload;
    std::string output;
    std::string wheel;
    std::string source_root;
    std::string harness_source_root;
    std::string workload = "custom";
    std::string input_policy = "common-f64";
    std::string variant;
    std::string canonical_evidence_path;
    std::string loop_plan_path;
    bool calibration_only = false;
    std::string evidence_lane = "multi_lane";
    std::string artifact_kind = "rocm_events";
    int64_t ab_block = -1;
    std::vector<std::string> variant_sequence;
    uint32_t device = 0;
    uint64_t rows = 4096;
    uint32_t features = 8;
    uint32_t candidates = 8;
    uint32_t arity = 1;
    uint32_t mi_bins = 64;
    uint32_t top_k = 2;
    uint32_t warmups = 10;
    uint32_t repeats = 30;
    uint32_t order_repetitions = kMinimumOrderRepetitions;
    uint64_t order_seed = 0x475346494d45524fULL;
    uint64_t dataset_seed = 0x524f434d31312d31ULL;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw BenchmarkError(
        message +
        "\nusage: gafime_rocm_native_timing --payload PATH --json PATH "
        "[--workload NAME] [--rows N] [--features N] [--candidates N] [--arity 1..5] "
        "[--mi-bins N] [--top-k N] [--warmups N] [--repeats N] "
        "[--order-repetitions N] [--order-seed N] [--dataset-seed N] [--device N] [--source-root PATH] "
        "[--harness-source-root PATH] [--wheel PATH] "
        "[--input-policy common-f64|native] [--variant NAME] "
        "[--canonical-evidence PATH] [--loop-plan PATH] [--calibration-only] "
        "[--evidence-lane NAME] [--artifact-kind NAME] "
        "[--ab-block N --variant-sequence baseline,candidate]");
}

std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(',', start);
        const std::string item = value.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (item.empty()) usage_error("comma-separated values must not contain empty entries");
        result.push_back(item);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

uint64_t parse_u64(const char* option, const std::string& text) {
    if (text.empty() || text.front() == '-') {
        usage_error(std::string(option) + " requires a non-negative integer");
    }
    size_t end = 0;
    uint64_t value = 0;
    try {
        value = std::stoull(text, &end, 0);
    } catch (...) {
        usage_error(std::string(option) + " has an invalid integer: " + text);
    }
    if (end != text.size()) {
        usage_error(std::string(option) + " has an invalid integer: " + text);
    }
    return value;
}

uint32_t parse_u32(const char* option, const std::string& text) {
    const uint64_t value = parse_u64(option, text);
    if (value > std::numeric_limits<uint32_t>::max()) {
        usage_error(std::string(option) + " is outside uint32 range");
    }
    return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage_error("");
        }
        if (argument == "--calibration-only") {
            options.calibration_only = true;
            continue;
        }
        if (index + 1 >= argc) {
            usage_error(argument + " requires a value");
        }
        const std::string value = argv[++index];
        if (argument == "--payload") {
            options.payload = value;
        } else if (argument == "--json" || argument == "--output") {
            options.output = value;
        } else if (argument == "--wheel") {
            options.wheel = value;
        } else if (argument == "--source-root") {
            options.source_root = value;
        } else if (argument == "--harness-source-root") {
            options.harness_source_root = value;
        } else if (argument == "--workload") {
            options.workload = value;
        } else if (argument == "--input-policy") {
            options.input_policy = value;
        } else if (argument == "--variant") {
            options.variant = value;
        } else if (argument == "--canonical-evidence") {
            options.canonical_evidence_path = value;
        } else if (argument == "--loop-plan") {
            options.loop_plan_path = value;
        } else if (argument == "--evidence-lane") {
            options.evidence_lane = value;
        } else if (argument == "--artifact-kind") {
            options.artifact_kind = value;
        } else if (argument == "--ab-block") {
            options.ab_block = static_cast<int64_t>(parse_u32("--ab-block", value));
        } else if (argument == "--variant-sequence") {
            options.variant_sequence = split_csv(value);
        } else if (argument == "--device") {
            options.device = parse_u32("--device", value);
        } else if (argument == "--rows") {
            options.rows = parse_u64("--rows", value);
        } else if (argument == "--features") {
            options.features = parse_u32("--features", value);
        } else if (argument == "--candidates") {
            options.candidates = parse_u32("--candidates", value);
        } else if (argument == "--arity") {
            options.arity = parse_u32("--arity", value);
        } else if (argument == "--mi-bins") {
            options.mi_bins = parse_u32("--mi-bins", value);
        } else if (argument == "--top-k") {
            options.top_k = parse_u32("--top-k", value);
        } else if (argument == "--warmups") {
            options.warmups = parse_u32("--warmups", value);
        } else if (argument == "--repeats") {
            options.repeats = parse_u32("--repeats", value);
        } else if (argument == "--order-repetitions") {
            options.order_repetitions = parse_u32("--order-repetitions", value);
        } else if (argument == "--order-seed") {
            options.order_seed = parse_u64("--order-seed", value);
        } else if (argument == "--dataset-seed") {
            options.dataset_seed = parse_u64("--dataset-seed", value);
        } else {
            usage_error("unknown option: " + argument);
        }
    }
    if (options.payload.empty() || options.output.empty()) {
        usage_error("--payload and --json are required");
    }
    if (options.input_policy != "common-f64" && options.input_policy != "native") {
        usage_error("--input-policy must be common-f64 or native");
    }
    if (options.calibration_only && !options.loop_plan_path.empty()) {
        usage_error("--calibration-only cannot be combined with --loop-plan");
    }
    if (!options.calibration_only && options.loop_plan_path.empty()) {
        usage_error("recorded native evidence requires an immutable --loop-plan");
    }
    if (options.evidence_lane.empty() || options.artifact_kind.empty()) {
        usage_error("--evidence-lane and --artifact-kind must not be empty");
    }
    if (std::find(kEvidenceLanes.begin(), kEvidenceLanes.end(), options.evidence_lane) ==
        kEvidenceLanes.end()) {
        usage_error("--evidence-lane must name exactly one canonical, internal, or host lane");
    }
    if (options.calibration_only) {
        if (options.variant.empty() || options.ab_block >= 0 ||
            !options.variant_sequence.empty()) {
            usage_error(
                "calibration requires --variant and forbids A/B block/sequence fields");
        }
    } else if (
        options.variant.empty() || options.ab_block < 0 ||
        options.variant_sequence.size() != 2 ||
        std::find(
            options.variant_sequence.begin(), options.variant_sequence.end(),
            options.variant) == options.variant_sequence.end()) {
        usage_error("native A/B scheduling requires --variant, --ab-block, and a two-entry "
                    "--variant-sequence containing that variant");
    }
    if (options.rows == 0 || options.features == 0 || options.candidates == 0) {
        usage_error("rows, features, and candidates must be positive");
    }
    if (options.arity == 0 || options.arity > 5 || options.arity > options.features) {
        usage_error("arity must be in 1..5 and no greater than features");
    }
    if (options.mi_bins != 2 && options.mi_bins != 4 && options.mi_bins != 8 &&
        options.mi_bins != 12 && options.mi_bins != 16 && options.mi_bins != 24 &&
        options.mi_bins != 32 && options.mi_bins != 48 && options.mi_bins != 64 &&
        options.mi_bins != 96) {
        usage_error("mi-bins must be one of 2,4,8,12,16,24,32,48,64,96");
    }
    const uint32_t minimum_warmups = options.calibration_only ? 1u : 10u;
    const uint32_t minimum_repeats = options.calibration_only ? 1u : 30u;
    const uint32_t minimum_order_repetitions = options.calibration_only ? 1u : kMinimumOrderRepetitions;
    if (options.warmups < minimum_warmups || options.repeats < minimum_repeats ||
        options.order_repetitions < minimum_order_repetitions) {
        usage_error(
            "warmups must be at least 10, repeats at least 30, and order-repetitions at least 30");
    }
    if (options.top_k > options.candidates) {
        usage_error("top-k cannot exceed candidates");
    }
    if (options.calibration_only) {
        options.order_repetitions = 1;
        options.warmups = 1;
        options.repeats = 1;
    }
    return options;
}

void require_status(int status, const std::string& operation) {
    if (status != GAFIME_STATUS_OK) {
        throw BenchmarkError(operation + " returned GAFIME status " + std::to_string(status));
    }
}

void require_hip(hipError_t status, const std::string& operation) {
    if (status != hipSuccess) {
        throw BenchmarkError(
            operation + " returned HIP error " + std::to_string(static_cast<int>(status)) +
            ": " + hipGetErrorString(status));
    }
}

std::string absolute_path(const std::string& input) {
    std::error_code error;
    std::filesystem::path path(input);
    if (!path.is_absolute()) {
        path = std::filesystem::absolute(path, error);
    }
    if (!error) {
        const auto canonical = std::filesystem::weakly_canonical(path, error);
        if (!error) {
            path = canonical;
        }
    }
    return path.string();
}

std::string shell_quote(const std::string& value) {
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

std::string command_output(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) return {};
    std::string output;
    char buffer[512]{};
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
    const int status = pclose(pipe);
    if (status != 0) return {};
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}

std::string environment_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string observed_python_executable() {
    const std::string virtual_env = environment_value("VIRTUAL_ENV");
    if (!virtual_env.empty()) {
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

std::string canonical_path(const std::string& input) {
    std::error_code error;
    std::filesystem::path path(input);
    if (path.empty()) return {};
    if (!path.is_absolute()) path = std::filesystem::absolute(path, error);
    if (error) return path.string();
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    return error ? path.string() : canonical.string();
}

std::string relative_path_from_file(
    const std::string& base_file, const std::string& target) {
    std::error_code error;
    const auto relative = std::filesystem::relative(
        std::filesystem::path(canonical_path(target)),
        std::filesystem::path(canonical_path(base_file)).parent_path(), error);
    return error ? std::string() : relative.generic_string();
}

std::string inferred_source_root() {
    std::error_code error;
    std::filesystem::path source_path(__FILE__);
    if (!source_path.is_absolute()) source_path = std::filesystem::absolute(source_path, error);
    if (error) return {};
    return canonical_path(source_path.parent_path().parent_path().parent_path().string());
}

std::string source_root_path(const Options& options) {
    return canonical_path(options.source_root.empty() ? inferred_source_root() : options.source_root);
}

struct SourceTreeState {
    std::string status = "not_supplied";
    std::vector<std::string> entries;
    size_t entry_count = 0;
    std::string detail;
};

using TrustedGitRepository = gafime_native_trusted_git::RepositoryIdentity;

SourceTreeState source_tree_state(const std::string& source_root) {
    SourceTreeState state;
    if (source_root.empty()) return state;
    const auto repository = gafime_native_trusted_git::inspect(source_root);
    if (!repository.verified) {
        state.status = "unavailable";
        state.detail = repository.detail.empty()
            ? "source root is not a verified Git work tree"
            : repository.detail;
        return state;
    }
    const auto porcelain = gafime_native_trusted_git::git_command(
        source_root, "status --porcelain=v1 --untracked-files=all");
    if (!porcelain.succeeded()) {
        state.status = "unavailable";
        state.detail = "trusted Git status failed with exit status " +
            std::to_string(porcelain.exit_status);
        return state;
    }
    std::istringstream lines(porcelain.output);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) continue;
        ++state.entry_count;
        if (state.entries.size() < 512) state.entries.push_back(line);
    }
    state.status = state.entry_count == 0 ? "clean" : "dirty";
    return state;
}

std::string source_relative_path(const std::string& source_root) {
    if (source_root.empty()) return {};
    std::error_code error;
    const auto relative = std::filesystem::relative(
        std::filesystem::path(canonical_path(__FILE__)),
        std::filesystem::path(source_root), error);
    if (error || relative.empty() || relative.string().starts_with("..")) return {};
    return relative.generic_string();
}

std::string git_blob(const std::string& source_root, const std::string& relative_path,
                     bool head_blob) {
    if (source_root.empty() || relative_path.empty()) return {};
    const std::string command = head_blob
        ? "rev-parse HEAD:" + gafime_native_trusted_git::shell_quote(relative_path)
        : "hash-object --path=" + gafime_native_trusted_git::shell_quote(relative_path) +
              " -- " + gafime_native_trusted_git::shell_quote(relative_path);
    const auto result = gafime_native_trusted_git::git_command(source_root, command);
    if (!result.succeeded()) return {};
    const std::string& output = result.output;
    if (output.size() != 40 ||
        !std::all_of(output.begin(), output.end(), [](unsigned char value) {
            return std::isxdigit(value) != 0;
        })) {
        return {};
    }
    return output;
}

struct ToolIdentity {
    std::string command;
    std::string path;
    std::string version;
    std::string status = "unavailable";
};

std::string json_escape(std::string_view value);

ToolIdentity identify_tool(const char* executable, const char* version_argument = "--version") {
    ToolIdentity identity;
    identity.command = std::string(executable) + " " + version_argument;
    identity.path = command_output("command -v " + shell_quote(executable));
    identity.version = command_output(identity.command + " 2>&1");
    identity.status = identity.version.empty() ? "unavailable" : "observed";
    return identity;
}

void append_tool_identity(std::ostringstream& stream, const ToolIdentity& identity) {
    stream << "{\"command\":" << json_escape(identity.command)
           << ",\"path\":" << json_escape(identity.path)
           << ",\"status\":" << json_escape(identity.status)
           << ",\"version\":" << json_escape(identity.version) << '}';
}

struct SourceBinding {
    std::string root;
    std::string relative_path;
    std::string commit;
    std::string source_sha256;
    std::string current_git_blob;
    std::string head_git_blob;
    SourceTreeState tree;
    TrustedGitRepository git;
};

struct FileIdentity;
SourceBinding identify_source(
    const Options& options, const FileIdentity& benchmark_source, bool bind_source_file = false);

/* Small self-contained SHA-256 implementation used for provenance binding. */
class Sha256 {
public:
    Sha256() {
        state_ = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        };
    }

    void update(const uint8_t* bytes, size_t length) {
        bit_length_ += static_cast<uint64_t>(length) * 8u;
        while (length != 0) {
            const size_t copy = std::min(length, block_.size() - block_used_);
            std::memcpy(block_.data() + block_used_, bytes, copy);
            block_used_ += copy;
            bytes += copy;
            length -= copy;
            if (block_used_ == block_.size()) {
                transform(block_.data());
                block_used_ = 0;
            }
        }
    }

    std::string finish() {
        block_[block_used_++] = 0x80u;
        if (block_used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_used_), block_.end(), 0);
            transform(block_.data());
            block_used_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_used_), block_.begin() + 56, 0);
        for (int shift = 7; shift >= 0; --shift) {
            block_[56 + static_cast<size_t>(7 - shift)] =
                static_cast<uint8_t>((bit_length_ >> (shift * 8)) & 0xffu);
        }
        transform(block_.data());
        std::ostringstream stream;
        stream << std::hex << std::setfill('0');
        for (const uint32_t word : state_) {
            stream << std::setw(8) << word;
        }
        return stream.str();
    }

private:
    static uint32_t rotr(uint32_t value, uint32_t amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void transform(const uint8_t* block) {
        static constexpr std::array<uint32_t, 64> k = {
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
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            words[index] =
                (static_cast<uint32_t>(block[index * 4]) << 24) |
                (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                (static_cast<uint32_t>(block[index * 4 + 2]) << 8) |
                static_cast<uint32_t>(block[index * 4 + 3]);
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 = rotr(words[index - 15], 7) ^
                rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotr(words[index - 2], 17) ^
                rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + k[index] + words[index];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> block_{};
    size_t block_used_ = 0;
    uint64_t bit_length_ = 0;
};

struct FileIdentity {
    std::string path;
    std::string sha256;
    uint64_t size_bytes = 0;
};

FileIdentity identify_file_with_path_policy(
    const std::string& input, bool resolve_symlinks
) {
    std::error_code error;
    const std::string path = resolve_symlinks
        ? absolute_path(input)
        : std::filesystem::absolute(input, error).lexically_normal().string();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw BenchmarkError("cannot open provenance file: " + path);
    }
    Sha256 hash;
    std::array<uint8_t, 1 << 16> buffer{};
    uint64_t size = 0;
    while (stream) {
        stream.read(reinterpret_cast<char*>(buffer.data()),
                    static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            hash.update(buffer.data(), static_cast<size_t>(count));
            size += static_cast<uint64_t>(count);
        }
    }
    if (stream.bad() || size == 0) {
        throw BenchmarkError("cannot read non-empty provenance file: " + path);
    }
    return FileIdentity{path, hash.finish(), size};
}

FileIdentity identify_file(const std::string& input) {
    return identify_file_with_path_policy(input, true);
}

FileIdentity identify_external_canonical_evidence(const std::string& input) {
    if (input.empty()) {
        throw BenchmarkError("noncanonical ROCm helper requires --canonical-evidence");
    }
    const std::string path = absolute_path(input);
    std::error_code error;
    if (path.empty() || !std::filesystem::is_regular_file(path, error) || error) {
        throw BenchmarkError(
            "--canonical-evidence must identify a readable regular file");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw BenchmarkError(
            "--canonical-evidence must identify a readable regular file");
    }
    Sha256 hash;
    std::array<uint8_t, 1 << 16> buffer{};
    uint64_t size = 0;
    while (stream) {
        stream.read(reinterpret_cast<char*>(buffer.data()),
                    static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            hash.update(buffer.data(), static_cast<size_t>(count));
            size += static_cast<uint64_t>(count);
        }
    }
    if (stream.bad()) {
        throw BenchmarkError(
            "--canonical-evidence must identify a readable regular file");
    }
    return FileIdentity{path, hash.finish(), size};
}

std::string json_escape(std::string_view value) {
    std::ostringstream stream;
    stream << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': stream << "\\\""; break;
        case '\\': stream << "\\\\"; break;
        case '\b': stream << "\\b"; break;
        case '\f': stream << "\\f"; break;
        case '\n': stream << "\\n"; break;
        case '\r': stream << "\\r"; break;
        case '\t': stream << "\\t"; break;
        default:
            if (character < 0x20) {
                stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            } else {
                stream << static_cast<char>(character);
            }
            break;
        }
    }
    stream << '"';
    return stream.str();
}

void append_identity(std::ostringstream& stream, const FileIdentity& identity) {
    stream << "{\"path\":" << json_escape(identity.path)
           << ",\"sha256\":" << json_escape(identity.sha256)
           << ",\"size_bytes\":" << identity.size_bytes << '}';
}

struct CommandSnapshot {
    std::string command;
    std::string status = "unavailable";
    std::string output;
    std::string detail;
    std::string source = "command";
};

CommandSnapshot capture_command(const std::string& command) {
    CommandSnapshot snapshot;
    snapshot.command = command;
    snapshot.output = command_output(command);
    if (!snapshot.output.empty()) {
        snapshot.status = "pass";
    } else {
        snapshot.detail = "command was unavailable or returned no output";
    }
    return snapshot;
}

struct CpuGovernorSnapshot {
    std::string status = "unavailable";
    std::vector<std::string> values;
    std::string detail;
};

CpuGovernorSnapshot capture_cpu_governors() {
    CpuGovernorSnapshot snapshot;
#if defined(__linux__)
    const std::filesystem::path root("/sys/devices/system/cpu/cpufreq");
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) {
        snapshot.detail = "Linux CPU frequency policy directory is unavailable";
        return snapshot;
    }
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        if (error) break;
        const std::string name = entry.path().filename().string();
        if (!name.starts_with("policy")) continue;
        std::ifstream governor(entry.path() / "scaling_governor");
        std::string value;
        if (governor && std::getline(governor, value) && !value.empty()) {
            snapshot.values.push_back(value);
        }
    }
    std::sort(snapshot.values.begin(), snapshot.values.end());
    snapshot.values.erase(
        std::unique(snapshot.values.begin(), snapshot.values.end()), snapshot.values.end());
    if (!snapshot.values.empty()) {
        snapshot.status = "observed";
    } else {
        snapshot.detail = error
            ? "failed while reading Linux CPU frequency policies"
            : "no readable scaling_governor policy was found";
    }
#else
    snapshot.detail = "CPU frequency governor is not exposed by this platform helper";
#endif
    return snapshot;
}

std::string read_sysfs_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) return {};
    std::ostringstream value;
    value << input.rdbuf();
    std::string result = value.str();
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

std::string amd_sysfs_snapshot(bool dynamic) {
#if defined(__linux__)
    const std::filesystem::path root("/sys/class/drm");
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) return {};
    std::ostringstream output;
    output << '[';
    bool first = true;
    bool dynamic_observed = false;
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        if (error) break;
        const std::string card = entry.path().filename().string();
        if (!card.starts_with("card") || card.size() == 4 ||
            !std::isdigit(static_cast<unsigned char>(card[4]))) {
            continue;
        }
        const std::filesystem::path device = entry.path() / "device";
        if (read_sysfs_text(device / "vendor") != "0x1002") continue;
        const std::string device_id = read_sysfs_text(device / "device");
        if (!first) output << ',';
        first = false;
        output << "{\"card\":" << json_escape(card)
               << ",\"device\":" << json_escape(device_id);
        if (dynamic) {
            for (const char* name : {
                "pp_dpm_sclk", "pp_dpm_mclk", "power_dpm_state", "gpu_busy_percent"
            }) {
                const std::string value = read_sysfs_text(device / name);
                if (!value.empty()) {
                    output << ',' << json_escape(name) << ':' << json_escape(value);
                    dynamic_observed = true;
                }
            }
            const std::filesystem::path hwmon = device / "hwmon";
            std::error_code hwmon_error;
            for (const auto& hwmon_entry : std::filesystem::directory_iterator(hwmon, hwmon_error)) {
                if (hwmon_error) break;
                const std::string average = read_sysfs_text(hwmon_entry.path() / "power1_average");
                if (!average.empty()) {
                    output << ",\"power1_average\":" << json_escape(average);
                    dynamic_observed = true;
                }
            }
        }
        output << '}';
    }
    output << ']';
    if (first || (dynamic && !dynamic_observed)) return {};
    return output.str();
#else
    static_cast<void>(dynamic);
    return {};
#endif
}

CommandSnapshot capture_rocm_device_state() {
    const std::string command =
        "rocm-smi --showproductname --showdriverversion --showuniqueid "
        "--showclocks --showpower --json 2>&1";
    CommandSnapshot snapshot = capture_command(command);
    if (snapshot.status == "pass") return snapshot;
    snapshot.output = amd_sysfs_snapshot(true);
    if (!snapshot.output.empty()) {
        snapshot.status = "pass";
        snapshot.source = "linux_drm_sysfs";
        snapshot.command = "linux_drm_sysfs:/sys/class/drm";
        snapshot.detail = "rocm-smi unavailable; dynamic DRM sysfs fallback used";
    } else {
        snapshot.detail =
            "rocm-smi and Linux DRM sysfs dynamic clock/power sources were unavailable";
    }
    return snapshot;
}

struct ClockPowerState {
    CpuGovernorSnapshot cpu_governor;
    CommandSnapshot rocm_smi;
};

ClockPowerState capture_clock_power_state() {
    ClockPowerState state;
    state.cpu_governor = capture_cpu_governors();
    state.rocm_smi = capture_rocm_device_state();
    return state;
}

void append_command_snapshot(std::ostringstream& stream, const CommandSnapshot& snapshot) {
    stream << "{\"command\":" << json_escape(snapshot.command)
           << ",\"status\":" << json_escape(snapshot.status)
           << ",\"source\":" << json_escape(snapshot.source)
           << ",\"output\":" << json_escape(snapshot.output);
    if (!snapshot.detail.empty()) {
        stream << ",\"detail\":" << json_escape(snapshot.detail);
    }
    stream << '}';
}

void append_cpu_governor_snapshot(
    std::ostringstream& stream, const CpuGovernorSnapshot& snapshot
) {
    stream << "{\"status\":" << json_escape(snapshot.status) << ",\"values\":[";
    for (size_t index = 0; index < snapshot.values.size(); ++index) {
        if (index != 0) stream << ',';
        stream << json_escape(snapshot.values[index]);
    }
    stream << ']';
    if (!snapshot.detail.empty()) {
        stream << ",\"detail\":" << json_escape(snapshot.detail);
    }
    stream << '}';
}

void append_clock_power_state(std::ostringstream& stream, const ClockPowerState& state) {
    stream << "{\"cpu_governor\":";
    append_cpu_governor_snapshot(stream, state.cpu_governor);
    stream << ",\"rocm_smi\":";
    append_command_snapshot(stream, state.rocm_smi);
    stream << '}';
}

SourceBinding identify_source(
    const Options& options, const FileIdentity& benchmark_source, bool bind_source_file) {
    SourceBinding binding;
    binding.root = source_root_path(options);
    binding.relative_path = source_relative_path(binding.root);
    binding.git = gafime_native_trusted_git::inspect(binding.root);
    binding.commit = binding.git.commit;
    binding.source_sha256 = bind_source_file ? benchmark_source.sha256 : std::string();
    binding.current_git_blob = git_blob(binding.root, binding.relative_path, false);
    binding.head_git_blob = git_blob(binding.root, binding.relative_path, true);
    binding.tree = source_tree_state(binding.root);
    return binding;
}

SourceBinding identify_product_source(const Options& options) {
    SourceBinding binding;
    binding.root = source_root_path(options);
    binding.relative_path = "src/rocm/kernels.hip";
    binding.git = gafime_native_trusted_git::inspect(binding.root);
    binding.commit = binding.git.commit;
    const std::string product_source = absolute_path(binding.root + "/" + binding.relative_path);
    binding.source_sha256 = identify_file(product_source).sha256;
    binding.current_git_blob = git_blob(binding.root, binding.relative_path, false);
    binding.head_git_blob = git_blob(binding.root, binding.relative_path, true);
    binding.tree = source_tree_state(binding.root);
    return binding;
}

const char* expected_compiled_lane_name() {
    switch (GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE) {
    case 1: return "supplemental_internal_kernel";
    case 2: return "canonical_payload_api";
    case 3: return "supplemental_host_phase";
    default: return "invalid";
    }
}

void validate_compiled_lane(const Options& options) {
    if (std::string_view(GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE_NAME) !=
        expected_compiled_lane_name()) {
        throw BenchmarkError("ROCm native helper compiled lane name is inconsistent");
    }
    if (options.evidence_lane != expected_compiled_lane_name()) {
        throw BenchmarkError(
            "ROCm native helper --evidence-lane does not match its compile-time lane");
    }
}

void validate_linked_direct_product(
    const Options& options,
    const SourceBinding& source_binding,
    const SourceBinding& harness_source_binding
) {
    if (GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE != 1) return;
    const std::string linked_root = canonical_path(
        GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_ROOT);
    if (linked_root.empty() || linked_root != source_binding.root ||
        source_binding.commit != GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_COMMIT) {
        throw BenchmarkError(
            "ROCm direct helper product source does not match --source-root provenance");
    }
    const FileIdentity kernels = identify_file(linked_root + "/src/rocm/kernels.hip");
    const FileIdentity header = identify_file(linked_root + "/src/rocm/kernels.hpp");
    const FileIdentity direct = identify_file(
        harness_source_binding.root + "/tests/gpu/rocm_native_direct_lane.hip");
    if (kernels.sha256 != GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_SHA256 ||
        header.sha256 != GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_HEADER_SHA256 ||
        direct.sha256 != GAFIME_ROCM_NATIVE_TIMING_LINKED_DIRECT_SOURCE_SHA256) {
        throw BenchmarkError(
            "ROCm direct helper linked product/harness hashes do not match compiled provenance");
    }
    static_cast<void>(options);
}

void append_source_tree_state(std::ostringstream& stream, const SourceTreeState& state) {
    stream << "{\"status\":" << json_escape(state.status)
           << ",\"entry_count\":" << state.entry_count << ",\"entries\":[";
    for (size_t index = 0; index < state.entries.size(); ++index) {
        if (index != 0) stream << ',';
        stream << json_escape(state.entries[index]);
    }
    stream << ']';
    if (!state.detail.empty()) {
        stream << ",\"detail\":" << json_escape(state.detail);
    }
    stream << '}';
}

void append_source_binding(std::ostringstream& stream, const SourceBinding& binding) {
    stream << "{\"path\":" << json_escape(binding.root)
           << ",\"relative_path\":" << json_escape(binding.relative_path)
           << ",\"relative_source\":" << json_escape(binding.relative_path)
           << ",\"source_path\":" << json_escape(binding.relative_path)
           << ",\"commit\":" << json_escape(binding.commit)
           << ",\"sha256\":" << json_escape(binding.source_sha256)
           << ",\"source_sha256\":" << json_escape(binding.source_sha256)
           << ",\"git_blob\":" << json_escape(binding.current_git_blob)
           << ",\"current_git_blob\":" << json_escape(binding.current_git_blob)
           << ",\"head_git_blob\":" << json_escape(binding.head_git_blob)
           << ",\"git_identity\":{\"executable\":"
           << json_escape(binding.git.git.executable)
           << ",\"sha256\":" << json_escape(binding.git.git.sha256)
           << ",\"runtime_sha256\":"
           << json_escape(binding.git.git.runtime_sha256)
           << ",\"compiled_version\":"
           << json_escape(binding.git.git.compiled_version)
           << ",\"runtime_version\":"
           << json_escape(binding.git.git.runtime_version)
           << ",\"environment_scrubbed\":"
           << (binding.git.git.environment_scrubbed ? "true" : "false")
           << ",\"repository_verified\":"
           << (binding.git.verified ? "true" : "false")
           << ",\"executable_verified\":"
           << (binding.git.git.executable_verified ? "true" : "false")
           << ",\"show_toplevel\":"
           << json_escape(binding.git.show_toplevel)
           << ",\"git_dir\":"
           << json_escape(binding.git.git_dir)
           << ",\"common_dir\":"
           << json_escape(binding.git.common_dir)
           << ",\"expected_git_dir\":"
           << json_escape(binding.git.expected_git_dir)
           << ",\"expected_common_dir\":"
           << json_escape(binding.git.expected_common_dir)
           << ",\"detail\":"
           << json_escape(binding.git.detail) << "}"
           << ",\"tree_state\":";
    append_source_tree_state(stream, binding.tree);
    stream << '}';
}

void append_git_identity(
    std::ostringstream& stream, const TrustedGitRepository& repository) {
    const auto& git = repository.git;
    const std::string version = git.runtime_version.empty()
        ? git.compiled_version : git.runtime_version;
    stream << "{\"path\":" << json_escape(git.executable)
           << ",\"sha256\":" << json_escape(git.sha256)
           << ",\"runtime_sha256\":" << json_escape(git.runtime_sha256)
           << ",\"version\":" << json_escape(version)
           << ",\"removed_environment\":[\"GIT_*\",\"GIT_DIR\",\"GIT_WORK_TREE\","
              "\"GIT_INDEX_FILE\",\"GIT_OBJECT_DIRECTORY\","
              "\"GIT_ALTERNATE_OBJECT_DIRECTORIES\",\"GIT_COMMON_DIR\","
              "\"GIT_CONFIG\",\"GIT_CONFIG_COUNT\",\"GIT_CONFIG_KEY_*\","
              "\"GIT_CONFIG_VALUE_*\"],\"git_dir\":"
           << json_escape(repository.git_dir)
           << ",\"git_common_dir\":" << json_escape(repository.common_dir)
           << ",\"show_toplevel\":" << json_escape(repository.show_toplevel)
           << ",\"expected_git_dir\":"
           << json_escape(repository.expected_git_dir)
           << ",\"expected_git_common_dir\":"
           << json_escape(repository.expected_common_dir)
           << ",\"repository_verified\":"
           << (repository.verified ? "true" : "false")
           << ",\"executable_verified\":"
           << (git.executable_verified ? "true" : "false")
           << ",\"environment_scrubbed\":"
           << (git.environment_scrubbed ? "true" : "false")
           << '}';
}

std::string sha256_bytes(const void* data, size_t size) {
    Sha256 hash;
    if (size != 0) hash.update(reinterpret_cast<const uint8_t*>(data), size);
    return hash.finish();
}

template <typename T>
std::string sha256_vector(const std::vector<T>& values) {
    return sha256_bytes(values.data(), values.size() * sizeof(T));
}

using ImmutableLoopPlan = gafime_native_loop_plan::Plan;

ImmutableLoopPlan load_loop_plan(const Options& options) {
    ImmutableLoopPlan plan;
    plan.path = absolute_path(options.loop_plan_path);
    std::ifstream input(plan.path, std::ios::binary);
    if (!input) throw BenchmarkError("cannot open loop plan: " + plan.path);
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) throw BenchmarkError("cannot read loop plan: " + plan.path);
    try {
        plan = gafime_native_loop_plan::parse_plan(
            contents.str(), kMaxLoopCount,
            [](const void* data, size_t size) { return sha256_bytes(data, size); });
    } catch (const gafime_native_loop_plan::ParseError& error) {
        throw BenchmarkError(error.what());
    }
    plan.path = absolute_path(options.loop_plan_path);
    return plan;
}

/*
 * The baseline release predates the generic numeric-route symbols.  These
 * private copies are the historical pre-freeze ABI 1.1 layouts used by that payload; they
 * intentionally do not become part of the public header.  Keep the size and
 * pointer offsets checked here so a future native helper build cannot silently
 * reinterpret the baseline payload.
 */
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

struct FrozenResultTableF64 {
    uint32_t abi_version;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    double* metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    void* backend_private;
    uint64_t reserved[8];
};

struct FrozenInteractionDiagnosticBatch {
    uint32_t abi_version;
    uint32_t max_arity;
    uint64_t row_count;
    const uint32_t* combo_indices;
    uint64_t combo_index_count;
    uint64_t* overflow_row_counts;
    uint32_t* flags;
    uint32_t reserved32;
    uint64_t reserved[7];
};

static_assert(sizeof(FrozenPrecisionMatrixDesc) == 112);
static_assert(offsetof(FrozenPrecisionMatrixDesc, rows) == 24);
static_assert(offsetof(FrozenPrecisionMatrixDesc, bytes) == 40);
static_assert(offsetof(FrozenPrecisionMatrixDesc, reserved) == 48);
static_assert(sizeof(FrozenPrecisionCapabilities) == 88);
static_assert(offsetof(FrozenPrecisionCapabilities, reserved) == 24);
static_assert(sizeof(FrozenPrecisionLaunchProtocol) == 80);
static_assert(offsetof(FrozenPrecisionLaunchProtocol, base) == 8);
static_assert(sizeof(FrozenResultTableF64) == 152);
static_assert(offsetof(FrozenResultTableF64, metric_values) == 40);
static_assert(sizeof(FrozenInteractionDiagnosticBatch) == 112);
static_assert(offsetof(FrozenInteractionDiagnosticBatch, combo_indices) == 16);

enum class AbiSurface {
    GenericNumericRouteV2,
    TypedPrecisionV1_1,
};

constexpr const char* kGenericAbiSurface = "numeric-route-v2";
constexpr const char* kTypedAbiSurface = "precision-typed-v1.1";

constexpr std::array<const char*, 10> kGenericRequiredSymbols = {
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

constexpr std::array<const char*, 11> kTypedRequiredSymbols = {
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

constexpr std::array<const char*, 3> kTypedOptionalPermutationSymbols = {
    "gafime_gpu_permutation_memory_peak_v2",
    "gafime_gpu_permutation_pvalues_f32_v2",
    "gafime_gpu_permutation_pvalues_f64_v2",
};

struct Api {
    using GenericRoutes = int (*)(uint32_t, uint32_t, uint32_t, GafimeNumericRoute*, uint32_t, uint32_t*);
    using GenericAlloc = int (*)(uint32_t, const GafimeNumericMatrixDesc*, GafimeGpuMatrix*);
    using GenericUpload = int (*)(GafimeGpuMatrix, const GafimeNumericRoute*,
                                  const GafimeConstBufferView*, const GafimeConstBufferView*,
                                  uint64_t, uint32_t);
    using GenericUpdateTarget = int (*)(GafimeGpuMatrix, const GafimeNumericRoute*,
                                        const GafimeConstBufferView*, uint64_t);
    using GenericExecute = int (*)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*,
                                   GafimeNumericResultTable*);
    using GenericExecutionMemory = int (*)(GafimeGpuMatrix,
                                           const GafimeNumericLaunchProtocol*, uint64_t*);
    using GenericPermutationMemory = int (*)(GafimeGpuMatrix,
                                             const GafimeNumericLaunchProtocol*, uint64_t,
                                             uint64_t*);
    using GenericPermutationPvalues = int (*)(GafimeGpuMatrix,
                                              const GafimeNumericLaunchProtocol*,
                                              GafimeNumericSignificanceTable*);
    using GenericDiagnostics = int (*)(GafimeGpuMatrix,
                                       GafimeNumericInteractionDiagnosticBatch*);
    using GenericFree = int (*)(GafimeGpuMatrix);

    using TypedCapabilities = int (*)(uint32_t, FrozenPrecisionCapabilities*);
    using TypedAlloc = int (*)(uint32_t, const FrozenPrecisionMatrixDesc*, GafimeGpuMatrix*);
    using TypedUploadF32 = int (*)(GafimeGpuMatrix, const float*, const float*, uint64_t, uint32_t);
    using TypedUploadF64 = int (*)(GafimeGpuMatrix, const double*, const double*, uint64_t, uint32_t);
    using TypedUpdateTargetF32 = int (*)(GafimeGpuMatrix, const float*, uint64_t);
    using TypedUpdateTargetF64 = int (*)(GafimeGpuMatrix, const double*, uint64_t);
    using TypedExecuteF32 = int (*)(GafimeGpuMatrix, const FrozenPrecisionLaunchProtocol*,
                                    GafimeResultTable*);
    using TypedExecuteF64 = int (*)(GafimeGpuMatrix, const FrozenPrecisionLaunchProtocol*,
                                    FrozenResultTableF64*);
    using TypedExecutionMemory = int (*)(GafimeGpuMatrix,
                                         const FrozenPrecisionLaunchProtocol*, uint64_t*);
    using TypedDiagnostics = int (*)(GafimeGpuMatrix, FrozenInteractionDiagnosticBatch*);
    using TypedFree = void (*)(GafimeGpuMatrix);

    void* handle = nullptr;
    AbiSurface surface = AbiSurface::GenericNumericRouteV2;
    GenericRoutes generic_routes = nullptr;
    GenericAlloc generic_alloc = nullptr;
    GenericUpload generic_upload = nullptr;
    GenericUpdateTarget generic_update_target = nullptr;
    GenericExecute generic_execute = nullptr;
    GenericExecutionMemory generic_execution_memory = nullptr;
    GenericPermutationMemory generic_permutation_memory = nullptr;
    GenericPermutationPvalues generic_permutation_pvalues = nullptr;
    GenericDiagnostics generic_diagnostics = nullptr;
    GenericFree generic_free = nullptr;
    TypedCapabilities typed_capabilities = nullptr;
    TypedAlloc typed_alloc = nullptr;
    TypedUploadF32 typed_upload_f32 = nullptr;
    TypedUploadF64 typed_upload_f64 = nullptr;
    TypedUpdateTargetF32 typed_update_target_f32 = nullptr;
    TypedUpdateTargetF64 typed_update_target_f64 = nullptr;
    TypedExecuteF32 typed_execute_f32 = nullptr;
    TypedExecuteF64 typed_execute_f64 = nullptr;
    TypedExecutionMemory typed_execution_memory = nullptr;
    TypedDiagnostics typed_diagnostics = nullptr;
    TypedFree typed_free = nullptr;
    std::vector<std::string> required_symbols;
    std::vector<std::string> optional_symbols_present;
    std::vector<std::string> optional_symbols_missing;
    bool canonical_symbols_authenticated = false;

    Api() = default;

    ~Api() {
        if (handle != nullptr) {
            dlclose(handle);
        }
    }

    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;
    Api(Api&& other) noexcept
        : handle(std::exchange(other.handle, nullptr)),
          surface(other.surface),
          generic_routes(other.generic_routes),
          generic_alloc(other.generic_alloc),
          generic_upload(other.generic_upload),
          generic_update_target(other.generic_update_target),
          generic_execute(other.generic_execute),
          generic_execution_memory(other.generic_execution_memory),
          generic_permutation_memory(other.generic_permutation_memory),
          generic_permutation_pvalues(other.generic_permutation_pvalues),
          generic_diagnostics(other.generic_diagnostics),
          generic_free(other.generic_free),
          typed_capabilities(other.typed_capabilities),
          typed_alloc(other.typed_alloc),
          typed_upload_f32(other.typed_upload_f32),
          typed_upload_f64(other.typed_upload_f64),
          typed_update_target_f32(other.typed_update_target_f32),
          typed_update_target_f64(other.typed_update_target_f64),
          typed_execute_f32(other.typed_execute_f32),
          typed_execute_f64(other.typed_execute_f64),
          typed_execution_memory(other.typed_execution_memory),
          typed_diagnostics(other.typed_diagnostics),
          typed_free(other.typed_free),
          required_symbols(std::move(other.required_symbols)),
          optional_symbols_present(std::move(other.optional_symbols_present)),
          optional_symbols_missing(std::move(other.optional_symbols_missing)),
          canonical_symbols_authenticated(other.canonical_symbols_authenticated) {}
    Api& operator=(Api&& other) noexcept {
        if (this == &other) return *this;
        if (handle != nullptr) dlclose(handle);
        handle = std::exchange(other.handle, nullptr);
        surface = other.surface;
        generic_routes = other.generic_routes;
        generic_alloc = other.generic_alloc;
        generic_upload = other.generic_upload;
        generic_update_target = other.generic_update_target;
        generic_execute = other.generic_execute;
        generic_execution_memory = other.generic_execution_memory;
        generic_permutation_memory = other.generic_permutation_memory;
        generic_permutation_pvalues = other.generic_permutation_pvalues;
        generic_diagnostics = other.generic_diagnostics;
        generic_free = other.generic_free;
        typed_capabilities = other.typed_capabilities;
        typed_alloc = other.typed_alloc;
        typed_upload_f32 = other.typed_upload_f32;
        typed_upload_f64 = other.typed_upload_f64;
        typed_update_target_f32 = other.typed_update_target_f32;
        typed_update_target_f64 = other.typed_update_target_f64;
        typed_execute_f32 = other.typed_execute_f32;
        typed_execute_f64 = other.typed_execute_f64;
        typed_execution_memory = other.typed_execution_memory;
        typed_diagnostics = other.typed_diagnostics;
        typed_free = other.typed_free;
        required_symbols = std::move(other.required_symbols);
        optional_symbols_present = std::move(other.optional_symbols_present);
        optional_symbols_missing = std::move(other.optional_symbols_missing);
        canonical_symbols_authenticated = other.canonical_symbols_authenticated;
        return *this;
    }

    bool typed() const { return surface == AbiSurface::TypedPrecisionV1_1; }

    const char* abi_surface_name() const {
        return typed() ? kTypedAbiSurface : kGenericAbiSurface;
    }

    const char* route_source_name() const {
        return typed() ? "precision_capabilities" : "numeric_routes_v2";
    }

    const char* permutation_surface_status() const {
        if (!typed()) return "required_generic_surface";
        if (optional_symbols_present.empty()) return "not_exported_typed_optional";
        if (optional_symbols_missing.empty()) return "available_typed_optional";
        return "incomplete_typed_optional";
    }

    int free_matrix(GafimeGpuMatrix matrix) const {
        if (matrix == nullptr) return GAFIME_STATUS_OK;
        if (typed()) {
            typed_free(matrix);
            return GAFIME_STATUS_OK;
        }
        return generic_free(matrix);
    }
};

template <typename Function>
Function load_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
        throw BenchmarkError(std::string("payload is missing ") + name +
                             (error == nullptr ? "" : std::string(": ") + error));
    }
    return reinterpret_cast<Function>(symbol);
}

template <typename Function>
Function try_load_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) return nullptr;
    return reinterpret_cast<Function>(symbol);
}

Api open_payload(const std::string& input) {
    Api api;
    const std::string path = absolute_path(input);
    api.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr) {
        const char* error = dlerror();
        throw BenchmarkError("cannot load payload " + path +
                             (error == nullptr ? "" : std::string(": ") + error));
    }
    api.generic_routes = try_load_symbol<Api::GenericRoutes>(
        api.handle, "gafime_gpu_numeric_routes_v2");
    if (api.generic_routes != nullptr) {
        api.surface = AbiSurface::GenericNumericRouteV2;
        for (const char* symbol : kGenericRequiredSymbols) {
            if (try_load_symbol<void*>(api.handle, symbol) == nullptr) {
                throw BenchmarkError(
                    std::string("payload advertises generic numeric ABI but is missing ") +
                    symbol);
            }
        }
        api.required_symbols.assign(
            kGenericRequiredSymbols.begin(), kGenericRequiredSymbols.end());
        api.generic_alloc = load_symbol<Api::GenericAlloc>(
            api.handle, "gafime_gpu_matrix_alloc_v2");
        api.generic_upload = load_symbol<Api::GenericUpload>(
            api.handle, "gafime_gpu_matrix_upload_v2");
        api.generic_update_target = load_symbol<Api::GenericUpdateTarget>(
            api.handle, "gafime_gpu_matrix_update_target_v2");
        api.generic_execute = load_symbol<Api::GenericExecute>(
            api.handle, "gafime_gpu_execute_v2");
        api.generic_execution_memory = load_symbol<Api::GenericExecutionMemory>(
            api.handle, "gafime_gpu_execution_memory_peak_v2");
        api.generic_permutation_memory = load_symbol<Api::GenericPermutationMemory>(
            api.handle, "gafime_gpu_permutation_memory_peak_v2");
        api.generic_permutation_pvalues = load_symbol<Api::GenericPermutationPvalues>(
            api.handle, "gafime_gpu_permutation_pvalues_v2");
        api.generic_diagnostics = load_symbol<Api::GenericDiagnostics>(
            api.handle, "gafime_gpu_interaction_diagnostics_v2");
        api.generic_free = load_symbol<Api::GenericFree>(
            api.handle, "gafime_gpu_matrix_free_v2");
        api.canonical_symbols_authenticated = true;
    } else {
        if (try_load_symbol<void*>(api.handle, "gafime_gpu_precision_capabilities") == nullptr) {
            throw BenchmarkError(
                "payload exports neither a complete numeric-route-v2 nor typed-precision-v1.1 surface");
        }
        api.surface = AbiSurface::TypedPrecisionV1_1;
        for (const char* symbol : kTypedRequiredSymbols) {
            if (try_load_symbol<void*>(api.handle, symbol) == nullptr) {
                throw BenchmarkError(
                    std::string("payload exposes typed precision ABI but is missing ") + symbol);
            }
        }
        api.required_symbols.assign(
            kTypedRequiredSymbols.begin(), kTypedRequiredSymbols.end());
        for (const char* symbol : kTypedOptionalPermutationSymbols) {
            if (try_load_symbol<void*>(api.handle, symbol) != nullptr) {
                api.optional_symbols_present.emplace_back(symbol);
            } else {
                api.optional_symbols_missing.emplace_back(symbol);
            }
        }
        if (!api.optional_symbols_present.empty() && !api.optional_symbols_missing.empty()) {
            throw BenchmarkError(
                "typed payload exposes an incomplete optional permutation symbol family");
        }
        api.typed_capabilities = load_symbol<Api::TypedCapabilities>(
            api.handle, "gafime_gpu_precision_capabilities");
        api.typed_alloc = load_symbol<Api::TypedAlloc>(
            api.handle, "gafime_gpu_matrix_alloc_v2");
        api.typed_upload_f32 = load_symbol<Api::TypedUploadF32>(
            api.handle, "gafime_gpu_matrix_upload_f32_v2");
        api.typed_upload_f64 = load_symbol<Api::TypedUploadF64>(
            api.handle, "gafime_gpu_matrix_upload_f64_v2");
        api.typed_update_target_f32 = load_symbol<Api::TypedUpdateTargetF32>(
            api.handle, "gafime_gpu_matrix_update_target_f32_v2");
        api.typed_update_target_f64 = load_symbol<Api::TypedUpdateTargetF64>(
            api.handle, "gafime_gpu_matrix_update_target_f64_v2");
        api.typed_execute_f32 = load_symbol<Api::TypedExecuteF32>(
            api.handle, "gafime_gpu_execute_f32_v2");
        api.typed_execute_f64 = load_symbol<Api::TypedExecuteF64>(
            api.handle, "gafime_gpu_execute_f64_v2");
        api.typed_execution_memory = load_symbol<Api::TypedExecutionMemory>(
            api.handle, "gafime_gpu_execution_memory_peak_v2");
        api.typed_diagnostics = load_symbol<Api::TypedDiagnostics>(
            api.handle, "gafime_gpu_interaction_diagnostics");
        api.typed_free = load_symbol<Api::TypedFree>(
            api.handle, "gafime_gpu_matrix_free");
        api.canonical_symbols_authenticated = true;
    }
    return api;
}

bool payload_is_loaded(const std::string& input) {
    const std::string path = absolute_path(input);
#if defined(RTLD_NOLOAD)
    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (handle != nullptr) {
        dlclose(handle);
        return true;
    }
#endif
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(path) != std::string::npos) return true;
    }
    return false;
}

uint64_t dtype_size(uint32_t dtype) {
    if (dtype == GAFIME_DTYPE_F32) return sizeof(float);
    if (dtype == GAFIME_DTYPE_F64) return sizeof(double);
    throw BenchmarkError("unknown route dtype " + std::to_string(dtype));
}

GafimeNumericRoute synthetic_route(uint32_t profile) {
    GafimeNumericRoute route{};
    route.abi_version = kAbi11;
    route.struct_size = sizeof(route);
    route.route_id = profile;
    route.profile = profile;
    if (profile == GAFIME_PRECISION_FP32) {
        route.storage_dtype = GAFIME_DTYPE_F32;
        route.pointwise_dtype = GAFIME_DTYPE_F32;
        route.reduction_dtype = GAFIME_DTYPE_F32;
        route.result_dtype = GAFIME_DTYPE_F32;
    } else if (profile == GAFIME_PRECISION_MIXED) {
        route.storage_dtype = GAFIME_DTYPE_F32;
        route.pointwise_dtype = GAFIME_DTYPE_F32;
        route.reduction_dtype = GAFIME_DTYPE_F64;
        route.result_dtype = GAFIME_DTYPE_F64;
    } else if (profile == GAFIME_PRECISION_FP64) {
        route.storage_dtype = GAFIME_DTYPE_F64;
        route.pointwise_dtype = GAFIME_DTYPE_F64;
        route.reduction_dtype = GAFIME_DTYPE_F64;
        route.result_dtype = GAFIME_DTYPE_F64;
    } else {
        throw BenchmarkError("unknown precision profile " + std::to_string(profile));
    }
    route.overflow_policy = GAFIME_OVERFLOW_IEEE;
    return route;
}

bool route_matches_profile(const GafimeNumericRoute& route, uint32_t profile) {
    const GafimeNumericRoute expected = synthetic_route(profile);
    return route.abi_version == kAbi11 && route.struct_size >= sizeof(GafimeNumericRoute) &&
        route.route_id == expected.route_id && route.profile == expected.profile &&
        route.storage_dtype == expected.storage_dtype &&
        route.pointwise_dtype == expected.pointwise_dtype &&
        route.reduction_dtype == expected.reduction_dtype &&
        route.result_dtype == expected.result_dtype &&
        route.overflow_policy == expected.overflow_policy && route.flags == 0;
}

struct RouteDiscovery {
    std::array<GafimeNumericRoute, 3> by_profile{};
    uint32_t profile_mask = 0;
    uint32_t storage_dtype_mask = 0;
    uint32_t result_dtype_mask = 0;
    uint32_t flags = 0;
    bool route_synthesized = false;
};

RouteDiscovery synthetic_discovery() {
    RouteDiscovery discovery;
    discovery.profile_mask = 0x7u;
    discovery.storage_dtype_mask = GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64;
    discovery.result_dtype_mask = GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64;
    discovery.route_synthesized = true;
    for (size_t index = 0; index < kProfileIds.size(); ++index) {
        discovery.by_profile[index] = synthetic_route(kProfileIds[index]);
    }
    return discovery;
}

RouteDiscovery discover_routes(const Api& api, uint32_t device) {
    RouteDiscovery discovery;
    if (!api.typed()) {
        uint32_t route_count = 0;
        require_status(
            api.generic_routes(device, kAbi11, sizeof(GafimeNumericRoute), nullptr, 0, &route_count),
            "numeric_routes_v2 count query");
        if (route_count != kProfileIds.size()) {
            throw BenchmarkError("canonical ROCm payload advertised " + std::to_string(route_count) +
                                 " routes, expected fp32/mixed/fp64");
        }
        std::vector<GafimeNumericRoute> routes(route_count);
        require_status(
            api.generic_routes(device, kAbi11, sizeof(GafimeNumericRoute),
                               routes.data(), route_count, &route_count),
            "numeric_routes_v2 enumeration");
        std::array<bool, 3> seen{};
        for (const auto& route : routes) {
            for (size_t index = 0; index < kProfileIds.size(); ++index) {
                if (route.profile != kProfileIds[index]) continue;
                if (seen[index]) throw BenchmarkError("duplicate canonical route profile");
                if (!route_matches_profile(route, kProfileIds[index])) {
                    throw BenchmarkError("payload exported an invalid canonical route for " +
                                         std::string(kProfileNames[index]));
                }
                discovery.by_profile[index] = route;
                seen[index] = true;
                discovery.profile_mask |= 1u << index;
                discovery.storage_dtype_mask |= 1u << (route.storage_dtype - 1u);
                discovery.result_dtype_mask |= 1u << (route.result_dtype - 1u);
            }
        }
        if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
            throw BenchmarkError("payload did not advertise all canonical precision profiles");
        }
        return discovery;
    }

    FrozenPrecisionCapabilities capabilities{};
    require_status(
        api.typed_capabilities(device, &capabilities), "precision_capabilities");
    if (capabilities.abi_version != kAbi11 || capabilities.backend_kind != kBackend) {
        throw BenchmarkError("typed precision capability identity does not match ROCm ABI 1.1");
    }
    if (capabilities.profile_mask != 0x7u ||
        capabilities.storage_dtype_mask != (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64) ||
        capabilities.result_dtype_mask != (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64)) {
        throw BenchmarkError("typed precision capability masks do not cover the exact ROCm fp32/mixed/fp64 contract");
    }
    discovery.profile_mask = capabilities.profile_mask;
    discovery.storage_dtype_mask = capabilities.storage_dtype_mask;
    discovery.result_dtype_mask = capabilities.result_dtype_mask;
    discovery.flags = capabilities.flags;
    discovery.route_synthesized = true;
    for (size_t index = 0; index < kProfileIds.size(); ++index) {
        const uint32_t profile_bit = 1u << index;
        if ((capabilities.profile_mask & profile_bit) == 0) {
            throw BenchmarkError("typed payload does not advertise " +
                                 std::string(kProfileNames[index]));
        }
        const GafimeNumericRoute route = synthetic_route(kProfileIds[index]);
        const uint32_t storage_bit = 1u << (route.storage_dtype - 1u);
        const uint32_t result_bit = 1u << (route.result_dtype - 1u);
        if ((capabilities.storage_dtype_mask & storage_bit) == 0 ||
            (capabilities.result_dtype_mask & result_bit) == 0) {
            throw BenchmarkError("typed capability dtype masks omit " +
                                 std::string(kProfileNames[index]));
        }
        discovery.by_profile[index] = route;
    }
    return discovery;
}

GafimeConstBufferView const_view(uint32_t dtype, const void* data, uint64_t count) {
    GafimeConstBufferView view{};
    view.abi_version = kAbi11;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_count = count;
    view.byte_length = count * dtype_size(dtype);
    view.byte_stride = dtype_size(dtype);
    return view;
}

GafimeMutableBufferView mutable_view(uint32_t dtype, void* data, uint64_t count) {
    GafimeMutableBufferView view{};
    view.abi_version = kAbi11;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_capacity = count;
    view.byte_length = count * dtype_size(dtype);
    view.byte_stride = dtype_size(dtype);
    return view;
}

struct Dataset {
    std::vector<double> features;
    std::vector<double> target;
    std::vector<float> features_f32;
    std::vector<float> target_f32;
};

Dataset make_dataset(const Options& options) {
    Dataset dataset;
    const uint64_t count = options.rows * static_cast<uint64_t>(options.features);
    dataset.features.resize(static_cast<size_t>(count));
    dataset.target.resize(static_cast<size_t>(options.rows));
    dataset.features_f32.resize(static_cast<size_t>(count));
    dataset.target_f32.resize(static_cast<size_t>(options.rows));
    std::mt19937_64 generator(options.dataset_seed);
    const double phase = static_cast<double>(generator() & 0xffffu) / 65536.0;
    for (uint64_t row = 0; row < options.rows; ++row) {
        const double row_index = static_cast<double>(row + 1);
        double target = 0.35 * std::sin(phase + static_cast<double>(row) * 0.017);
        for (uint32_t column = 0; column < options.features; ++column) {
            const double frequency = static_cast<double>(column + 1);
            const double value =
                std::sin(phase + row_index * (0.009 + frequency * 0.0013)) +
                0.31 * std::cos(row_index * (0.004 + frequency * 0.0007)) +
                0.013 * static_cast<double>(column) +
                0.000001 * static_cast<double>((row + column * 17) % 97);
            dataset.features[static_cast<size_t>(row) * options.features + column] = value;
            dataset.features_f32[static_cast<size_t>(row) * options.features + column] =
                static_cast<float>(value);
            target += (0.21 / frequency) * value;
        }
        target += 0.07 * std::sin(row_index * 0.031 + phase * 0.7);
        dataset.target[static_cast<size_t>(row)] = target;
        dataset.target_f32[static_cast<size_t>(row)] = static_cast<float>(target);
    }
    return dataset;
}

std::string rocm_dataset_identity_json(const Options& options, const Dataset& dataset) {
    std::ostringstream stream;
    const std::string feature_names = [&]() {
        std::ostringstream names;
        for (uint32_t column = 0; column < options.features; ++column) {
            if (column != 0) names << '\0';
            names << 'x' << column;
        }
        return names.str();
    }();
    stream << "{\"algorithm\":\"gafime.rocm.native_timing.dataset.v2\""
           << ",\"input_policy\":" << json_escape(options.input_policy)
           << ",\"generator\":\"make_dataset.v1\""
           << ",\"dataset_seed\":" << options.dataset_seed
           << ",\"matrix_sha256\":" << json_escape(sha256_vector(dataset.features))
           << ",\"target_sha256\":" << json_escape(sha256_vector(dataset.target))
           << ",\"feature_names_sha256\":" << json_escape(
               sha256_bytes(feature_names.data(), feature_names.size()))
           << ",\"matrix_shape\":[" << options.rows << ',' << options.features << ']'
           << ",\"target_shape\":[" << options.rows << ']'
           << ",\"matrix_dtype\":"
           << json_escape(options.input_policy == "common-f64" ? "float64" : "profile-native")
           << ",\"target_dtype\":"
           << json_escape(options.input_policy == "common-f64" ? "float64" : "profile-native")
           << ",\"native_fp32_matrix_sha256\":"
           << json_escape(sha256_vector(dataset.features_f32))
           << ",\"native_fp32_target_sha256\":"
           << json_escape(sha256_vector(dataset.target_f32))
           << ",\"profile_execution_inputs\":{"
           << "\"fp32\":{\"matrix_dtype\":\"float32\",\"target_dtype\":\"float32\","
           << "\"matrix_sha256\":" << json_escape(sha256_vector(dataset.features_f32))
           << ",\"target_sha256\":" << json_escape(sha256_vector(dataset.target_f32)) << "},"
           << "\"mixed\":{\"matrix_dtype\":\"float32\",\"target_dtype\":\"float32\","
           << "\"matrix_sha256\":" << json_escape(sha256_vector(dataset.features_f32))
           << ",\"target_sha256\":" << json_escape(sha256_vector(dataset.target_f32)) << "},"
           << "\"fp64\":{\"matrix_dtype\":\"float64\",\"target_dtype\":\"float64\","
           << "\"matrix_sha256\":" << json_escape(sha256_vector(dataset.features))
           << ",\"target_sha256\":" << json_escape(sha256_vector(dataset.target)) << "}}"
           << ",\"layout\":\"row_major\"}";
    return stream.str();
}

void materialize_candidates(
    uint32_t features,
    uint32_t arity,
    uint32_t requested,
    std::vector<uint32_t>& output
) {
    output.assign(static_cast<size_t>(arity) * requested, UINT32_MAX);
    uint32_t written = 0;
    std::array<uint32_t, 5> current{};
    const auto visit = [&](auto&& self, uint32_t start, uint32_t depth) -> void {
        if (written >= requested) return;
        if (depth == arity) {
            std::copy_n(current.begin(), arity,
                        output.begin() + static_cast<size_t>(written) * arity);
            ++written;
            return;
        }
        const uint32_t remaining = arity - depth;
        for (uint32_t candidate = start; candidate + remaining <= features; ++candidate) {
            current[depth] = candidate;
            self(self, candidate + 1, depth + 1);
            if (written >= requested) return;
        }
    };
    visit(visit, 0, 0);
    if (written != requested) {
        throw BenchmarkError("candidates exceed combinations for requested arity/features");
    }
}

struct Protocol {
    std::vector<uint32_t> combos;
    std::array<uint32_t, 1> metrics{};
    GafimeShapeHint hint{};
    GafimeArityChunk chunk{};
    GafimeLaunchProtocol base{};
    GafimeNumericLaunchProtocol numeric{};
    FrozenPrecisionLaunchProtocol typed{};

    Protocol(
        const GafimeNumericRoute& route,
        const Options& options,
        const std::vector<uint32_t>& candidate_descriptors,
        uint32_t metric,
        uint32_t top_k,
        uint64_t generation
    ) : combos(candidate_descriptors) {
        metrics[0] = metric;
        hint.vendor_hint = options.mi_bins;
        chunk.arity = options.arity;
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.metric_mask = 1u << (metric - 1u);
        chunk.shape_hint_index = 0;
        chunk.combo_row_offset = 0;
        chunk.combo_count = options.candidates;
        chunk.local_chunk_id = 0;
        chunk.descriptor_offset = 0;
        chunk.descriptor_count = options.candidates;

        base.abi_version = GAFIME_ABI_VERSION;
        base.backend_kind = kBackend;
        base.flags = GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
        base.max_arity = options.arity;
        base.n_samples = options.rows;
        base.n_features = options.features;
        base.family_count = 1;
        base.combo_indices = {combos.data(), combos.size()};
        base.metric_ids = {metrics.data(), metrics.size()};
        base.chunks = &chunk;
        base.chunk_count = 1;
        base.shape_hints = &hint;
        base.shape_hint_count = 1;
        base.rank.top_k = top_k;
        base.rank.primary_metric = metric;
        base.rank.descending = 1;
        base.rank.include_ties = 0;
        base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = generation;

        numeric.abi_version = kAbi11;
        numeric.struct_size = sizeof(numeric);
        numeric.route = route;
        numeric.base = &base;
        typed.abi_version = kAbi11;
        typed.profile = route.profile;
        typed.base = &base;
    }
};

struct Result {
    std::vector<uint32_t> combos;
    std::vector<float> values_f32;
    std::vector<double> values_f64;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> families;
    std::vector<uint64_t> candidate_ids;
    std::vector<uint32_t> flags;
    GafimeNumericResultTable table{};
    GafimeResultTable typed_f32{};
    FrozenResultTableF64 typed_f64{};

    Result(
        const GafimeNumericRoute& route,
        uint32_t max_arity,
        uint64_t capacity,
        bool typed_surface
    ) {
        combos.resize(static_cast<size_t>(capacity) * max_arity);
        ranks.resize(static_cast<size_t>(capacity));
        families.resize(static_cast<size_t>(capacity));
        candidate_ids.resize(static_cast<size_t>(capacity));
        flags.resize(static_cast<size_t>(capacity));
        if (route.result_dtype == GAFIME_DTYPE_F32) {
            values_f32.resize(static_cast<size_t>(capacity));
        } else {
            values_f64.resize(static_cast<size_t>(capacity));
        }
        if (typed_surface) {
            if (route.result_dtype == GAFIME_DTYPE_F32) {
                typed_f32.abi_version = GAFIME_ABI_VERSION;
                typed_f32.max_arity = max_arity;
                typed_f32.metric_count = 1;
                typed_f32.capacity = capacity;
                typed_f32.combo_indices = combos.data();
                typed_f32.metric_values = values_f32.data();
                typed_f32.ranks = ranks.data();
                typed_f32.families = families.data();
                typed_f32.candidate_ids = candidate_ids.data();
                typed_f32.row_flags = flags.data();
            } else {
                typed_f64.abi_version = kAbi11;
                typed_f64.max_arity = max_arity;
                typed_f64.metric_count = 1;
                typed_f64.capacity = capacity;
                typed_f64.combo_indices = combos.data();
                typed_f64.metric_values = values_f64.data();
                typed_f64.ranks = ranks.data();
                typed_f64.families = families.data();
                typed_f64.candidate_ids = candidate_ids.data();
                typed_f64.row_flags = flags.data();
            }
        } else {
            table.abi_version = kAbi11;
            table.struct_size = sizeof(table);
            table.max_arity = max_arity;
            table.metric_count = 1;
            table.capacity = capacity;
            table.combo_indices = combos.data();
            table.metric_values = mutable_view(
                route.result_dtype,
                route.result_dtype == GAFIME_DTYPE_F32
                    ? static_cast<void*>(values_f32.data())
                    : static_cast<void*>(values_f64.data()),
                capacity);
            table.ranks = ranks.data();
            table.families = families.data();
            table.candidate_ids = candidate_ids.data();
            table.row_flags = flags.data();
        }
        reset();
    }

    void reset() {
        table.row_count = 0;
        table.flags = 0;
        typed_f32.row_count = 0;
        typed_f32.flags = 0;
        typed_f64.row_count = 0;
        typed_f64.flags = 0;
    }

    double first_value(uint32_t dtype) const {
        if (dtype == GAFIME_DTYPE_F32) return static_cast<double>(values_f32[0]);
        return values_f64[0];
    }

    void require_finite_values(uint32_t dtype, bool typed_surface) const {
        const uint64_t row_count = !typed_surface
            ? table.row_count
            : dtype == GAFIME_DTYPE_F32 ? typed_f32.row_count : typed_f64.row_count;
        const uint64_t capacity = dtype == GAFIME_DTYPE_F32
            ? values_f32.size()
            : values_f64.size();
        if (row_count == 0 || row_count > capacity) {
            throw BenchmarkError("payload result row count is empty or exceeds capacity");
        }
        if (dtype == GAFIME_DTYPE_F32) {
            for (uint64_t index = 0; index < row_count; ++index) {
                if (!std::isfinite(values_f32[static_cast<size_t>(index)])) {
                    throw BenchmarkError("payload returned a non-finite fp32 metric value");
                }
            }
            return;
        }
        for (uint64_t index = 0; index < row_count; ++index) {
            if (!std::isfinite(values_f64[static_cast<size_t>(index)])) {
                throw BenchmarkError("payload returned a non-finite fp64 metric value");
            }
        }
    }
};

struct SampledValues {
    // samples_us is normalized to one operation; raw_samples_us retains the
    // complete calibrated region measured by the selected clock.
    std::vector<double> samples_us;
    std::vector<double> raw_samples_us;
    std::vector<uint32_t> loop_counts_per_sample;
    std::string calibration_key;
    uint32_t loop_count_per_sample = 1;
    uint32_t precondition_iterations = 0;
    double precondition_duration_us = 0.0;
    uint32_t precondition_max_batch_iterations = 1;
    std::string precondition_clock = "host_steady_clock";
};

struct EventSamples {
    SampledValues gpu;
    SampledValues host;
};

struct TimingCalibrationCache {
    std::map<std::string, uint32_t> loop_counts;
    bool immutable_plan = false;
    std::string plan_path;
    std::string plan_semantic_sha256;
    std::string plan_file_sha256;
    std::set<std::string> plan_lookups;
};

struct CalibrationPrepassSummary {
    bool performed = false;
    std::vector<std::string> profile_order;
    size_t discarded_record_count = 0;
    size_t discarded_sample_count = 0;
    size_t calibrated_key_count = 0;
};

struct PreconditionStats {
    uint32_t iterations = 0;
    double duration_us = 0.0;
    uint32_t max_batch_iterations = 1;
    std::string clock = "host_steady_clock";
};

std::string timing_calibration_key(
    std::string_view lane,
    std::string_view profile,
    std::string_view operation,
    std::string_view metric
) {
    return std::string(lane) + "/" + std::string(profile) + "/" +
        std::string(operation) + "/" + std::string(metric);
}

template <typename Measure>
uint32_t fixed_loop_count(
    TimingCalibrationCache& cache,
    std::string_view key,
    Measure&& measure
) {
    if (key.empty()) throw BenchmarkError("native timing calibration key must not be empty");
    if (cache.immutable_plan) {
        const auto planned = cache.loop_counts.find(std::string(key));
        if (planned == cache.loop_counts.end()) {
            throw BenchmarkError("immutable native loop plan has no exact key: " + std::string(key));
        }
        cache.plan_lookups.insert(std::string(key));
        return planned->second;
    }
    const auto existing = cache.loop_counts.find(std::string(key));
    if (existing != cache.loop_counts.end()) return existing->second;
    uint32_t loop_count = 1;
    double calibration_us = measure(loop_count);
    while (calibration_us < kSampleRegionCalibrationTargetUs &&
           loop_count < kMaxLoopCount) {
        loop_count = loop_count > kMaxLoopCount / 2 ? kMaxLoopCount : loop_count * 2;
        calibration_us = measure(loop_count);
    }
    cache.loop_counts.emplace(std::string(key), loop_count);
    return loop_count;
}

void validate_loop_plan_consumed(const TimingCalibrationCache& cache) {
    if (!cache.immutable_plan) return;
    if (cache.plan_lookups.size() != cache.loop_counts.size()) {
        for (const auto& [key, count] : cache.loop_counts) {
            static_cast<void>(count);
            if (!cache.plan_lookups.contains(key)) {
                throw BenchmarkError("immutable native loop plan contains unused/out-of-scope key: " + key);
            }
        }
        throw BenchmarkError("immutable native loop plan lookup coverage mismatch");
    }
}

template <typename Function>
PreconditionStats precondition_host(
    uint32_t minimum_iterations,
    Function&& function,
    bool synchronize_device
) {
    if (synchronize_device) {
        require_hip(hipDeviceSynchronize(), "host precondition initial synchronize");
    }
    const auto start = Clock::now();
    uint32_t iterations = 0;
    double duration_us = 0.0;
    do {
        if (synchronize_device) {
            require_hip(hipDeviceSynchronize(), "host precondition pre-synchronize");
        }
        require_status(function(), "host timing precondition");
        if (synchronize_device) {
            require_hip(hipDeviceSynchronize(), "host precondition synchronize");
        }
        ++iterations;
        duration_us = std::max(
            1.0e-6,
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
    } while (iterations < kMaxHostPreconditionIterations &&
             (iterations < minimum_iterations ||
              duration_us < kPerRecordUntimedPreconditionMinUs));
    if (iterations >= kMaxHostPreconditionIterations &&
        (iterations < minimum_iterations ||
         duration_us < kPerRecordUntimedPreconditionMinUs)) {
        throw BenchmarkError("host precondition exceeded its bounded iteration budget");
    }
    return {
        iterations,
        duration_us,
        1,
        synchronize_device ? "host_steady_clock_with_hip_synchronization"
                           : "host_steady_clock",
    };
}

template <typename Function>
SampledValues host_samples(
    uint32_t warmups,
    uint32_t repeats,
    TimingCalibrationCache& calibration_cache,
    std::string_view calibration_key,
    Function function,
    bool synchronize_device_precondition = false
) {
    const PreconditionStats precondition = precondition_host(
        std::max(warmups, kPerRecordUntimedSameCellPreconditions),
        function,
        synchronize_device_precondition);
    auto measure = [&](uint32_t loop_count) {
        const auto start = Clock::now();
        for (uint32_t loop = 0; loop < loop_count; ++loop) {
            require_status(function(), "host timing calibration/sample");
        }
        const auto stop = Clock::now();
        const uint64_t nanoseconds = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        return std::max(1.0e-6, static_cast<double>(nanoseconds) / 1000.0);
    };
    const uint32_t loop_count = fixed_loop_count(
        calibration_cache, calibration_key, measure);

    SampledValues result;
    result.calibration_key = std::string(calibration_key);
    result.loop_count_per_sample = loop_count;
    result.precondition_iterations = precondition.iterations;
    result.precondition_duration_us = precondition.duration_us;
    result.precondition_max_batch_iterations = precondition.max_batch_iterations;
    result.precondition_clock = precondition.clock;
    result.samples_us.reserve(repeats);
    result.raw_samples_us.reserve(repeats);
    result.loop_counts_per_sample.reserve(repeats);
    for (uint32_t index = 0; index < repeats; ++index) {
        const double raw_us = measure(loop_count);
        result.loop_counts_per_sample.push_back(loop_count);
        result.raw_samples_us.push_back(raw_us);
        result.samples_us.push_back(std::max(1.0e-6, raw_us / loop_count));
    }
    return result;
}

template <typename Prepare, typename Execute>
PreconditionStats precondition_hip_events(
    uint32_t minimum_iterations,
    Prepare& prepare,
    Execute& execute,
    bool prepare_each_execute,
    hipStream_t timing_stream
) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    require_hip(hipEventCreateWithFlags(&start, hipEventDefault),
                "hipEventCreate(precondition_start)");
    require_hip(hipEventCreateWithFlags(&stop, hipEventDefault),
                "hipEventCreate(precondition_stop)");
    uint32_t iterations = 0;
    double duration_us = 0.0;
    uint32_t batch_size = std::min(
        minimum_iterations, kMaxPreconditionBatchIterations);
    uint32_t max_batch_size = 0;
    auto destroy_events = [&]() {
        if (stop != nullptr) static_cast<void>(hipEventDestroy(stop));
        if (start != nullptr) static_cast<void>(hipEventDestroy(start));
    };
    try {
        require_hip(hipDeviceSynchronize(),
                    "hipDeviceSynchronize(precondition initial)");
        while (iterations < kMaxDevicePreconditionIterations &&
               (iterations < minimum_iterations ||
                duration_us < kPerRecordUntimedPreconditionMinUs)) {
            const uint32_t current_batch = std::min(
                batch_size, kMaxDevicePreconditionIterations - iterations);
            if (current_batch == 0) break;
            if (!prepare_each_execute) {
                require_status(prepare(), "HIP event precondition preparation");
            }
            require_hip(hipEventRecord(start, timing_stream),
                        "hipEventRecord(precondition_start)");
            for (uint32_t index = 0; index < current_batch; ++index) {
                if (prepare_each_execute) {
                    require_status(
                        prepare(), "HIP event precondition per-execute preparation");
                }
                require_status(execute(), "HIP event precondition execute");
            }
            require_hip(hipEventRecord(stop, timing_stream),
                        "hipEventRecord(precondition_stop)");
            require_hip(hipEventSynchronize(stop),
                        "hipEventSynchronize(precondition_stop)");
            float milliseconds = 0.0f;
            require_hip(
                hipEventElapsedTime(&milliseconds, start, stop),
                "hipEventElapsedTime(precondition)");
            const double batch_duration_us = std::max(
                1.0e-6, static_cast<double>(milliseconds) * 1000.0);
            duration_us += batch_duration_us;
            iterations += current_batch;
            max_batch_size = std::max(max_batch_size, current_batch);
            if (batch_duration_us < kPreconditionDeviceBatchTargetUs &&
                batch_size < kMaxPreconditionBatchIterations) {
                batch_size = std::min(
                    kMaxPreconditionBatchIterations, batch_size * 2u);
            }
        }
        if (iterations >= kMaxDevicePreconditionIterations &&
            (iterations < minimum_iterations ||
             duration_us < kPerRecordUntimedPreconditionMinUs)) {
            throw BenchmarkError(
                "HIP event precondition exceeded its bounded iteration budget");
        }
        destroy_events();
        return {
            iterations,
            duration_us,
            max_batch_size,
            timing_stream == nullptr ? "hip_event_default_stream" :
                "hip_event_timing_stream",
        };
    } catch (...) {
        destroy_events();
        throw;
    }
}

template <typename Prepare, typename Execute>
EventSamples event_samples(
    uint32_t warmups,
    uint32_t repeats,
    TimingCalibrationCache& calibration_cache,
    std::string_view calibration_key,
    Prepare prepare,
    Execute execute,
    bool prepare_each_execute = false,
    hipStream_t timing_stream = nullptr
) {
    const PreconditionStats precondition = precondition_hip_events(
        std::max(warmups, kPerRecordUntimedSameCellPreconditions),
        prepare,
        execute,
        prepare_each_execute,
        timing_stream);
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    require_hip(hipEventCreateWithFlags(&start, hipEventDefault), "hipEventCreate(start)");
    require_hip(hipEventCreateWithFlags(&stop, hipEventDefault), "hipEventCreate(stop)");
    auto destroy_events = [&]() {
        if (start != nullptr) static_cast<void>(hipEventDestroy(start));
        if (stop != nullptr) static_cast<void>(hipEventDestroy(stop));
    };
    try {
        auto one = [&](uint32_t loop_count) -> std::pair<double, double> {
            if (!prepare_each_execute) {
                require_status(prepare(), "event timing preparation");
            }
            require_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(before event)");
            require_hip(hipEventRecord(start, timing_stream), "hipEventRecord(start)");
            const auto host_start = Clock::now();
            for (uint32_t loop = 0; loop < loop_count; ++loop) {
                if (prepare_each_execute) {
                    require_status(prepare(), "event timing per-execute preparation");
                }
                require_status(execute(), "event timing execute");
            }
            const auto host_stop = Clock::now();
            require_hip(hipEventRecord(stop, timing_stream), "hipEventRecord(stop)");
            require_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
            float milliseconds = 0.0f;
            require_hip(hipEventElapsedTime(&milliseconds, start, stop),
                        "hipEventElapsedTime");
            const uint64_t nanoseconds = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    host_stop - host_start).count());
            return {
                std::max(1.0e-6, static_cast<double>(milliseconds) * 1000.0),
                std::max(1.0e-6, static_cast<double>(nanoseconds) / 1000.0),
            };
        };
        const uint32_t loop_count = fixed_loop_count(
            calibration_cache,
            calibration_key,
            [&](uint32_t count) { return one(count).first; });
        EventSamples samples;
        samples.gpu.calibration_key = std::string(calibration_key);
        samples.host.calibration_key = std::string(calibration_key);
        samples.gpu.loop_count_per_sample = loop_count;
        samples.host.loop_count_per_sample = loop_count;
        samples.gpu.precondition_iterations = precondition.iterations;
        samples.host.precondition_iterations = precondition.iterations;
        samples.gpu.precondition_duration_us = precondition.duration_us;
        samples.host.precondition_duration_us = precondition.duration_us;
        samples.gpu.precondition_max_batch_iterations =
            precondition.max_batch_iterations;
        samples.host.precondition_max_batch_iterations =
            precondition.max_batch_iterations;
        samples.gpu.precondition_clock = precondition.clock;
        samples.host.precondition_clock = precondition.clock;
        samples.gpu.samples_us.reserve(repeats);
        samples.gpu.raw_samples_us.reserve(repeats);
        samples.gpu.loop_counts_per_sample.reserve(repeats);
        samples.host.samples_us.reserve(repeats);
        samples.host.raw_samples_us.reserve(repeats);
        samples.host.loop_counts_per_sample.reserve(repeats);
        for (uint32_t index = 0; index < repeats; ++index) {
            const auto sample = one(loop_count);
            samples.gpu.loop_counts_per_sample.push_back(loop_count);
            samples.host.loop_counts_per_sample.push_back(loop_count);
            samples.gpu.raw_samples_us.push_back(sample.first);
            samples.gpu.samples_us.push_back(
                std::max(1.0e-6, sample.first / loop_count));
            samples.host.raw_samples_us.push_back(sample.second);
            samples.host.samples_us.push_back(
                std::max(1.0e-6, sample.second / loop_count));
        }
        destroy_events();
        return samples;
    } catch (...) {
        destroy_events();
        throw;
    }
}

struct Record {
    std::string profile;
    uint32_t order_index = 0;
    std::vector<std::string> profile_order;
    std::string operation;
    std::string metric;
    std::string calibration_key;
    std::vector<double> samples;
    std::vector<double> raw_samples;
    std::vector<uint32_t> loop_counts_per_sample;
    uint32_t loop_count_per_sample = 1;
    uint32_t precondition_iterations = 0;
    double precondition_duration_us = 0.0;
    uint32_t precondition_max_batch_iterations = 1;
    std::string precondition_clock = "host_steady_clock";
    std::string clock;
    std::string synchronization;
    std::string note;
    std::string timing_mode = "default";
    std::string evidence_lane = "canonical_payload_api";
    std::string comparability = "within_abi_surface_only";
};

void append_record(
    std::vector<Record>& records,
    const std::string& profile,
    uint32_t order_index,
    const std::string& operation,
    const std::string& metric,
    SampledValues samples,
    const std::string& clock,
    const std::string& synchronization,
    const std::string& note,
    const std::string& timing_mode = "default",
    const std::string& evidence_lane = "canonical_payload_api",
    const std::string& comparability = "within_abi_surface_only"
) {
    if (samples.samples_us.empty()) {
        throw BenchmarkError("timing record has no samples: " + operation);
    }
    records.push_back(Record{
        profile, order_index, {}, operation, metric, std::move(samples.calibration_key),
        std::move(samples.samples_us),
        std::move(samples.raw_samples_us), std::move(samples.loop_counts_per_sample),
        samples.loop_count_per_sample, samples.precondition_iterations,
        samples.precondition_duration_us, samples.precondition_max_batch_iterations,
        std::move(samples.precondition_clock),
        clock, synchronization, note, timing_mode, evidence_lane, comparability,
    });
}

double percentile50(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1u) != 0) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(values.size() - 1, lower + 1);
    const double weight = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * weight;
}

double median_absolute_deviation(const std::vector<double>& values) {
    const double center = percentile50(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) deviations.push_back(std::abs(value - center));
    return percentile50(std::move(deviations));
}

uint64_t stable_seed(const Record& record) {
    uint64_t hash = kBootstrapSeed;
    const auto mix = [&](std::string_view value) {
        for (const unsigned char character : value) {
            hash ^= static_cast<uint64_t>(character);
            hash *= 1099511628211ULL;
        }
    };
    mix(record.profile);
    mix(record.operation);
    mix(record.metric);
    hash ^= record.order_index;
    hash *= 1099511628211ULL;
    return hash;
}

std::array<double, 2> bootstrap_median_ci(
    const std::vector<double>& values,
    uint64_t seed
) {
    if (values.empty()) return {NAN, NAN};
    std::mt19937_64 generator(seed);
    std::vector<double> bootstrap;
    bootstrap.reserve(kBootstrapResamples);
    std::vector<double> resample(values.size());
    for (uint32_t iteration = 0; iteration < kBootstrapResamples; ++iteration) {
        for (double& value : resample) {
            value = values[static_cast<size_t>(generator() % values.size())];
        }
        bootstrap.push_back(percentile50(resample));
    }
    const double lower = percentile(bootstrap, 0.025);
    const double upper = percentile(std::move(bootstrap), 0.975);
    return {lower, upper};
}

void append_samples_json(std::ostringstream& stream, const std::vector<double>& samples) {
    stream << '[' << std::setprecision(17);
    for (size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) stream << ',';
        stream << samples[index];
    }
    stream << ']';
}

void append_string_array(
    std::ostringstream& stream, const std::vector<std::string>& values);

void append_record_json(std::ostringstream& stream, const Record& record) {
    const double mean = std::accumulate(record.samples.begin(), record.samples.end(), 0.0) /
        static_cast<double>(record.samples.size());
    const auto minmax = std::minmax_element(record.samples.begin(), record.samples.end());
    const std::vector<double>& raw_samples = record.raw_samples.empty()
        ? record.samples : record.raw_samples;
    const auto raw_minmax = std::minmax_element(raw_samples.begin(), raw_samples.end());
    const auto ci = bootstrap_median_ci(record.samples, stable_seed(record));
    stream << "{\"profile\":" << json_escape(record.profile)
           << ",\"order_index\":" << record.order_index
           << ",\"profile_order\":";
    append_string_array(stream, record.profile_order);
    stream
           << ",\"operation\":" << json_escape(record.operation)
           << ",\"metric\":" << json_escape(record.metric)
           << ",\"calibration_key\":" << json_escape(record.calibration_key)
           << ",\"timing_mode\":" << json_escape(record.timing_mode)
           << ",\"evidence_lane\":"
           << json_escape(record.evidence_lane)
           << ",\"comparability\":"
           << json_escape(record.comparability)
           << ",\"clock\":" << json_escape(record.clock)
           << ",\"synchronization\":" << json_escape(record.synchronization)
           << ",\"samples_us\":";
    append_samples_json(stream, record.samples);
    stream << ",\"raw_samples_us\":";
    append_samples_json(stream, raw_samples);
    stream << ",\"median_us\":" << std::setprecision(17) << percentile50(record.samples)
           << ",\"mad_us\":" << median_absolute_deviation(record.samples)
           << ",\"p05_us\":" << percentile(record.samples, 0.05)
           << ",\"p95_us\":" << percentile(record.samples, 0.95)
           << ",\"bootstrap_median_95_ci_us\":[" << ci[0] << ',' << ci[1] << ']'
           << ",\"mean_us\":" << mean
           << ",\"min_us\":" << *minmax.first
           << ",\"max_us\":" << *minmax.second
           << ",\"sample_count\":" << record.samples.size()
           << ",\"loop_count_per_sample\":" << record.loop_count_per_sample
           << ",\"loop_counts_per_sample\":[";
    for (size_t index = 0; index < record.loop_counts_per_sample.size(); ++index) {
        if (index != 0) stream << ',';
        stream << record.loop_counts_per_sample[index];
    }
    stream << ']'
           << ",\"precondition_iterations\":" << record.precondition_iterations
           << ",\"precondition_duration_us\":"
           << record.precondition_duration_us
           << ",\"precondition_max_batch_iterations\":"
           << record.precondition_max_batch_iterations
           << ",\"precondition_clock\":"
           << json_escape(record.precondition_clock)
           << ",\"sample_region_target_us\":" << kSampleRegionTargetUs
           << ",\"sample_region_min_observed_us\":" << *raw_minmax.first
           << ",\"sample_region_target_met\":"
           << (*raw_minmax.first >= kSampleRegionTargetUs ? "true" : "false")
           << ",\"bootstrap_resamples\":" << kBootstrapResamples
           << ",\"bootstrap_seed\":" << stable_seed(record);
    if (!record.note.empty()) {
        stream << ",\"note\":" << json_escape(record.note);
    }
    stream << '}';
}

struct AffinityInfo {
    std::vector<int> cpus;
    int current_cpu = -1;
};

AffinityInfo affinity_info() {
    cpu_set_t set;
    CPU_ZERO(&set);
    AffinityInfo info;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) info.cpus.push_back(cpu);
        }
    }
    info.current_cpu = sched_getcpu();
    return info;
}

void append_affinity_json(std::ostringstream& stream, const AffinityInfo& info) {
    stream << "{\"current_cpu\":" << info.current_cpu << ",\"allowed_cpus\":[";
    for (size_t index = 0; index < info.cpus.size(); ++index) {
        if (index != 0) stream << ',';
        stream << info.cpus[index];
    }
    stream << "]}";
}

void append_environment_json(std::ostringstream& stream) {
    static constexpr std::array<const char*, 20> keys = {
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "ROCR_MEM_THRASH_LIMIT",
        "HIP_FORCE_DEV_KERNARG", "HIP_LAUNCH_BLOCKING", "GAFIME_ROCM_V1_LIB",
        "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "VIRTUAL_ENV", "RAYON_NUM_THREADS",
        "SHELL", "HOSTNAME", "TERM", "GAFIME_WHEEL_PATH",
        "GAFIME_NATIVE_RUNNER_INVOCATION_ID", "GAFIME_NATIVE_RUNNER_PID",
    };
    stream << '{';
    bool first = true;
    for (const char* key : keys) {
        const char* value = std::getenv(key);
        if (value == nullptr) continue;
        if (!first) stream << ',';
        first = false;
        stream << json_escape(key) << ':' << json_escape(value);
    }
    stream << '}';
}

uint64_t required_runner_pid() {
    const std::string value = environment_value("GAFIME_NATIVE_RUNNER_PID");
    if (value.empty()) {
        throw BenchmarkError("GAFIME_NATIVE_RUNNER_PID is required for process attestation");
    }
    size_t end = 0;
    uint64_t result = 0;
    try {
        result = std::stoull(value, &end, 10);
    } catch (...) {
        throw BenchmarkError("GAFIME_NATIVE_RUNNER_PID is invalid");
    }
    if (end != value.size() || result == 0) {
        throw BenchmarkError("GAFIME_NATIVE_RUNNER_PID is invalid");
    }
    return result;
}

std::string required_runner_invocation_id() {
    const std::string value = environment_value("GAFIME_NATIVE_RUNNER_INVOCATION_ID");
    if (value.size() != 32 ||
        !std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isxdigit(character) != 0;
        })) {
        throw BenchmarkError(
            "GAFIME_NATIVE_RUNNER_INVOCATION_ID must be a 32-digit hexadecimal nonce");
    }
    return value;
}

void append_device_json(std::ostringstream& stream, uint32_t device) {
    hipDeviceProp_t properties{};
    require_hip(hipGetDeviceProperties(&properties, static_cast<int>(device)),
                "hipGetDeviceProperties");
    int runtime_version = 0;
    int driver_version = 0;
    require_hip(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    require_hip(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    stream << "{\"id\":" << device
           << ",\"name\":" << json_escape(properties.name)
           << ",\"gcn_arch\":" << json_escape(properties.gcnArchName)
           << ",\"runtime_version\":" << runtime_version
           << ",\"driver_version\":" << driver_version
           << ",\"multiprocessors\":" << properties.multiProcessorCount
           << ",\"warp_size\":" << properties.warpSize
           << ",\"global_memory_bytes\":" << properties.totalGlobalMem
           << ",\"clock_rate_khz\":" << properties.clockRate
           << ",\"memory_clock_rate_khz\":" << properties.memoryClockRate
           << ",\"compute_major\":" << properties.major
           << ",\"compute_minor\":" << properties.minor << '}';
}

std::string profile_name(uint32_t profile) {
    for (size_t index = 0; index < kProfileIds.size(); ++index) {
        if (kProfileIds[index] == profile) return std::string(kProfileNames[index]);
    }
    throw BenchmarkError("unknown advertised precision profile");
}

struct ProfileData {
    std::vector<float> features_f32;
    std::vector<float> target_f32;
    std::vector<double> features_f64;
    std::vector<double> target_f64;
};

ProfileData convert_dataset(
    const Dataset& source,
    const GafimeNumericRoute& route,
    const std::string& input_policy,
    uint64_t rows,
    uint32_t features
) {
    ProfileData converted;
    if (route.storage_dtype == GAFIME_DTYPE_F32) {
        if (input_policy == "native") {
            converted.features_f32 = source.features_f32;
            converted.target_f32 = source.target_f32;
        } else {
            converted.features_f32.resize(source.features.size());
            converted.target_f32.resize(source.target.size());
            for (size_t index = 0; index < source.features.size(); ++index) {
                converted.features_f32[index] = static_cast<float>(source.features[index]);
            }
            for (size_t index = 0; index < source.target.size(); ++index) {
                converted.target_f32[index] = static_cast<float>(source.target[index]);
            }
        }
    } else {
        converted.features_f64 = source.features;
        converted.target_f64 = source.target;
    }
    (void)rows;
    (void)features;
    return converted;
}

GafimeNumericMatrixDesc matrix_desc(
    const GafimeNumericRoute& route,
    const Options& options
) {
    GafimeNumericMatrixDesc desc{};
    desc.abi_version = kAbi11;
    desc.struct_size = sizeof(desc);
    desc.route = route;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = options.rows;
    desc.cols = options.features;
    desc.row_stride = options.features;
    desc.bytes = options.rows * static_cast<uint64_t>(options.features) * dtype_size(route.storage_dtype);
    return desc;
}

FrozenPrecisionMatrixDesc typed_matrix_desc(
    const GafimeNumericRoute& route,
    const Options& options
) {
    FrozenPrecisionMatrixDesc desc{};
    desc.abi_version = kAbi11;
    desc.profile = route.profile;
    desc.dtype = route.storage_dtype;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = options.rows;
    desc.cols = options.features;
    desc.row_stride = options.features;
    desc.bytes = options.rows * static_cast<uint64_t>(options.features) * dtype_size(route.storage_dtype);
    return desc;
}

int api_allocate(
    const Api& api,
    uint32_t device,
    const GafimeNumericRoute& route,
    const Options& options,
    GafimeGpuMatrix* matrix_out
) {
    if (api.typed()) {
        const FrozenPrecisionMatrixDesc desc = typed_matrix_desc(route, options);
        return api.typed_alloc(device, &desc, matrix_out);
    }
    const GafimeNumericMatrixDesc desc = matrix_desc(route, options);
    return api.generic_alloc(device, &desc, matrix_out);
}

int api_upload(
    const Api& api,
    GafimeGpuMatrix matrix,
    const GafimeNumericRoute& route,
    const GafimeConstBufferView& features,
    const GafimeConstBufferView& target,
    uint64_t rows,
    uint32_t columns
) {
    if (!api.typed()) {
        return api.generic_upload(matrix, &route, &features, &target, rows, columns);
    }
    if (route.storage_dtype == GAFIME_DTYPE_F32) {
        return api.typed_upload_f32(
            matrix,
            static_cast<const float*>(features.data),
            static_cast<const float*>(target.data),
            rows,
            columns);
    }
    return api.typed_upload_f64(
        matrix,
        static_cast<const double*>(features.data),
        static_cast<const double*>(target.data),
        rows,
        columns);
}

int api_update_target(
    const Api& api,
    GafimeGpuMatrix matrix,
    const GafimeNumericRoute& route,
    const GafimeConstBufferView& target,
    uint64_t rows
) {
    if (!api.typed()) return api.generic_update_target(matrix, &route, &target, rows);
    if (route.storage_dtype == GAFIME_DTYPE_F32) {
        return api.typed_update_target_f32(
            matrix, static_cast<const float*>(target.data), rows);
    }
    return api.typed_update_target_f64(
        matrix, static_cast<const double*>(target.data), rows);
}

int api_execution_memory(
    const Api& api,
    GafimeGpuMatrix matrix,
    const Protocol& protocol,
    uint64_t* peak_bytes
) {
    if (!api.typed()) {
        return api.generic_execution_memory(matrix, &protocol.numeric, peak_bytes);
    }
    return api.typed_execution_memory(matrix, &protocol.typed, peak_bytes);
}

int api_execute(
    const Api& api,
    GafimeGpuMatrix matrix,
    const GafimeNumericRoute& route,
    const Protocol& protocol,
    Result& result
) {
    if (!api.typed()) return api.generic_execute(matrix, &protocol.numeric, &result.table);
    if (route.result_dtype == GAFIME_DTYPE_F32) {
        return api.typed_execute_f32(matrix, &protocol.typed, &result.typed_f32);
    }
    return api.typed_execute_f64(matrix, &protocol.typed, &result.typed_f64);
}

uint64_t result_buffer_bytes(const GafimeNumericRoute& route, const Options& options) {
    const uint64_t rows = options.candidates;
    const uint64_t combo_bytes = rows * static_cast<uint64_t>(options.arity) * sizeof(uint32_t);
    const uint64_t metric_bytes = rows * dtype_size(route.result_dtype);
    const uint64_t metadata_bytes = rows * (sizeof(uint32_t) + sizeof(uint32_t) +
                                            sizeof(uint64_t) + sizeof(uint32_t));
    if (combo_bytes > std::numeric_limits<uint64_t>::max() - metric_bytes - metadata_bytes) {
        throw BenchmarkError("result buffer byte count overflow");
    }
    return combo_bytes + metric_bytes + metadata_bytes;
}

EventSamples representative_d2h_samples(
    uint32_t warmups,
    uint32_t repeats,
    uint64_t bytes,
    TimingCalibrationCache& calibration_cache,
    std::string_view calibration_key
) {
    if (bytes == 0) throw BenchmarkError("representative result buffer is empty");
    void* device_buffer = nullptr;
    require_hip(hipMalloc(&device_buffer, bytes), "hipMalloc(representative result buffer)");
    std::vector<uint8_t> host_buffer(static_cast<size_t>(bytes));
    try {
        require_hip(hipMemset(device_buffer, 0x5a, bytes), "hipMemset(representative result buffer)");
        require_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(representative result buffer)");
        EventSamples samples = event_samples(
            warmups, repeats, calibration_cache, calibration_key,
            []() -> int { return GAFIME_STATUS_OK; },
            [&]() -> int {
                return hipMemcpy(host_buffer.data(), device_buffer, bytes, hipMemcpyDeviceToHost) ==
                        hipSuccess
                    ? GAFIME_STATUS_OK
                    : GAFIME_STATUS_DEVICE_ERROR;
            });
        require_hip(hipFree(device_buffer), "hipFree(representative result buffer)");
        return samples;
    } catch (...) {
        static_cast<void>(hipFree(device_buffer));
        throw;
    }
}

struct MatrixGuard {
    const Api* api = nullptr;
    GafimeGpuMatrix handle = nullptr;
    ~MatrixGuard() {
        if (api != nullptr && handle != nullptr) {
            static_cast<void>(api->free_matrix(handle));
        }
    }
};

void run_canonical_profile(
    const Options& options,
    const Api& api,
    const Dataset& source,
    const GafimeNumericRoute& route,
    const std::string& name,
    uint32_t order_index,
    TimingCalibrationCache& calibration_cache,
    std::vector<Record>& records
) {
    // Prepare caller-owned inputs and descriptors outside the canonical timing
    // records. Host-only decomposition is produced by a separate fresh process.
    ProfileData converted = convert_dataset(
        source, route, options.input_policy, options.rows, options.features);
    std::vector<uint32_t> descriptors;
    materialize_candidates(options.features, options.arity, options.candidates, descriptors);

    Protocol planning_protocol(
        route, options, descriptors, GAFIME_METRIC_PEARSON, 0,
        0x10000000ULL + static_cast<uint64_t>(order_index) * 16ULL + route.profile);
    planning_protocol.numeric.route = route;
    planning_protocol.numeric.base = &planning_protocol.base;
    planning_protocol.typed.profile = route.profile;
    planning_protocol.typed.base = &planning_protocol.base;
    planning_protocol.base.rank.top_k = 0;
    planning_protocol.base.rank.primary_metric = GAFIME_METRIC_PEARSON;

    const auto allocation = host_samples(
        options.warmups, options.repeats, calibration_cache,
        timing_calibration_key("host", name, "allocation", ""),
        [&]() -> int {
        GafimeGpuMatrix temporary = nullptr;
        const int status = api_allocate(api, options.device, route, options, &temporary);
        if (status != GAFIME_STATUS_OK) return status;
        return api.free_matrix(temporary);
    }, true);
    append_record(
        records, name, order_index, "allocation", "", std::move(allocation),
        "host_steady_clock",
        std::string("steady_clock around ") + api.abi_surface_name() +
            " matrix allocation plus teardown",
        "fresh device allocation/free pair per sample");

    MatrixGuard matrix{&api, nullptr};
    require_status(
        api_allocate(api, options.device, route, options, &matrix.handle),
        "matrix allocation");
    const auto upload = host_samples(
        options.warmups, options.repeats, calibration_cache,
        timing_calibration_key("host", name, "h2d_upload", ""),
        [&]() -> int {
        if (route.storage_dtype == GAFIME_DTYPE_F32) {
            const auto features = const_view(route.storage_dtype, converted.features_f32.data(),
                                             converted.features_f32.size());
            const auto target = const_view(route.storage_dtype, converted.target_f32.data(),
                                           converted.target_f32.size());
            return api_upload(
                api, matrix.handle, route, features, target, options.rows, options.features);
        }
        const auto features = const_view(route.storage_dtype, converted.features_f64.data(),
                                         converted.features_f64.size());
        const auto target = const_view(route.storage_dtype, converted.target_f64.data(),
                                       converted.target_f64.size());
        return api_upload(
            api, matrix.handle, route, features, target, options.rows, options.features);
    }, true);
    append_record(
        records, name, order_index, "h2d_upload", "", std::move(upload),
        "host_steady_clock",
        std::string("steady_clock around synchronous ") + api.abi_surface_name() +
            " matrix upload",
        "upload includes route-typed host/device transfer and payload-side resident statistics; typed and generic wrapper validation differ");

    const auto target_view = [&]() {
        if (route.storage_dtype == GAFIME_DTYPE_F32) {
            return const_view(route.storage_dtype, converted.target_f32.data(), converted.target_f32.size());
        }
        return const_view(route.storage_dtype, converted.target_f64.data(), converted.target_f64.size());
    };
    const GafimeConstBufferView target = target_view();

    const auto target_update = host_samples(
        options.warmups, options.repeats, calibration_cache,
        timing_calibration_key("host", name, "target_update", "spearman"),
        [&]() -> int {
            return api_update_target(api, matrix.handle, route, target, options.rows);
        }, true);
    append_record(
        records, name, order_index, "target_update", "spearman", std::move(target_update),
        "host_steady_clock",
        std::string("steady_clock around ") + api.abi_surface_name() +
            " target replacement and cache invalidation",
        "target update is measured as its own host-synchronized wrapper; the final update leaves the resident Spearman cache invalidated",
        "target_update_host_wrapper");

    const auto execution_memory_forecast = host_samples(
        options.warmups, options.repeats, calibration_cache,
        timing_calibration_key(
            "host", name, "execution_memory_forecast", "pearson"),
        [&]() -> int {
            uint64_t peak_bytes = 0;
            const int status = api_execution_memory(
                api, matrix.handle, planning_protocol, &peak_bytes);
            if (status != GAFIME_STATUS_OK) return status;
            return peak_bytes == 0 ? GAFIME_STATUS_DEVICE_ERROR : GAFIME_STATUS_OK;
        });
    append_record(
        records, name, order_index, "execution_memory_forecast", "pearson",
        std::move(execution_memory_forecast),
        "host_steady_clock",
        std::string("steady_clock around ") + api.abi_surface_name() +
            " state-aware execution-memory forecast",
        "forecast is a synchronous admission query; no device event is claimed");

    auto run_execute = [&](std::string_view operation, std::string_view metric,
                           Protocol& protocol, Result& result, auto&& prepare,
                           bool prepare_each_execute = false) {
        result.reset();
        EventSamples samples = event_samples(
            options.warmups, options.repeats, calibration_cache,
            timing_calibration_key("payload", name, operation, metric),
            [&]() -> int {
                result.reset();
                return prepare();
            },
            [&]() -> int {
                result.reset();
                return api_execute(api, matrix.handle, route, protocol, result);
            },
            prepare_each_execute);
        result.require_finite_values(route.result_dtype, api.typed());
        return samples;
    };

    Result* pearson_result_for_report = nullptr;
    std::unique_ptr<Result> pearson_result;
    SampledValues pearson_host_samples;
    for (const uint32_t metric : kMetricIds) {
        // Spearman is measured below in two deliberately separate phases:
        // target-rank construction after invalidation, then the cached metric
        // route.  Do not emit a third ambiguous Spearman record here.
        if (metric == GAFIME_METRIC_SPEARMAN) continue;
        Protocol protocol(
            route, options, descriptors, metric, 0,
            0x20000000ULL + static_cast<uint64_t>(order_index) * 32ULL + metric);
        auto result = std::make_unique<Result>(
            route, options.arity, options.candidates, api.typed());
        const std::string metric_name = metric == GAFIME_METRIC_PEARSON
            ? "pearson"
            : metric == GAFIME_METRIC_R2
                ? "r2"
                : metric == GAFIME_METRIC_MUTUAL_INFO ? "mutual_info" : "spearman";
        const auto timing = run_execute(
            "metric_kernel", metric_name, protocol, *result, []() -> int {
                return GAFIME_STATUS_OK;
            });
        append_record(
            records, name, order_index, "metric_kernel", metric_name,
            std::move(timing.gpu),
            "hip_event_elapsed_after_synchronized_execute",
            std::string("hipDeviceSynchronize before start; hipEventRecord/hipEventSynchronize around ") +
                api.abi_surface_name() + " execute wrapper",
            std::string(api.abi_surface_name()) +
                " payload metric route; selected execute wrapper synchronously materializes result buffers");
        if (metric == GAFIME_METRIC_PEARSON) {
            pearson_host_samples = std::move(timing.host);
            pearson_result = std::move(result);
            pearson_result_for_report = pearson_result.get();
        }
    }

    /* Force the resident target-rank cache invalidation before measuring its build. */
    Protocol ranking_protocol(
        route, options, descriptors, GAFIME_METRIC_SPEARMAN, 0,
        0x30000000ULL + static_cast<uint64_t>(order_index) * 32ULL + route.profile);
    auto ranking_result = std::make_unique<Result>(
        route, options.arity, options.candidates, api.typed());
    const auto ranking_timing = run_execute(
        "ranking_target_ranks", "spearman", ranking_protocol, *ranking_result,
        [&]() -> int {
            return api_update_target(api, matrix.handle, route, target, options.rows);
        },
        true);
    append_record(
        records, name, order_index, "ranking_target_ranks", "spearman",
        std::move(ranking_timing.gpu),
        "hip_event_elapsed_after_synchronized_execute",
        std::string("hipDeviceSynchronize before start; target update precedes every measured ") +
            "cold execute; hipEventRecord/hipEventSynchronize bracket the " +
            api.abi_surface_name() + " execute wrapper",
        "each measured execute is preceded by target replacement and cache invalidation; this is a combined target-update plus cold target-rank-build boundary",
        "combined_target_update_and_cold_execute");

    Protocol spearman_protocol(
        route, options, descriptors, GAFIME_METRIC_SPEARMAN, 0,
        0x31000000ULL + static_cast<uint64_t>(order_index) * 32ULL + route.profile);
    auto spearman_result = std::make_unique<Result>(
        route, options.arity, options.candidates, api.typed());
    const auto spearman_timing = run_execute(
        "metric_kernel", "spearman", spearman_protocol, *spearman_result,
        []() -> int { return GAFIME_STATUS_OK; });
        append_record(
            records, name, order_index, "metric_kernel", "spearman",
        std::move(spearman_timing.gpu),
            "hip_event_elapsed_after_synchronized_execute",
        std::string("hipDeviceSynchronize before start; hipEventRecord/hipEventSynchronize around cached ") +
            api.abi_surface_name() + " execute wrapper",
        "Spearman metric timing follows the separate cold target-rank record; this route reuses the resident target-rank cache");

    if (options.top_k != 0) {
        Protocol topk_protocol(
            route, options, descriptors, GAFIME_METRIC_PEARSON, options.top_k,
            0x32000000ULL + static_cast<uint64_t>(order_index) * 32ULL + route.profile);
        auto topk_result = std::make_unique<Result>(
            route, options.arity, options.top_k, api.typed());
        const auto topk_timing = run_execute(
            "ranking_topk_and_gather", "pearson", topk_protocol, *topk_result,
            []() -> int { return GAFIME_STATUS_OK; });
        append_record(
            records, name, order_index, "ranking_topk_and_gather", "pearson",
            std::move(topk_timing.gpu),
            "hip_event_elapsed_after_synchronized_execute",
            std::string("hipDeviceSynchronize before start; HIP events bracket ") +
                api.abi_surface_name() + " top-k execute wrapper",
            "top-k selection, selected-row gather, and result materialization are exercised by the public payload");
    }

    if (pearson_result_for_report == nullptr ||
        pearson_host_samples.samples_us.size() != options.repeats) {
        throw BenchmarkError("Pearson result was not captured for D2H/report decomposition");
    }
    append_record(
        records, name, order_index, "payload_execute", "pearson",
        std::move(pearson_host_samples),
        "host_steady_clock",
        "host steady clock around the synchronous selected execute wrapper return and caller-owned result visibility",
        "the payload execute boundary bundles device synchronization and its internal D2H/readback; the payload-internal D2H is not independently observable",
        "bundled_payload_execute_boundary", "canonical_payload_api", "within_abi_surface_only");

}

uint64_t checked_bytes(uint64_t elements, uint64_t element_size, std::string_view label) {
    if (element_size == 0 || elements > std::numeric_limits<uint64_t>::max() / element_size) {
        throw BenchmarkError(std::string(label) + " byte count overflow");
    }
    return elements * element_size;
}

void run_internal_profile(
    const Options& options,
    const Dataset& source,
    const GafimeNumericRoute& route,
    const std::string& name,
    uint32_t order_index,
    TimingCalibrationCache& calibration_cache,
    std::vector<Record>& records
) {
#if GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE == 1
    const ProfileData converted = convert_dataset(
        source, route, options.input_policy, options.rows, options.features);
    std::vector<uint32_t> descriptors;
    materialize_candidates(options.features, options.arity, options.candidates, descriptors);

    const uint64_t matrix_elements = options.rows * static_cast<uint64_t>(options.features);
    if (matrix_elements == 0 || matrix_elements > std::numeric_limits<size_t>::max()) {
        throw BenchmarkError("direct ROCm matrix element count overflow");
    }
    const uint64_t candidate_elements = options.rows * static_cast<uint64_t>(options.candidates);
    if (candidate_elements == 0 || candidate_elements > std::numeric_limits<size_t>::max()) {
        throw BenchmarkError("direct ROCm candidate element count overflow");
    }

    // The product kernels consume column-major resident matrices.  The public
    // helper dataset is row-major, so this conversion is setup-only and is not
    // part of any direct HIP-event kernel record.
    std::vector<float> features_f32;
    std::vector<double> features_f64;
    std::vector<float> target_f32;
    std::vector<double> target_f64;
    std::vector<float> means_f32;
    std::vector<double> means_f64;
    if (route.storage_dtype == GAFIME_DTYPE_F32) {
        features_f32.resize(static_cast<size_t>(matrix_elements));
        target_f32 = converted.target_f32;
        if (route.reduction_dtype == GAFIME_DTYPE_F32) {
            means_f32.resize(options.features);
        } else {
            means_f64.resize(options.features);
        }
        for (uint32_t column = 0; column < options.features; ++column) {
            double sum = 0.0;
            for (uint64_t row = 0; row < options.rows; ++row) {
                const float value = converted.features_f32[
                    static_cast<size_t>(row) * options.features + column];
                features_f32[static_cast<size_t>(column) * options.rows + row] = value;
                sum += static_cast<double>(value);
            }
            if (route.reduction_dtype == GAFIME_DTYPE_F32) {
                means_f32[column] = static_cast<float>(
                    sum / static_cast<double>(options.rows));
            } else {
                means_f64[column] = sum / static_cast<double>(options.rows);
            }
        }
    } else {
        features_f64.resize(static_cast<size_t>(matrix_elements));
        target_f64 = converted.target_f64;
        means_f64.resize(options.features);
        for (uint32_t column = 0; column < options.features; ++column) {
            double sum = 0.0;
            for (uint64_t row = 0; row < options.rows; ++row) {
                const double value = converted.features_f64[
                    static_cast<size_t>(row) * options.features + column];
                features_f64[static_cast<size_t>(column) * options.rows + row] = value;
                sum += value;
            }
            means_f64[column] = sum / static_cast<double>(options.rows);
        }
    }

    const size_t storage_size = route.storage_dtype == GAFIME_DTYPE_F32
        ? sizeof(float) : sizeof(double);
    const size_t reduction_size = route.reduction_dtype == GAFIME_DTYPE_F32
        ? sizeof(float) : sizeof(double);
    const size_t result_size = route.result_dtype == GAFIME_DTYPE_F32
        ? sizeof(float) : sizeof(double);
    const size_t rank_size = route.result_dtype == GAFIME_DTYPE_F32
        ? sizeof(float) : sizeof(double);
    const uint32_t partial_blocks = std::max<uint32_t>(
        1u, static_cast<uint32_t>((options.candidates + 256u - 1u) / 256u));
    const size_t partial_count = static_cast<size_t>(partial_blocks) * options.top_k;

    void* device_features = nullptr;
    void* device_target = nullptr;
    void* device_means = nullptr;
    void* device_materialized = nullptr;
    void* device_target_ranks = nullptr;
    void* device_metric_values = nullptr;
    void* device_partial_scores = nullptr;
    uint32_t* device_metric_ids = nullptr;
    uint32_t* device_descriptors = nullptr;
    uint32_t* device_partial_indices = nullptr;
    uint32_t* device_selected_indices = nullptr;
    void* device_selected_metric_values = nullptr;
    hipStream_t stream = nullptr;
    auto release = [&]() {
        static_cast<void>(hipFree(device_selected_metric_values));
        static_cast<void>(hipFree(device_selected_indices));
        static_cast<void>(hipFree(device_partial_indices));
        static_cast<void>(hipFree(device_partial_scores));
        static_cast<void>(hipFree(device_metric_ids));
        static_cast<void>(hipFree(device_metric_values));
        static_cast<void>(hipFree(device_target_ranks));
        static_cast<void>(hipFree(device_materialized));
        static_cast<void>(hipFree(device_descriptors));
        static_cast<void>(hipFree(device_means));
        static_cast<void>(hipFree(device_target));
        static_cast<void>(hipFree(device_features));
        if (stream != nullptr) static_cast<void>(hipStreamDestroy(stream));
        device_selected_metric_values = nullptr;
        device_selected_indices = nullptr;
        device_partial_indices = nullptr;
        device_partial_scores = nullptr;
        device_metric_ids = nullptr;
        device_metric_values = nullptr;
        device_target_ranks = nullptr;
        device_materialized = nullptr;
        device_descriptors = nullptr;
        device_means = nullptr;
        device_target = nullptr;
        device_features = nullptr;
        stream = nullptr;
    };
    try {
        const size_t matrix_bytes = static_cast<size_t>(matrix_elements) * storage_size;
        const size_t target_bytes = static_cast<size_t>(options.rows) * storage_size;
        const size_t means_bytes = static_cast<size_t>(options.features) * reduction_size;
        const size_t materialized_bytes = static_cast<size_t>(candidate_elements) * storage_size;
        const size_t metric_bytes = static_cast<size_t>(options.candidates) * result_size;
        const size_t ranks_bytes = static_cast<size_t>(options.rows) * rank_size;
        const size_t partial_bytes = partial_count * result_size;
        const size_t selected_bytes = static_cast<size_t>(options.top_k) * result_size;
        const size_t descriptor_bytes = descriptors.size() * sizeof(uint32_t);
        require_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                    "hipStreamCreate(direct lane)");
        require_hip(hipMalloc(&device_features, matrix_bytes), "hipMalloc(direct features)");
        require_hip(hipMalloc(&device_target, target_bytes), "hipMalloc(direct target)");
        require_hip(hipMalloc(&device_means, means_bytes), "hipMalloc(direct means)");
        require_hip(hipMalloc(&device_materialized, materialized_bytes),
                    "hipMalloc(direct materialized candidates)");
        require_hip(hipMalloc(&device_target_ranks, ranks_bytes),
                    "hipMalloc(direct target ranks)");
        require_hip(hipMalloc(&device_metric_values, metric_bytes),
                    "hipMalloc(direct metric values)");
        require_hip(hipMalloc(&device_metric_ids, sizeof(uint32_t)),
                    "hipMalloc(direct metric ids)");
        require_hip(hipMalloc(&device_partial_scores, partial_bytes),
                    "hipMalloc(direct partial scores)");
        require_hip(hipMalloc(&device_descriptors, descriptor_bytes),
                    "hipMalloc(direct descriptors)");
        require_hip(hipMalloc(&device_partial_indices, partial_count * sizeof(uint32_t)),
                    "hipMalloc(direct partial indices)");
        require_hip(hipMalloc(&device_selected_indices,
                              static_cast<size_t>(options.top_k) * sizeof(uint32_t)),
                    "hipMalloc(direct selected indices)");
        require_hip(hipMalloc(&device_selected_metric_values, selected_bytes),
                    "hipMalloc(direct selected metric values)");

        const void* host_features = route.storage_dtype == GAFIME_DTYPE_F32
            ? static_cast<const void*>(features_f32.data())
            : static_cast<const void*>(features_f64.data());
        const void* host_target = route.storage_dtype == GAFIME_DTYPE_F32
            ? static_cast<const void*>(target_f32.data())
            : static_cast<const void*>(target_f64.data());
        const void* host_means = route.reduction_dtype == GAFIME_DTYPE_F32
            ? static_cast<const void*>(means_f32.data())
            : static_cast<const void*>(means_f64.data());
        require_hip(hipMemcpyAsync(device_features, host_features, matrix_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "direct H2D features");
        require_hip(hipMemcpyAsync(device_target, host_target, target_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "direct H2D target");
        require_hip(hipMemcpyAsync(device_means, host_means, means_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "direct H2D means");
        require_hip(hipMemcpyAsync(device_descriptors, descriptors.data(), descriptor_bytes,
                                   hipMemcpyHostToDevice, stream),
                    "direct H2D descriptors");
        require_hip(hipStreamSynchronize(stream), "direct setup synchronize");

        const auto hip_status = [](hipError_t status) {
            return status == hipSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
        };
        const auto time_kernel = [&](std::string_view operation, std::string_view metric,
                                     auto&& launch) {
            return event_samples(
                options.warmups, options.repeats, calibration_cache,
                timing_calibration_key("direct", name, operation, metric),
                []() -> int { return GAFIME_STATUS_OK; },
                [&]() -> int { return hip_status(launch()); },
                false, stream);
        };

        const auto verify_metric_output = [&](uint32_t metric_id,
                                              std::string_view metric_name) {
            if (route.result_dtype == GAFIME_DTYPE_F32) {
                std::vector<float> values(options.candidates);
                require_hip(
                    hipMemcpyAsync(
                        values.data(), device_metric_values, metric_bytes,
                        hipMemcpyDeviceToHost, stream),
                    "direct D2H metric probe");
                require_hip(hipStreamSynchronize(stream), "direct metric probe synchronize");
                for (const float value : values) {
                    if (!std::isfinite(value)) {
                        throw BenchmarkError(
                            "direct product metric returned a non-finite fp32 value for " +
                            std::string(metric_name));
                    }
                    if ((metric_id == GAFIME_METRIC_PEARSON ||
                         metric_id == GAFIME_METRIC_SPEARMAN) &&
                        (value < -1.0001f || value > 1.0001f)) {
                        throw BenchmarkError(
                            "direct product correlation escaped [-1,1] for " +
                            std::string(metric_name));
                    }
                    if (metric_id == GAFIME_METRIC_R2 && (value < -0.0001f || value > 1.0001f)) {
                        throw BenchmarkError(
                            "direct product R2 escaped [0,1] for " + std::string(metric_name));
                    }
                    if (metric_id == GAFIME_METRIC_MUTUAL_INFO && value < -0.0001f) {
                        throw BenchmarkError(
                            "direct product mutual information became negative");
                    }
                }
                return;
            }
            std::vector<double> values(options.candidates);
            require_hip(
                hipMemcpyAsync(
                    values.data(), device_metric_values, metric_bytes,
                    hipMemcpyDeviceToHost, stream),
                "direct D2H metric probe");
            require_hip(hipStreamSynchronize(stream), "direct metric probe synchronize");
            for (const double value : values) {
                if (!std::isfinite(value)) {
                    throw BenchmarkError(
                        "direct product metric returned a non-finite fp64 value for " +
                        std::string(metric_name));
                }
                if ((metric_id == GAFIME_METRIC_PEARSON ||
                     metric_id == GAFIME_METRIC_SPEARMAN) &&
                    (value < -1.0001 || value > 1.0001)) {
                    throw BenchmarkError(
                        "direct product correlation escaped [-1,1] for " +
                        std::string(metric_name));
                }
                if (metric_id == GAFIME_METRIC_R2 && (value < -0.0001 || value > 1.0001)) {
                    throw BenchmarkError(
                        "direct product R2 escaped [0,1] for " + std::string(metric_name));
                }
                if (metric_id == GAFIME_METRIC_MUTUAL_INFO && value < -0.0001) {
                    throw BenchmarkError(
                        "direct product mutual information became negative");
                }
            }
        };

        const auto target_timing = time_kernel(
            "ranking_target_ranks", "spearman", [&]() {
                return gafime_rocm_native_direct::target_ranks(
                    route.profile, device_target, device_target_ranks, options.rows, stream);
            });
        append_record(
            records, name, order_index, "ranking_target_ranks", "spearman",
            std::move(target_timing.gpu), "hip_event_elapsed_after_synchronized_execute",
            "hipDeviceSynchronize before start; HIP events bracket the common-harness target-rank helper",
            "direct ROCm common-harness target-rank preparation; product static Spearman scoring is measured separately and the payload is not loaded",
            "direct_kernel", "supplemental_internal_kernel", "direct_kernel_and_common_helper");

        const auto materialize_timing = time_kernel(
            "candidate_materialization", "", [&]() {
                return gafime_rocm_native_direct::materialize(
                    route.profile, device_features, device_descriptors, device_materialized,
                    options.rows, options.candidates, options.arity, stream);
            });
        append_record(
            records, name, order_index, "candidate_materialization", "",
            std::move(materialize_timing.gpu), "hip_event_elapsed_after_synchronized_execute",
            "hipDeviceSynchronize before start; HIP events bracket the direct helper materialization kernel",
            "direct helper-owned candidate interaction materialization; product metric kernels remain source-bound",
            "direct_kernel", "supplemental_internal_kernel", "direct_kernel_only");

        for (const uint32_t metric_id : kMetricIds) {
            const std::string metric_name = metric_id == GAFIME_METRIC_PEARSON
                ? "pearson"
                : metric_id == GAFIME_METRIC_R2
                    ? "r2"
                    : metric_id == GAFIME_METRIC_MUTUAL_INFO ? "mutual_info" : "spearman";
            require_hip(
                hipMemcpyAsync(
                    device_metric_ids, &metric_id, sizeof(metric_id),
                    hipMemcpyHostToDevice, stream),
                "direct H2D metric id");
            require_hip(hipStreamSynchronize(stream), "direct metric-id setup synchronize");
            const auto metric_timing = time_kernel(
                "metric_kernel", metric_name, [&]() {
                    return gafime_rocm_native_direct::metric(
                        route.profile, metric_id, options.arity, options.mi_bins,
                        device_features, device_target, device_means, device_target_ranks,
                        device_descriptors, device_metric_ids, options.rows, options.candidates,
                        device_metric_values, stream);
                });
            append_record(
                records, name, order_index, "metric_kernel", metric_name,
                std::move(metric_timing.gpu), "hip_event_elapsed_after_synchronized_execute",
                "hipDeviceSynchronize before start; HIP events bracket the product-bound metric kernel",
                "direct ROCm product precision metric kernel; result dtype follows the selected numeric route",
                "direct_kernel", "supplemental_internal_kernel", "direct_kernel_only");
            verify_metric_output(metric_id, metric_name);

            if (options.top_k != 0) {
                const auto ranking_timing = time_kernel(
                    "ranking_topk", metric_name, [&]() {
                        return gafime_rocm_native_direct::ranking_topk(
                            route.profile, device_metric_values, options.candidates, options.top_k,
                            device_partial_scores, device_partial_indices, device_selected_indices,
                            partial_blocks, stream);
                    });
                append_record(
                    records, name, order_index, "ranking_topk", metric_name,
                    std::move(ranking_timing.gpu), "hip_event_elapsed_after_synchronized_execute",
                    "hipDeviceSynchronize before start; HIP events bracket product top-k selection",
                    "direct ROCm product ranking top-k kernel sequence; stable candidate identity remains integer metadata",
                    "direct_kernel", "supplemental_internal_kernel", "direct_kernel_only");

                const auto gather_timing = time_kernel(
                    "selected_row_gather", metric_name, [&]() {
                        return gafime_rocm_native_direct::selected_rows(
                            route.profile, device_metric_values, device_selected_indices,
                            options.top_k, device_selected_metric_values, stream);
                    });
                append_record(
                    records, name, order_index, "selected_row_gather", metric_name,
                    std::move(gather_timing.gpu), "hip_event_elapsed_after_synchronized_execute",
                    "hipDeviceSynchronize before start; HIP events bracket product selected-row gather",
                    "direct ROCm product selected-row result gather; visible result dtype remains route result_dtype",
                    "direct_kernel", "supplemental_internal_kernel", "direct_kernel_only");
            }
        }
        release();
    } catch (...) {
        release();
        throw;
    }
#else
    static_cast<void>(options);
    static_cast<void>(source);
    static_cast<void>(route);
    static_cast<void>(name);
    static_cast<void>(order_index);
    static_cast<void>(calibration_cache);
    static_cast<void>(records);
    throw BenchmarkError(
        "ROCm supplemental_internal_kernel requested from a non-direct compiled helper");
#endif
}

void run_host_profile(
    const Options& options,
    const Dataset& source,
    const GafimeNumericRoute& route,
    const std::string& name,
    uint32_t order_index,
    TimingCalibrationCache& calibration_cache,
    std::vector<Record>& records
) {
    const auto add_host = [&](std::string operation, std::string metric,
                              SampledValues samples, std::string note) {
        append_record(
            records, name, order_index, operation, metric, std::move(samples),
            "host_steady_clock", "host-only control in a payload-free fresh process",
            note, "supplemental_host_phase", "supplemental_host_phase",
            "supplemental_host_control");
    };

    add_host(
        "ingest_conversion", "",
        host_samples(
            options.warmups, options.repeats, calibration_cache,
            timing_calibration_key("host", name, "ingest_conversion", ""),
            [&]() -> int {
                const ProfileData converted = convert_dataset(
                    source, route, options.input_policy, options.rows,
                    options.features);
                const bool valid = route.storage_dtype == GAFIME_DTYPE_F32
                    ? !converted.features_f32.empty() && !converted.target_f32.empty()
                    : !converted.features_f64.empty() && !converted.target_f64.empty();
                return valid ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
            }),
        options.input_policy == "common-f64"
            ? "common f64 source conversion into the selected route storage dtype"
            : "profile-native host ownership/copy preparation without cross-dtype conversion");

    ProfileData converted = convert_dataset(
        source, route, options.input_policy, options.rows, options.features);
    std::vector<uint32_t> descriptors;
    add_host(
        "candidate_materialization", "",
        host_samples(
            options.warmups, options.repeats, calibration_cache,
            timing_calibration_key("host", name, "candidate_materialization", ""),
            [&]() -> int {
                materialize_candidates(
                    options.features, options.arity, options.candidates,
                    descriptors);
                return GAFIME_STATUS_OK;
            }),
        "caller-owned integer combination descriptors; no payload call");
    materialize_candidates(
        options.features, options.arity, options.candidates, descriptors);
    add_host(
        "planning", "",
        host_samples(
            options.warmups, options.repeats, calibration_cache,
            timing_calibration_key("host", name, "planning", ""),
            [&]() -> int {
                Protocol protocol(
                    route, options, descriptors, GAFIME_METRIC_PEARSON, 0,
                    0x41000000ULL + static_cast<uint64_t>(order_index) * 32ULL +
                        route.profile);
                protocol.base.reserved[
                    GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT]++;
                return protocol.numeric.route.profile == route.profile
                    ? GAFIME_STATUS_OK
                    : GAFIME_STATUS_DEVICE_ERROR;
            }),
        "caller-owned ABI descriptor construction; no payload call");

    if (options.rows > std::numeric_limits<uint64_t>::max() / options.features) {
        throw BenchmarkError("host lane matrix element count overflow");
    }
    const uint64_t matrix_elements = options.rows * options.features;
    const uint64_t matrix_bytes = checked_bytes(
        matrix_elements, dtype_size(route.storage_dtype), "host lane matrix");
    const uint64_t target_bytes = checked_bytes(
        options.rows, dtype_size(route.storage_dtype), "host lane target");
    const uint64_t result_bytes = result_buffer_bytes(route, options);

    add_host(
        "allocation", "",
        host_samples(
            options.warmups, options.repeats, calibration_cache,
            timing_calibration_key("host", name, "allocation", ""),
            [&]() -> int {
                void* matrix = nullptr;
                void* target = nullptr;
                void* result = nullptr;
                if (hipMalloc(&matrix, matrix_bytes) != hipSuccess ||
                    hipMalloc(&target, target_bytes) != hipSuccess ||
                    hipMalloc(&result, result_bytes) != hipSuccess) {
                    static_cast<void>(hipFree(result));
                    static_cast<void>(hipFree(target));
                    static_cast<void>(hipFree(matrix));
                    return GAFIME_STATUS_DEVICE_ERROR;
                }
                const hipError_t result_status = hipFree(result);
                const hipError_t target_status = hipFree(target);
                const hipError_t matrix_status = hipFree(matrix);
                const bool clean = result_status == hipSuccess &&
                    target_status == hipSuccess && matrix_status == hipSuccess;
                return clean ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
            },
            true),
        "helper-owned HIP allocation/free lifecycle; payload is not loaded");

    void* device_matrix = nullptr;
    void* device_target = nullptr;
    void* device_result = nullptr;
    require_hip(hipMalloc(&device_matrix, matrix_bytes), "hipMalloc(host lane matrix)");
    require_hip(hipMalloc(&device_target, target_bytes), "hipMalloc(host lane target)");
    require_hip(hipMalloc(&device_result, result_bytes), "hipMalloc(host lane result)");
    try {
        const void* matrix_data = route.storage_dtype == GAFIME_DTYPE_F32
            ? static_cast<const void*>(converted.features_f32.data())
            : static_cast<const void*>(converted.features_f64.data());
        const void* target_data = route.storage_dtype == GAFIME_DTYPE_F32
            ? static_cast<const void*>(converted.target_f32.data())
            : static_cast<const void*>(converted.target_f64.data());
        add_host(
            "h2d_upload", "",
            host_samples(
                options.warmups, options.repeats, calibration_cache,
                timing_calibration_key("host", name, "h2d_upload", ""),
                [&]() -> int {
                    return hipMemcpy(
                               device_matrix, matrix_data, matrix_bytes,
                               hipMemcpyHostToDevice) == hipSuccess &&
                            hipMemcpy(
                               device_target, target_data, target_bytes,
                               hipMemcpyHostToDevice) == hipSuccess
                        ? GAFIME_STATUS_OK
                        : GAFIME_STATUS_DEVICE_ERROR;
                },
                true),
            "synchronous helper-owned H2D transfer; payload is not loaded");
        require_hip(hipMemset(device_result, 0x5a, result_bytes),
                    "hipMemset(host lane result)");
        std::vector<uint8_t> host_result(static_cast<size_t>(result_bytes));
        add_host(
            "d2h_transfer", "pearson",
            host_samples(
                options.warmups, options.repeats, calibration_cache,
                timing_calibration_key("host", name, "d2h_transfer", "pearson"),
                [&]() -> int {
                    return hipMemcpy(
                               host_result.data(), device_result, result_bytes,
                               hipMemcpyDeviceToHost) == hipSuccess
                        ? GAFIME_STATUS_OK
                        : GAFIME_STATUS_DEVICE_ERROR;
                },
                true),
            "synchronous helper-owned D2H transfer; payload-internal readback is not claimed");
        add_host(
            "report_construction", "",
            host_samples(
                options.warmups, options.repeats, calibration_cache,
                timing_calibration_key("host", name, "report_construction", ""),
                [&]() -> int {
                    std::ostringstream summary;
                    summary << std::setprecision(17) << "rows=" << options.rows
                            << ";features=" << options.features
                            << ";candidates=" << options.candidates
                            << ";first=" << static_cast<uint32_t>(host_result.front());
                    return summary.str().empty() ? GAFIME_STATUS_DEVICE_ERROR
                                                 : GAFIME_STATUS_OK;
                }),
            "host-only result summary construction from caller-owned bytes");
        require_hip(hipFree(device_result), "hipFree(host lane result)");
        device_result = nullptr;
        require_hip(hipFree(device_target), "hipFree(host lane target)");
        device_target = nullptr;
        require_hip(hipFree(device_matrix), "hipFree(host lane matrix)");
        device_matrix = nullptr;
    } catch (...) {
        static_cast<void>(hipFree(device_result));
        static_cast<void>(hipFree(device_target));
        static_cast<void>(hipFree(device_matrix));
        throw;
    }
}

void append_string_array(std::ostringstream& stream, const std::vector<std::string>& values) {
    stream << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ',';
        stream << json_escape(values[index]);
    }
    stream << ']';
}

void append_orders_json(
    std::ostringstream& stream,
    const std::vector<std::array<uint32_t, 3>>& orders
) {
    stream << '[';
    for (size_t order_index = 0; order_index < orders.size(); ++order_index) {
        if (order_index != 0) stream << ',';
        stream << '[';
        for (size_t profile_index = 0; profile_index < 3; ++profile_index) {
            if (profile_index != 0) stream << ',';
            stream << json_escape(profile_name(orders[order_index][profile_index]));
        }
        stream << ']';
    }
    stream << ']';
}

void write_calibration_artifact(
    const Options& options,
    const SourceBinding& source_binding,
    const SourceBinding& harness_source_binding,
    const TimingCalibrationCache& calibration_cache,
    const std::vector<std::string>& command_line,
    const FileIdentity& benchmark_binary,
    const FileIdentity& payload,
    const FileIdentity* wheel,
    const FileIdentity* canonical_evidence,
    const std::string& dataset_identity
) {
    std::ofstream output(options.output);
    if (!output) throw BenchmarkError("cannot open calibration output: " + options.output);
    const auto identity_json = [](const FileIdentity& identity) {
        std::ostringstream stream;
        append_identity(stream, identity);
        return stream.str();
    };
    const auto strings_json = [](const std::vector<std::string>& values) {
        std::ostringstream stream;
        append_string_array(stream, values);
        return stream.str();
    };
    const auto source_json = [](const SourceBinding& binding) {
        std::ostringstream stream;
        append_source_binding(stream, binding);
        return stream.str();
    };
    const auto tree_json = [](const SourceTreeState& tree) {
        std::ostringstream stream;
        append_source_tree_state(stream, tree);
        return stream.str();
    };
    const auto git_json = [](const TrustedGitRepository& repository) {
        std::ostringstream stream;
        append_git_identity(stream, repository);
        return stream.str();
    };
    output << "{\n"
           << "  \"schema\": \"gafime.native-loop-calibration.v1\",\n"
           << "  \"status\": \"calibration_only\",\n"
           << "  \"backend\": \"rocm\",\n"
           << "  \"device\": {\"ordinal\": " << options.device << "},\n"
           << "  \"variant\": "
           << (options.variant.empty() ? "null" : json_escape(options.variant)) << ",\n"
           << "  \"scope_id\": "
           << json_escape(
               "rocm|" + options.workload + "|" + std::to_string(options.rows) + "|" +
               std::to_string(options.features) + "|" + std::to_string(options.candidates) + "|" +
               std::to_string(options.arity) + "|" + std::to_string(options.mi_bins) + "|" +
               std::to_string(options.top_k) + "|" + options.input_policy + "|" +
               options.evidence_lane + "|" + options.artifact_kind + "|" + std::to_string(options.device))
           << ",\n"
           << "  \"artifact_kind\": \"" << options.artifact_kind << "\",\n"
           << "  \"evidence_lane\": \"" << options.evidence_lane << "\",\n"
           << "  \"lane_isolation\": \"fresh_helper_process_per_variant_trial_and_lane\",\n"
           << "  \"payload_load_status\": "
           << json_escape(
               options.evidence_lane == "canonical_payload_api"
                   ? "loaded_canonical_lane_only"
                   : "not_loaded_by_lane_contract") << ",\n"
           << "  \"payload_absence_attested\": "
           << (options.evidence_lane == "canonical_payload_api" ? "false" : "true")
           << ",\n"
           << "  \"execution_mode\": "
           << json_escape(
               options.evidence_lane == "canonical_payload_api"
                   ? "canonical_payload"
                   : options.evidence_lane)
           << ",\n"
           << "  \"payload_not_loaded\": "
           << (options.evidence_lane == "canonical_payload_api" ? "false" : "true")
           << ",\n"
           << "  \"payload_loaded\": "
           << (options.evidence_lane == "canonical_payload_api" ? "true" : "false")
           << ",\n"
           << "  \"payload_execution_mode\": "
           << json_escape(
               options.evidence_lane == "canonical_payload_api"
                   ? "canonical_payload"
                   : "payload_not_loaded")
           << ",\n"
           << "  \"canonical_payload_lifecycle\": {\"status\": \"validated\", "
           << "\"schema\": \"gafime.native-decomposition.v1\", \"binding\": "
           << json_escape(
               canonical_evidence != nullptr
                   ? "external_canonical_evidence"
                   : "canonical_helper");
    if (canonical_evidence != nullptr) {
        output << ", \"path\": " << json_escape(canonical_evidence->path)
               << ", \"sha256\": " << json_escape(canonical_evidence->sha256);
    }
    output << "},\n"
           << "  \"runner_invocation_id\": "
           << json_escape(required_runner_invocation_id()) << ",\n"
           << "  \"runner_pid\": " << required_runner_pid() << ",\n"
           << "  \"process_id\": " << static_cast<uint64_t>(::getpid()) << ",\n"
           << "  \"source_commit\": " << json_escape(source_binding.commit) << ",\n"
           << "  \"product_source_commit\": " << json_escape(source_binding.commit) << ",\n"
           << "  \"harness_source_commit\": " << json_escape(harness_source_binding.commit) << ",\n"
           << "  \"source_root\": " << json_escape(source_binding.root) << ",\n"
           << "  \"product_source_root\": " << json_escape(source_binding.root) << ",\n"
           << "  \"harness_source_root\": " << json_escape(harness_source_binding.root) << ",\n"
           << "  \"source_tree_state\": " << tree_json(source_binding.tree) << ",\n"
           << "  \"product_source_tree_state\": "
           << tree_json(source_binding.tree) << ",\n"
           << "  \"harness_source_tree_state\": "
           << tree_json(harness_source_binding.tree) << ",\n"
           << "  \"source_blob\": " << source_json(source_binding) << ",\n"
           << "  \"harness_source_blob\": "
           << source_json(harness_source_binding) << ",\n"
           << "  \"git\": " << git_json(source_binding.git) << ",\n"
           << "  \"input_identity\": " << dataset_identity << ",\n"
           << "  \"provenance\": {\"benchmark_binary\": ";
    output << identity_json(benchmark_binary);
    output << ", \"payload\": " << identity_json(payload);
    output << ", \"wheel\": ";
    if (wheel != nullptr) output << identity_json(*wheel);
    else output << "null";
    output << "},\n"
           << "  \"command_line\": " << strings_json(command_line) << ",\n"
           << "  \"workload\": {\"name\": " << json_escape(options.workload)
           << ", \"rows\": " << options.rows << ", \"features\": " << options.features
           << ", \"candidates\": " << options.candidates << ", \"arity\": " << options.arity
           << ", \"mi_bins\": " << options.mi_bins << ", \"top_k\": " << options.top_k << "},\n"
           << "  \"input_policy\": " << json_escape(options.input_policy) << ",\n"
           << "  \"environment\": ";
    std::ostringstream environment_json;
    append_environment_json(environment_json);
    output << environment_json.str() << ",\n  \"affinity\": ";
    std::ostringstream affinity_json;
    append_affinity_json(affinity_json, affinity_info());
    output << affinity_json.str();
    output << ",\n"
           << "  \"entry_count\": " << calibration_cache.loop_counts.size() << ",\n"
           << "  \"entries\": [\n";
    size_t index = 0;
    for (const auto& [key, count] : calibration_cache.loop_counts) {
        output << "    {\"key\": " << json_escape(key)
               << ", \"loop_count\": " << count << "}"
               << (++index == calibration_cache.loop_counts.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void write_json(
    const Options& options,
    const std::vector<std::string>& command_line,
    const Api* api,
    const RouteDiscovery& discovery,
    const std::string& source_commit,
    const SourceBinding& source_binding,
    const SourceBinding& harness_source_binding,
    const std::string& dataset_identity,
    const FileIdentity& benchmark_source,
    const FileIdentity& benchmark_binary,
    const FileIdentity& payload,
    const FileIdentity* wheel,
    const FileIdentity* canonical_evidence,
    const FileIdentity& python_executable,
    const std::vector<std::array<uint32_t, 3>>& orders,
    const CalibrationPrepassSummary& calibration_prepass,
    const TimingCalibrationCache& calibration_cache,
    const std::vector<std::vector<std::array<uint32_t, 3>>>& profile_order_cycles,
    const ClockPowerState& clock_power_before,
    const ClockPowerState& clock_power_after,
    const std::vector<Record>& records
) {
    const ToolIdentity hipcc = identify_tool("hipcc", "--version");
    const ToolIdentity clangxx = identify_tool("clang++", "--version");
    const ToolIdentity linker = identify_tool("ld", "--version");
    const bool payload_loaded = api != nullptr;
    const bool typed_surface = payload_loaded && api->typed();
    const bool symbols_authenticated =
        payload_loaded && api->canonical_symbols_authenticated;
    const std::vector<std::string> no_symbols;
    const auto& required_symbols = payload_loaded ? api->required_symbols : no_symbols;
    const auto& optional_symbols_present =
        payload_loaded ? api->optional_symbols_present : no_symbols;
    const auto& optional_symbols_missing =
        payload_loaded ? api->optional_symbols_missing : no_symbols;
    const std::string surface = payload_loaded
        ? api->abi_surface_name()
        : "not_loaded_by_lane_contract";
    const std::string route_source = payload_loaded
        ? api->route_source_name()
        : "authoritative_static_float_routes";
    const std::string execute_boundary = !payload_loaded
        ? "not applicable; payload was not loaded in this evidence lane"
        : typed_surface
            ? "typed precision execute_f32_v2/execute_f64_v2"
            : "generic gafime_gpu_execute_v2";
    // representative_direct_transfer remains a named direct-lane boundary;
    // it is intentionally not conflated with canonical payload D2H timing.
    const std::string execution_mode =
        options.evidence_lane == "canonical_payload_api"
            ? "canonical_payload"
            : options.evidence_lane;
    const bool payload_not_loaded = !payload_loaded;
    std::ostringstream stream;
    stream << "{\n"
           << "  \"schema\":\"gafime.rocm.native_timing.v2\",\n"
           << "  \"status\":\"pass\",\n"
           << "  \"backend\":\"rocm\",\n"
           << "  \"scope_id\":"
           << json_escape(
               "rocm|" + options.workload + "|" + std::to_string(options.rows) + "|" +
               std::to_string(options.features) + "|" + std::to_string(options.candidates) + "|" +
               std::to_string(options.arity) + "|" + std::to_string(options.mi_bins) + "|" +
               std::to_string(options.top_k) + "|" + options.input_policy + "|" +
               options.evidence_lane + "|" + options.artifact_kind + "|" + std::to_string(options.device))
           << ",\n"
           << "  \"artifact_kind\":" << json_escape(options.artifact_kind) << ",\n"
           << "  \"evidence_lane\":" << json_escape(options.evidence_lane) << ",\n"
           << "  \"lane_isolation\":\"fresh_helper_process_per_variant_trial_and_lane\",\n"
           << "  \"payload_load_status\":"
           << json_escape(payload_loaded ? "loaded_canonical_lane_only" : "not_loaded_by_lane_contract")
           << ",\n"
           << "  \"loop_plan\":{\"mode\":"
           << json_escape(calibration_cache.immutable_plan ? "immutable" : "adaptive_calibration_only")
           << ",\"path\":"
           << (calibration_cache.immutable_plan ? json_escape(calibration_cache.plan_path) : "null")
           << ",\"relative_path\":"
           << (calibration_cache.immutable_plan
               ? json_escape(relative_path_from_file(
                     options.output, calibration_cache.plan_path))
               : "null")
           << ",\"semantic_sha256\":"
           << (calibration_cache.immutable_plan
               ? json_escape(calibration_cache.plan_semantic_sha256) : "null")
           << ",\"file_sha256\":"
           << (calibration_cache.immutable_plan ? json_escape(calibration_cache.plan_file_sha256) : "null")
           << ",\"entry_count\":" << calibration_cache.loop_counts.size() << "},\n"
           << "  \"execution_mode\":" << json_escape(execution_mode) << ",\n"
           << "  \"payload_not_loaded\":" << (payload_not_loaded ? "true" : "false") << ",\n"
           << "  \"payload_loaded\":" << (payload_loaded ? "true" : "false") << ",\n"
           << "  \"payload_execution_mode\":"
           << json_escape(payload_loaded ? "canonical_payload" : "payload_not_loaded") << ",\n"
           << "  \"canonical_payload_lifecycle\":{\"status\":\"validated\","
           << "\"schema\":\"gafime.native-decomposition.v1\",\"binding\":"
           << json_escape(
               canonical_evidence != nullptr
                   ? "external_canonical_evidence"
                   : "canonical_helper");
    if (canonical_evidence != nullptr) {
        stream << ",\"path\":" << json_escape(canonical_evidence->path)
               << ",\"sha256\":" << json_escape(canonical_evidence->sha256);
    }
    stream << "},\n"
           << "  \"compiled_lane\":" << json_escape(expected_compiled_lane_name()) << ",\n"
           << "  \"direct_kernel_product\":{"
           << "\"compiled\":"
           << (GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE == 1 ? "true" : "false")
           << ",\"root\":" << json_escape(GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_ROOT)
           << ",\"commit\":" << json_escape(GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_COMMIT)
           << ",\"kernels_sha256\":"
           << json_escape(GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_SHA256)
           << ",\"kernels_header_sha256\":"
           << json_escape(GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_HEADER_SHA256)
           << ",\"direct_source_sha256\":"
           << json_escape(GAFIME_ROCM_NATIVE_TIMING_LINKED_DIRECT_SOURCE_SHA256)
           << "},\n"
           << "  \"abi_surface\":" << json_escape(surface) << ",\n"
           << "  \"route_source\":" << json_escape(route_source) << ",\n"
           << "  \"route_synthesized\":" << (discovery.route_synthesized ? "true" : "false") << ",\n"
           << "  \"canonical_payload_resolution\":{\"status\":"
           << json_escape(payload_loaded ? "resolved" : "not_loaded_by_lane_contract")
           << ",\"abi_surface\":"
           << json_escape(surface) << ",\"symbols\":";
    append_string_array(stream, required_symbols);
    stream << ",\"required_symbols\":";
    append_string_array(stream, required_symbols);
    stream << ",\"optional_symbols_present\":";
    append_string_array(stream, optional_symbols_present);
    stream << ",\"optional_symbols_missing\":";
    append_string_array(stream, optional_symbols_missing);
    stream << ",\"optional_permutation_status\":"
           << json_escape(payload_loaded ? api->permutation_surface_status() : "not_loaded")
           << ",\"canonical_symbols_authenticated\":"
           << (symbols_authenticated ? "true" : "false") << "},\n"
           << "  \"capability\":{\"profile_mask\":" << discovery.profile_mask
           << ",\"storage_dtype_mask\":" << discovery.storage_dtype_mask
           << ",\"result_dtype_mask\":" << discovery.result_dtype_mask
           << ",\"flags\":" << discovery.flags << "},\n"
           << "  \"wrapper_comparability\":{"
           << "\"symbol_resolution\":\"not_comparable\","
           << "\"capability_query\":\"not_comparable\","
           << "\"planning\":\"not_comparable\","
           << "\"allocation\":\"semantic_only\","
           << "\"h2d_upload\":\"semantic_only\","
           << "\"target_update\":\"semantic_only\","
           << "\"execution_memory_forecast\":\"semantic_only\","
           << "\"metric_kernel\":\"semantic_only\","
           << "\"ranking_target_ranks\":\"semantic_only\","
           << "\"ranking_topk_and_gather\":\"semantic_only\","
           << "\"d2h_transfer\":\"host_only_d2h_unobservable\","
           << "\"d2h_transfer_direct\":\"not_measured_in_product_bound_direct_lane\","
           << "\"explicit_cleanup\":\"semantic_only\"},\n"
           << "  \"source_commit\":" << json_escape(source_commit) << ",\n"
           << "  \"product_source_commit\":" << json_escape(source_commit) << ",\n"
           << "  \"harness_source_commit\":" << json_escape(harness_source_binding.commit) << ",\n"
           << "  \"source_root\":" << json_escape(source_binding.root) << ",\n"
           << "  \"product_source_root\":" << json_escape(source_binding.root) << ",\n"
           << "  \"harness_source_root\":" << json_escape(harness_source_binding.root) << ",\n"
           << "  \"source_tree_state\":";
    append_source_tree_state(stream, source_binding.tree);
    stream << ",\n  \"product_source_tree_state\":";
    append_source_tree_state(stream, source_binding.tree);
    stream << ",\n  \"harness_source_tree_state\":";
    append_source_tree_state(stream, harness_source_binding.tree);
    stream << ",\n  \"source_blob\":";
    append_source_binding(stream, source_binding);
    stream << ",\n  \"harness_source_blob\":";
    append_source_binding(stream, harness_source_binding);
    stream << ",\n  \"git\":";
    append_git_identity(stream, source_binding.git);
    stream << ",\n"
           << "  \"benchmark\":\"canonical ABI1.1 ROCm payload lifecycle and HIP-event timing\",\n"
           << "  \"command_line\":";
    append_string_array(stream, command_line);
    stream << ",\n"
           << "  \"profiles\":[\"fp32\",\"mixed\",\"fp64\"],\n"
           << "  \"process_isolation\":\"fresh_helper_process_per_variant_trial\",\n"
           << "  \"runner_invocation_id\":"
           << json_escape(required_runner_invocation_id()) << ",\n"
           << "  \"runner_pid\":" << required_runner_pid() << ",\n"
           << "  \"process_id\":" << static_cast<uint64_t>(::getpid()) << ",\n"
           << "  \"variant\":"
           << (options.variant.empty() ? "null" : json_escape(options.variant)) << ",\n"
           << "  \"ab_block\":";
    if (options.ab_block < 0) {
        stream << "null";
    } else {
        stream << options.ab_block;
    }
    stream << ",\n  \"variant_sequence\":";
    append_string_array(stream, options.variant_sequence);
    stream << ",\n"
           << "  \"profile_orders\":";
    append_orders_json(stream, orders);
    stream << ",\n  \"order_schedule\":\"deterministic_per_cycle_shuffle_v1\",\n"
           << "  \"order_seed\":" << options.order_seed << ",\n"
           << "  \"order_repetitions\":" << options.order_repetitions << ",\n"
           << "  \"calibration_prepass\":{\"performed\":"
           << (calibration_prepass.performed ? "true" : "false")
           << ",\"profile_order\":";
    append_string_array(stream, calibration_prepass.profile_order);
    stream << ",\"records_discarded\":" << calibration_prepass.discarded_record_count
           << ",\"samples_discarded\":" << calibration_prepass.discarded_sample_count
           << ",\"calibrated_key_count\":" << calibration_prepass.calibrated_key_count
           << ",\"uses_shared_calibration_cache\":true"
           << ",\"included_payload_api\":" << (payload_loaded ? "true" : "false")
           << ",\"included_in_profile_order_cycles\":false"
           << ",\"claim_scope\":\"discarded runtime/context initialization and fixed-loop calibration only; not benchmark evidence\"},\n"
           << "  \"profile_order_cycles\":[";
    for (size_t cycle_index = 0; cycle_index < profile_order_cycles.size(); ++cycle_index) {
        if (cycle_index != 0) stream << ',';
        append_orders_json(stream, profile_order_cycles[cycle_index]);
    }
    stream << "],\n"
           << "  \"dataset_seed\":" << options.dataset_seed << ",\n"
           << "  \"input_policy\":" << json_escape(options.input_policy) << ",\n"
           << "  \"input_identity\":" << dataset_identity << ",\n"
           << "  \"dataset_identity\":" << dataset_identity << ",\n"
           << "  \"workload\":{\"name\":" << json_escape(options.workload)
           << ",\"class\":\"native_backend_decomposition\""
           << ",\"rows\":" << options.rows
           << ",\"features\":" << options.features
           << ",\"candidates\":" << options.candidates
           << ",\"arity\":" << options.arity
           << ",\"mi_bins\":" << options.mi_bins
           << ",\"top_k\":" << options.top_k
           << ",\"metric_set\":[\"pearson\",\"spearman\",\"mutual_info\",\"r2\"]},\n"
           << "  \"rows\":" << options.rows << ",\n"
           << "  \"features\":" << options.features << ",\n"
           << "  \"candidates\":" << options.candidates << ",\n"
           << "  \"arity\":" << options.arity << ",\n"
           << "  \"mi_bins\":" << options.mi_bins << ",\n"
           << "  \"top_k\":" << options.top_k << ",\n"
           << "  \"warmups\":" << options.warmups << ",\n"
           << "  \"repeats\":" << options.repeats << ",\n"
           << "  \"per_record_untimed_same_cell_preconditions\":"
           << kPerRecordUntimedSameCellPreconditions << ",\n"
           << "  \"per_record_untimed_precondition_min_us\":"
           << kPerRecordUntimedPreconditionMinUs << ",\n"
           << "  \"precondition_device_batch_target_us\":"
           << kPreconditionDeviceBatchTargetUs << ",\n"
           << "  \"max_precondition_batch_iterations\":"
           << kMaxPreconditionBatchIterations << ",\n"
           << "  \"calibration_policy\":"
              "\"fixed_loop_count_per_cell_no_per_sample_rescaling\",\n"
           << "  \"sample_region_target_us\":" << kSampleRegionTargetUs << ",\n"
           << "  \"sample_region_calibration_target_us\":"
           << kSampleRegionCalibrationTargetUs << ",\n"
           << "  \"bootstrap_resamples\":" << kBootstrapResamples << ",\n"
           << "  \"bootstrap_seed\":" << kBootstrapSeed << ",\n"
           << "  \"gpu_timing_supported\":true,\n"
           << "  \"timing_clock\":\"hipEventElapsedTime\",\n"
           << "  \"decomposition_boundaries\":{\n"
           << "    \"ingest_conversion\":"
           << json_escape(options.input_policy == "common-f64"
                ? "common f64 host dataset converted to route storage dtype"
                : "profile-native fp32/fp64 host dataset ownership preparation without cross-dtype conversion")
           << ",\n"
           << "    \"candidate_materialization\":\"caller-owned combination descriptors materialized before ABI planning\",\n"
           << "    \"planning\":\"caller-owned ABI1.1 route/base/chunk/result descriptor assembly\",\n"
           << "    \"allocation\":\"ABI-specific matrix allocation plus explicit teardown\",\n"
           << "    \"h2d_upload\":\"synchronous typed or generic matrix upload including resident-stat preparation\",\n"
           << "    \"execution_memory_forecast\":\"synchronous state-aware execution-memory admission query\",\n"
           << "    \"metric_kernel\":" << json_escape(execute_boundary) << ",\n"
           << "    \"ranking_target_ranks\":\"common-harness target-rank preparation; direct product Spearman uses the product static rank kernel without the candidate-only cached-unary path\",\n"
           << "    \"ranking_topk_and_gather\":\"canonical top-k selection and selected-row gather inside the selected ABI execute wrapper\",\n"
           << "    \"d2h_transfer\":\"canonical payload execute bundles payload-internal D2H; the host lane separately measures helper-owned D2H and the direct lane stops at product selected-row gather\",\n"
           << "    \"report_construction\":\"host-only summary construction after result visibility\"\n"
           << "  },\n"
           << "  \"measurement_categories\":{"
           << "\"canonical_payload_api\":\"typed/generic ABI wrapper plus synchronized payload execution; not pure kernel time\","
           << "\"supplemental_internal_kernel\":\"common-harness timing around exact product-source ROCm metric/static-Spearman/top-k/gather kernels plus common-harness target-rank and materialization helpers; no canonical ABI or payload-internal D2H\","
           << "\"supplemental_host_phase\":\"host-only conversion/materialization/planning/report/D2H control timing; not product-kernel comparable\"},\n"
           << "  \"comparability_contract\":{"
           << "\"canonical_payload_api\":\"within_abi_surface_only\","
           << "\"supplemental_internal_kernel\":\"product_source_bound_with_common_harness_helpers_not_canonical_payload\","
           << "\"supplemental_host_phase\":\"supplemental_host_control\"},\n"
           << "  \"unobservable_phases\":["
           << "\"payload-private per-kernel launch decomposition\","
           << "\"payload-internal D2H separated from synchronous execute\","
           << "\"wrapper validation separated from the selected payload operation\"],\n"
           << "  \"compiler\":{\n"
           << "    \"predefined_version\":" << json_escape(__VERSION__) << ",\n"
#if defined(__clang_version__)
           << "    \"clang_version\":" << json_escape(__clang_version__) << ",\n"
#else
           << "    \"clang_version\":null,\n"
#endif
           << "    \"hipcc\":";
    append_tool_identity(stream, hipcc);
    stream << ",\n    \"clangxx\":";
    append_tool_identity(stream, clangxx);
    stream << ",\n    \"linker\":";
    append_tool_identity(stream, linker);
    stream << "\n  },\n"
           << "  \"device\":";
    append_device_json(stream, options.device);
    stream << ",\n  \"environment\":";
    append_environment_json(stream);
    stream << ",\n  \"affinity\":";
    append_affinity_json(stream, affinity_info());
    stream << ",\n  \"clock_and_power_capture_point\":\"measurement-before is captured after the discarded calibration prepass and before randomized recorded cycles; measurement-after follows cycle collection and record verification\""
           << ",\n  \"clock_and_power_state\":{\"before\":";
    append_clock_power_state(stream, clock_power_before);
    stream << ",\"after\":";
    append_clock_power_state(stream, clock_power_after);
    stream << "}"
           << ",\n  \"clock\":{\"host\":\"std::chrono::steady_clock\",\"device\":\"hipEventElapsedTime synchronized on the recorded stream\"}";
    stream << ",\n  \"provenance\":{\n"
           << "    \"source_root\":";
    append_source_binding(stream, source_binding);
    stream << ",\n    \"product_source\":";
    append_source_binding(stream, source_binding);
    stream << ",\n    \"harness_source_binding\":";
    append_source_binding(stream, harness_source_binding);
    stream << ",\n    \"harness_source\":";
    append_identity(stream, benchmark_source);
    stream << ",\n    \"benchmark_source\":";
    append_identity(stream, benchmark_source);
    stream << ",\n    \"benchmark_binary\":";
    append_identity(stream, benchmark_binary);
    stream << ",\n    \"helper\":";
    append_identity(stream, benchmark_binary);
    stream << ",\n    \"payload\":";
    append_identity(stream, payload);
    if (wheel != nullptr) {
        stream << ",\n    \"wheel\":";
        append_identity(stream, *wheel);
    }
    stream << ",\n    \"python_executable\":";
    append_identity(stream, python_executable);
    stream << "\n  },\n"
           << "  \"self_checks\":{\n"
           << "    \"abi_surface\":" << json_escape(surface) << ",\n"
           << "    \"canonical_routes\":"
           << ((payload_loaded && !typed_surface && symbols_authenticated) ? "true" : "false") << ",\n"
           << "    \"typed_precision_profiles\":" << (typed_surface ? "true" : "false") << ",\n"
           << "    \"abi_surface_detected\":" << (payload_loaded ? "true" : "false") << ",\n"
           << "    \"canonical_symbols_authenticated\":"
           << (symbols_authenticated ? "true" : "false") << ",\n"
           << "    \"payload_absence_attested\":"
           << (!payload_loaded ? "true" : "false") << ",\n"
           << "    \"required_symbol_count\":" << required_symbols.size() << ",\n"
           << "    \"symbols\":";
    append_string_array(stream, required_symbols);
    stream << ",\n    \"required_symbols\":";
    append_string_array(stream, required_symbols);
    stream << ",\n    \"optional_symbols_present\":";
    append_string_array(stream, optional_symbols_present);
    stream << ",\n    \"optional_symbols_missing\":";
    append_string_array(stream, optional_symbols_missing);
    stream << ",\n    \"optional_permutation_status\":"
           << json_escape(payload_loaded ? api->permutation_surface_status() : "not_loaded")
           << ",\n    \"optional_symbol_gaps_declared\":true,\n"
           << "    \"wrapper_comparability_declared\":true,\n"
           << "    \"all_profiles_exercised\":true,\n"
           << "    \"all_six_profile_orders\":true,\n"
           << "    \"hip_events_synchronized\":"
           << (options.evidence_lane == "supplemental_host_phase" ? "false" : "true")
           << ",\n"
           << "    \"raw_sample_counts_valid\":true,\n"
           << "    \"finite_results_observed\":" << (payload_loaded ? "true" : "false")
           << "\n"
           << "  },\n"
           << "  \"records\":[\n";
    for (size_t index = 0; index < records.size(); ++index) {
        if (index != 0) stream << ",\n";
        stream << "    ";
        append_record_json(stream, records[index]);
    }
    stream << "\n  ]\n}\n";

    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) throw BenchmarkError("cannot open JSON output: " + options.output);
    output << stream.str();
    if (!output) throw BenchmarkError("cannot write JSON output: " + options.output);
}

void verify_result_finiteness(const std::vector<Record>& records) {
    if (records.empty()) throw BenchmarkError("no timing records produced");
    for (const Record& record : records) {
        if (record.samples.size() < 30) {
            throw BenchmarkError("record has fewer than 30 raw samples: " + record.operation);
        }
        if (record.raw_samples.size() != record.samples.size()) {
            throw BenchmarkError("record raw/normalized sample count mismatch: " + record.operation);
        }
        if (record.loop_counts_per_sample.size() != record.samples.size()) {
            throw BenchmarkError("record loop/sample count mismatch: " + record.operation);
        }
        if (record.precondition_iterations < kPerRecordUntimedSameCellPreconditions ||
            !std::isfinite(record.precondition_duration_us) ||
            record.precondition_duration_us < kPerRecordUntimedPreconditionMinUs ||
            record.precondition_max_batch_iterations == 0 ||
            record.precondition_max_batch_iterations >
                kMaxPreconditionBatchIterations) {
            throw BenchmarkError(
                "record did not complete the bounded untimed same-cell precondition: " +
                record.operation);
        }
        if (record.loop_count_per_sample == 0 ||
            std::any_of(
                record.loop_counts_per_sample.begin(),
                record.loop_counts_per_sample.end(),
                [&](uint32_t value) {
                    return value != record.loop_count_per_sample;
                })) {
            throw BenchmarkError(
                "record changed its fixed calibration loop count across samples: " +
                record.operation);
        }
        for (const double sample : record.samples) {
            if (!std::isfinite(sample) || sample <= 0.0) {
                throw BenchmarkError("record has invalid timing sample: " + record.operation);
            }
        }
        for (const double sample : record.raw_samples) {
            if (!std::isfinite(sample) || sample <= 0.0) {
                throw BenchmarkError("record has invalid raw timing sample: " + record.operation);
            }
        }
        const double raw_min = *std::min_element(record.raw_samples.begin(), record.raw_samples.end());
        if (raw_min < kSampleRegionTargetUs) {
            throw BenchmarkError("record sampled region stayed below 5 ms after fixed calibration: " +
                                 record.operation);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> command_line;
        command_line.reserve(static_cast<size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            command_line.emplace_back(argv[index]);
        }
        const Options options = parse_options(argc, argv);
        validate_compiled_lane(options);
        const bool canonical_lane =
            options.evidence_lane == "canonical_payload_api";
        FileIdentity canonical_evidence;
        const FileIdentity* canonical_evidence_pointer = nullptr;
        if (!canonical_lane) {
            // Authenticate the external canonical lifecycle before HIP setup,
            // payload discovery, or any device allocation.  The helper binds
            // only the exact file bytes; perf13 independently parses and
            // validates the reopened evidence.
            canonical_evidence = identify_external_canonical_evidence(
                options.canonical_evidence_path);
            canonical_evidence_pointer = &canonical_evidence;
        }
        const std::string payload_path = absolute_path(options.payload);
        require_hip(hipSetDevice(static_cast<int>(options.device)), "hipSetDevice");
        const bool internal_lane =
            options.evidence_lane == "supplemental_internal_kernel";
        const bool host_lane = options.evidence_lane == "supplemental_host_phase";
        Api api;
        if (canonical_lane) {
            api = open_payload(payload_path);
        } else if (payload_is_loaded(payload_path)) {
            throw BenchmarkError(
                "noncanonical evidence lane inherited a loaded ROCm payload");
        }
        const RouteDiscovery discovery = canonical_lane
            ? discover_routes(api, options.device)
            : synthetic_discovery();
        const FileIdentity benchmark_source = identify_file(__FILE__);
        const std::string executable = "/proc/self/exe";
        const FileIdentity benchmark_binary = identify_file(executable);
        const FileIdentity payload = identify_file(payload_path);
        const FileIdentity python_executable = identify_file_with_path_policy(
            observed_python_executable(), false);
        FileIdentity wheel;
        const FileIdentity* wheel_pointer = nullptr;
        if (!options.wheel.empty()) {
            wheel = identify_file(options.wheel);
            wheel_pointer = &wheel;
        }
        Options harness_options = options;
        harness_options.source_root = options.harness_source_root;
        const SourceBinding harness_source_binding = identify_source(
            harness_options, benchmark_source, true);
        const SourceBinding source_binding = identify_product_source(options);
        validate_linked_direct_product(options, source_binding, harness_source_binding);
        const std::string source_commit = source_binding.commit;
        if (source_commit.size() != 40 ||
            !std::all_of(source_commit.begin(), source_commit.end(), [](unsigned char value) {
                return std::isxdigit(value) != 0;
            })) {
            throw BenchmarkError("could not resolve a full source commit for provenance");
        }
        if (harness_source_binding.commit.size() != 40 ||
            !std::all_of(harness_source_binding.commit.begin(), harness_source_binding.commit.end(),
                         [](unsigned char value) { return std::isxdigit(value) != 0; })) {
            throw BenchmarkError("could not resolve a full harness source commit for provenance");
        }
        if (!source_binding.git.verified || !harness_source_binding.git.verified) {
            throw BenchmarkError(
                "product and harness Git repository identities must be physically verified");
        }
        if (source_binding.tree.status != "clean") {
            throw BenchmarkError("product source root must be a clean Git tree for native evidence");
        }
        if (harness_source_binding.tree.status != "clean") {
            throw BenchmarkError("harness source root must be a clean Git tree for native evidence");
        }
        if (harness_source_binding.relative_path.empty() ||
            harness_source_binding.source_sha256.empty() ||
            harness_source_binding.current_git_blob.empty() ||
            harness_source_binding.current_git_blob != harness_source_binding.head_git_blob) {
            throw BenchmarkError(
                "harness source identity must bind a tracked helper with matching current and HEAD Git blobs");
        }

        // Payload discovery, route negotiation, and HIP setup are complete
        // before the calibration prepass.  The measurement-before snapshot
        // is captured after that discarded prepass, immediately before the
        // randomized cycles begin, so it describes the recorded region.
        const Dataset dataset = make_dataset(options);
        std::vector<std::array<uint32_t, 3>> orders;
        std::array<uint32_t, 3> order = kProfileIds;
        do {
            orders.push_back(order);
        } while (std::next_permutation(order.begin(), order.end()));
        if (orders.size() != 6) throw BenchmarkError("six profile permutations were not generated");

        const auto canonical_orders = orders;
        TimingCalibrationCache calibration_cache;
        if (!options.loop_plan_path.empty()) {
            const ImmutableLoopPlan plan = load_loop_plan(options);
            const std::string expected_scope_id =
                "rocm|" + options.workload + "|" + std::to_string(options.rows) + "|" +
                std::to_string(options.features) + "|" + std::to_string(options.candidates) + "|" +
                std::to_string(options.arity) + "|" + std::to_string(options.mi_bins) + "|" +
                std::to_string(options.top_k) + "|" + options.input_policy + "|" +
                options.evidence_lane + "|" + options.artifact_kind + "|" + std::to_string(options.device);
            gafime_native_loop_plan::ExpectedScope expected_scope;
            expected_scope.backend = "rocm";
            expected_scope.workload = options.workload;
            expected_scope.rows = options.rows;
            expected_scope.features = options.features;
            expected_scope.candidates = options.candidates;
            expected_scope.arity = options.arity;
            expected_scope.mi_bins = options.mi_bins;
            expected_scope.top_k = options.top_k;
            expected_scope.input_policy = options.input_policy;
            expected_scope.evidence_lane = options.evidence_lane;
            expected_scope.artifact_kind = options.artifact_kind;
            expected_scope.device_json = gafime_native_loop_plan::expected_rocm_device_json(options.device);
            expected_scope.scope_id = expected_scope_id;
            try {
                gafime_native_loop_plan::validate_scope(plan, expected_scope);
                gafime_native_loop_plan::validate_variant_binding(
                    plan, options.variant, source_binding.commit,
                    harness_source_binding.commit);
            } catch (const gafime_native_loop_plan::ParseError& error) {
                throw BenchmarkError(error.what());
            }
            calibration_cache.loop_counts = plan.entries;
            calibration_cache.immutable_plan = true;
            calibration_cache.plan_path = plan.path;
            calibration_cache.plan_semantic_sha256 = plan.semantic_sha256;
            calibration_cache.plan_file_sha256 = plan.file_sha256;
            if (plan.evidence_lane != options.evidence_lane) {
                throw BenchmarkError("immutable native loop plan evidence lane does not match helper scope");
            }
            if (plan.artifact_kind != options.artifact_kind) {
                throw BenchmarkError("immutable native loop plan artifact kind does not match helper scope");
            }
        }
        const auto run_profile_order = [&](const std::array<uint32_t, 3>& profile_order_ids,
                                           uint32_t order_index,
                                           std::vector<Record>& target_records) {
            std::vector<std::string> profile_order;
            profile_order.reserve(profile_order_ids.size());
            for (const uint32_t ordered_profile : profile_order_ids) {
                profile_order.push_back(profile_name(ordered_profile));
            }
            for (const uint32_t profile : profile_order_ids) {
                const size_t record_start = target_records.size();
                const size_t profile_index = static_cast<size_t>(
                    std::find(kProfileIds.begin(), kProfileIds.end(), profile) - kProfileIds.begin());
                const GafimeNumericRoute& route = discovery.by_profile[profile_index];
                if (canonical_lane) {
                    run_canonical_profile(
                        options, api, dataset, route, profile_name(profile),
                        order_index, calibration_cache, target_records);
                } else if (internal_lane) {
                    run_internal_profile(
                        options, dataset, route, profile_name(profile), order_index,
                        calibration_cache, target_records);
                } else if (host_lane) {
                    run_host_profile(
                        options, dataset, route, profile_name(profile), order_index,
                        calibration_cache, target_records);
                } else {
                    throw BenchmarkError("unreachable native evidence lane dispatch");
                }
                for (size_t record_index = record_start;
                     record_index < target_records.size(); ++record_index) {
                    target_records[record_index].profile_order = profile_order;
                }
            }
        };

        // Prime every canonical payload calibration key in fp32/mixed/fp64
        // order.  These records are intentionally discarded: this pass warms
        // runtime/context state and fills the shared fixed-loop cache, but it
        // is not part of the randomized order evidence or cycle indices.
        CalibrationPrepassSummary calibration_prepass;
        calibration_prepass.performed = true;
        for (const uint32_t profile : canonical_orders.front()) {
            calibration_prepass.profile_order.push_back(profile_name(profile));
        }
        std::vector<Record> discarded_prepass_records;
        run_profile_order(canonical_orders.front(), 0, discarded_prepass_records);
        calibration_prepass.discarded_record_count = discarded_prepass_records.size();
        for (const Record& record : discarded_prepass_records) {
            calibration_prepass.discarded_sample_count += record.samples.size();
        }
        calibration_prepass.calibrated_key_count = calibration_cache.loop_counts.size();
        discarded_prepass_records.clear();

        if (options.calibration_only) {
            if (!canonical_lane && payload_is_loaded(payload_path)) {
                throw BenchmarkError(
                    "noncanonical calibration lane loaded the ROCm payload");
            }
            const std::string calibration_dataset_identity = rocm_dataset_identity_json(options, dataset);
            write_calibration_artifact(
                options, source_binding, harness_source_binding, calibration_cache,
                command_line, benchmark_binary, payload, wheel_pointer,
                canonical_evidence_pointer,
                calibration_dataset_identity);
            return 0;
        }

        // Only this vector is serialized.  It is created after the discarded
        // prepass so no calibration/runtime-initialization records can enter a
        // randomized order cycle by accident.
        const ClockPowerState clock_power_before = capture_clock_power_state();
        std::mt19937_64 order_generator(options.order_seed);
        std::vector<std::array<uint32_t, 3>> previous_orders;
        std::vector<std::vector<std::array<uint32_t, 3>>> profile_order_cycles;
        std::vector<Record> records;
        records.reserve(6 * options.order_repetitions * 3 * 16);
        for (uint32_t order_repeat = 0;
             order_repeat < options.order_repetitions;
             ++order_repeat) {
            // Shuffle a complete six-order cycle from the reproducible seed
            // stream.  order_index retains the cycle/slot identity so the
            // validator can cluster records without inferring it from names.
            orders = canonical_orders;
            std::shuffle(orders.begin(), orders.end(), order_generator);
            if (!previous_orders.empty() && orders == previous_orders) {
                // Do not let an exact shuffle collision recreate the same
                // temporal schedule.  This deterministic rotation preserves
                // all six permutations and remains seed/provenance bound.
                std::rotate(orders.begin(), orders.begin() + 1, orders.end());
            }
            if (std::set<std::array<uint32_t, 3>>(orders.begin(), orders.end()).size() != 6) {
                throw BenchmarkError("six profile permutations were not covered in this cycle");
            }
            previous_orders = orders;
            profile_order_cycles.push_back(orders);
            for (size_t order_index = 0; order_index < orders.size(); ++order_index) {
                run_profile_order(
                    orders[order_index],
                    static_cast<uint32_t>(order_index + order_repeat * orders.size()),
                    records);
            }
        }
        verify_result_finiteness(records);
        validate_loop_plan_consumed(calibration_cache);
        if (!canonical_lane && payload_is_loaded(payload_path)) {
            throw BenchmarkError(
                "noncanonical recorded lane loaded the ROCm payload");
        }
        const ClockPowerState clock_power_after = capture_clock_power_state();

        const std::string dataset_identity = rocm_dataset_identity_json(options, dataset);
        write_json(
            options, command_line, canonical_lane ? &api : nullptr, discovery,
            source_commit, source_binding,
            harness_source_binding,
            dataset_identity, benchmark_source,
            benchmark_binary, payload, wheel_pointer, canonical_evidence_pointer,
            python_executable, orders,
            calibration_prepass,
            calibration_cache,
            profile_order_cycles, clock_power_before, clock_power_after, records);
        std::cout << "wrote " << options.output << " with " << records.size()
                  << " records, six profile orders, and " << options.repeats
                  << " raw samples per record\n";
        return 0;
    } catch (const BenchmarkError& error) {
        std::cerr << "gafime_rocm_native_timing: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "gafime_rocm_native_timing: unexpected error: " << error.what() << '\n';
        return 1;
    }
}
