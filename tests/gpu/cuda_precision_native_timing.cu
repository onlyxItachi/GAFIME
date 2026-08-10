#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sched.h>
#include <sys/types.h>
#endif

#include "../../src/common/gafime_gpu_abi.hpp"
#include "../../src/cuda/precision_kernels.cuh"

namespace {

constexpr uint32_t kDefaultRows = 256;
constexpr uint32_t kDefaultCandidates = 8;
constexpr uint32_t kDefaultFeatures = 8;
constexpr uint32_t kDefaultArity = 1;
constexpr uint32_t kDefaultMiBins = 32;
constexpr uint32_t kDefaultTopK = 2;
constexpr uint32_t kDefaultWarmups = 10;
constexpr uint32_t kDefaultRepeats = 30;
constexpr uint32_t kMinimumOrderRepetitions = 5;
constexpr double kSampleRegionTargetUs = 5000.0;
constexpr double kSampleRegionCalibrationTargetUs = kSampleRegionTargetUs * 2.0;
constexpr uint32_t kMaxLoopCount = 1u << 20;
constexpr uint32_t kBootstrapResamples = 2000;
constexpr uint64_t kBootstrapSeed = 20260809ULL;

struct SampledValues {
    // samples_us is normalized to one operation; raw_samples_us retains the
    // complete calibrated region measured by the selected clock.
    std::vector<double> samples_us;
    std::vector<double> raw_samples_us;
    std::vector<uint32_t> loop_counts_per_sample;
    uint32_t loop_count_per_sample = 1;
};

struct Options {
    std::string workload = "cuda-native-precision";
    uint32_t rows = kDefaultRows;
    uint32_t features = kDefaultFeatures;
    uint32_t candidates = kDefaultCandidates;
    uint32_t arity = kDefaultArity;
    uint32_t mi_bins = kDefaultMiBins;
    uint32_t top_k = kDefaultTopK;
    uint32_t warmups = kDefaultWarmups;
    uint32_t repeats = kDefaultRepeats;
    uint32_t order_repetitions = kMinimumOrderRepetitions;
    uint64_t seed = 20260809;
    std::vector<std::string> requested_profiles{"fp32", "mixed", "fp64"};
    std::string json_path;
    std::string csv_path;
    std::string payload_path;
    std::string wheel_path;
    std::string source_root;
    std::string harness_source_root;
    std::string canonical_evidence_path;
    std::string input_policy = "common-f64";
    std::string variant;
    int64_t ab_block = -1;
    std::vector<std::string> variant_sequence;
};

struct TimingRecord {
    std::string profile;
    uint32_t order_index = 0;
    std::vector<std::string> profile_order;
    std::string operation;
    std::string metric;
    std::string clock;
    std::string timing_scope;
    bool supplemental = false;
    std::string evidence_lane = "supplemental_internal_kernel";
    std::string comparability = "direct_kernel_only";
    std::string note;
    std::vector<double> samples_us;
    std::vector<double> raw_samples_us;
    std::vector<uint32_t> loop_counts_per_sample;
    uint32_t loop_count_per_sample = 1;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_status(cudaError_t status, const char* operation) {
    check_cuda(status, operation);
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
        default: escaped += character; break;
        }
    }
    return escaped;
}

std::string csv_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value) {
        if (character == '"') escaped.push_back('"');
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

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
            const uint32_t sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sigma1 + choose + kRoundConstants[index] + words[index];
            const uint32_t sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
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
            block[block_size++] = static_cast<uint8_t>(original_bit_count >> (index * 8));
        }
        transform();
        std::ostringstream digest;
        digest << std::hex << std::setfill('0');
        for (const uint32_t value : state) digest << std::setw(8) << value;
        return digest.str();
    }
};

std::string sha256_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot open file for SHA-256: " + path);
    Sha256 hash;
    std::array<uint8_t, 64 * 1024> buffer{};
    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = file.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!file.eof()) fail("failed while hashing: " + path);
    return hash.finish();
}

std::string shell_quote(const std::string& value);
std::string command_output(const std::string& command);

struct CommandSnapshot {
    std::string command;
    std::string status = "unavailable";
    std::string output;
    std::string detail;
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

struct ClockPowerState {
    CpuGovernorSnapshot cpu_governor;
    CommandSnapshot nvidia_smi;
};

ClockPowerState capture_clock_power_state() {
    ClockPowerState state;
    state.cpu_governor = capture_cpu_governors();
    state.nvidia_smi = capture_command(
        "nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,"
        "clocks.current.sm,clocks.current.memory,power.draw,power.limit "
        "--format=csv,noheader,nounits 2>&1");
    return state;
}

struct SourceTreeState {
    std::string status = "not_supplied";
    std::vector<std::string> entries;
    size_t entry_count = 0;
    std::string detail;
};

std::string source_root_path(const Options& options);
std::string harness_source_root_path(const Options& options);
std::string source_relative_path(const std::string& source_root);
std::string git_commit(const std::string& source_root);
std::string git_blob(const std::string& source_root, const std::string& relative_path,
                     bool head_blob);
SourceTreeState source_tree_state(const std::string& source_root);

std::string sha256_bytes(const void* data, size_t size) {
    Sha256 hash;
    if (size != 0) {
        hash.update(reinterpret_cast<const uint8_t*>(data), size);
    }
    return hash.finish();
}

template <typename T>
std::string sha256_vector(const std::vector<T>& values) {
    return sha256_bytes(values.data(), values.size() * sizeof(T));
}

struct ToolIdentity {
    std::string command;
    std::string path;
    std::string version;
    std::string status = "unavailable";
};

ToolIdentity identify_tool(const char* executable, const char* version_argument = "--version") {
    ToolIdentity identity;
    identity.command = std::string(executable) + " " + version_argument;
    identity.path = command_output("command -v " + shell_quote(executable));
    identity.version = command_output(identity.command + " 2>&1");
    identity.status = identity.version.empty() ? "unavailable" : "observed";
    return identity;
}

void append_tool_identity(std::ostream& output, const ToolIdentity& identity) {
    output << "{\"command\":\"" << json_escape(identity.command)
           << "\",\"path\":\"" << json_escape(identity.path)
           << "\",\"status\":\"" << identity.status
           << "\",\"version\":\"" << json_escape(identity.version) << "\"}";
}

struct SourceBinding {
    std::string root;
    std::string relative_path;
    std::string commit;
    std::string source_sha256;
    std::string current_git_blob;
    std::string head_git_blob;
    SourceTreeState tree;
};

SourceBinding identify_product_source(const Options& options) {
    SourceBinding binding;
    binding.root = source_root_path(options);
    binding.commit = git_commit(binding.root);
    binding.tree = source_tree_state(binding.root);
    return binding;
}

SourceBinding identify_harness_source(const Options& options, const std::string& source_sha256) {
    SourceBinding binding;
    binding.root = harness_source_root_path(options);
    binding.relative_path = source_relative_path(binding.root);
    binding.commit = git_commit(binding.root);
    binding.source_sha256 = source_sha256;
    binding.current_git_blob = git_blob(binding.root, binding.relative_path, false);
    binding.head_git_blob = git_blob(binding.root, binding.relative_path, true);
    binding.tree = source_tree_state(binding.root);
    return binding;
}

void append_source_tree_state(std::ostream& output, const SourceTreeState& state) {
    output << "{\"status\":\"" << json_escape(state.status)
           << "\",\"entry_count\":" << state.entry_count << ",\"entries\":[";
    for (size_t index = 0; index < state.entries.size(); ++index) {
        if (index != 0) output << ",";
        output << "\"" << json_escape(state.entries[index]) << "\"";
    }
    output << ']';
    if (!state.detail.empty()) {
        output << ",\"detail\":\"" << json_escape(state.detail) << "\"";
    }
    output << '}';
}

void append_source_binding(std::ostream& output, const SourceBinding& binding) {
    output << "{\"path\":\"" << json_escape(binding.root)
           << "\",\"relative_source\":\"" << json_escape(binding.relative_path)
           << "\",\"source_path\":\"" << json_escape(binding.relative_path)
           << "\",\"commit\":\"" << json_escape(binding.commit)
           << "\",\"sha256\":\"" << binding.source_sha256
           << "\",\"source_sha256\":\"" << binding.source_sha256
           << "\",\"git_blob\":\"" << json_escape(binding.current_git_blob)
           << "\",\"current_git_blob\":\"" << json_escape(binding.current_git_blob)
           << "\",\"head_git_blob\":\"" << json_escape(binding.head_git_blob)
           << "\",\"tree_state\":";
    append_source_tree_state(output, binding.tree);
    output << '}';
}

std::string shell_quote(const std::string& value) {
    std::string quoted("'");
    for (const char character : value) {
        if (character == '\'') quoted += "'\\''";
        else quoted += character;
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
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
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

std::string inferred_source_root() {
    std::error_code error;
    std::filesystem::path source_path(__FILE__);
    if (!source_path.is_absolute()) {
        source_path = std::filesystem::absolute(source_path, error);
    }
    if (error) return {};
    // This helper lives at tests/gpu/<file>; the repository root is three
    // parents above the source file.
    return canonical_path(source_path.parent_path().parent_path().parent_path().string());
}

std::string source_root_path(const Options& options) {
    return canonical_path(options.source_root.empty() ? inferred_source_root() : options.source_root);
}

std::string harness_source_root_path(const Options& options) {
    return canonical_path(
        options.harness_source_root.empty() ? source_root_path(options)
                                             : options.harness_source_root);
}

std::string git_commit(const std::string& source_root) {
    if (source_root.empty()) return {};
    return command_output("git -C " + shell_quote(source_root) + " rev-parse HEAD");
}

SourceTreeState source_tree_state(const std::string& source_root) {
    SourceTreeState state;
    if (source_root.empty()) return state;
    const std::string inside = command_output(
        "git -C " + shell_quote(source_root) + " rev-parse --is-inside-work-tree 2>/dev/null");
    if (inside != "true") {
        state.status = "unavailable";
        state.detail = "source root is not a Git work tree";
        return state;
    }
    const std::string porcelain = command_output(
        "git -C " + shell_quote(source_root) +
        " status --porcelain=v1 --untracked-files=all 2>/dev/null");
    std::istringstream lines(porcelain);
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
    const auto root = std::filesystem::path(source_root);
    auto source_path = std::filesystem::path(canonical_path(__FILE__));
    const auto relative = std::filesystem::relative(source_path, root, error);
    if (error || relative.empty() || relative.string().starts_with("..")) return {};
    return relative.generic_string();
}

std::string git_blob(const std::string& source_root, const std::string& relative_path,
                     bool head_blob) {
    if (source_root.empty() || relative_path.empty()) return {};
    const std::string command = head_blob
        ? "git -C " + shell_quote(source_root) + " rev-parse HEAD:" + shell_quote(relative_path) +
              " 2>/dev/null"
        : "git -C " + shell_quote(source_root) + " hash-object --path=" +
              shell_quote(relative_path) + " -- " + shell_quote(relative_path) + " 2>/dev/null";
    const std::string output = command_output(command);
    if (output.size() != 40 ||
        !std::all_of(output.begin(), output.end(), [](unsigned char value) {
            return std::isxdigit(value) != 0;
        })) {
        return {};
    }
    return output;
}

bool is_full_commit(const std::string& value) {
    return value.size() == 40 &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isxdigit(character) != 0;
        });
}

void validate_source_provenance(
    const SourceBinding& product_source,
    const SourceBinding& harness_source,
    const std::string& source_path,
    const std::string& source_sha256
) {
    if (!is_full_commit(product_source.commit)) {
        fail("could not resolve a full product source commit for provenance");
    }
    if (product_source.tree.status != "clean") {
        fail("product source tree must be clean for native evidence");
    }
    if (!is_full_commit(harness_source.commit)) {
        fail("could not resolve a full harness source commit for provenance");
    }
    if (harness_source.tree.status != "clean") {
        fail("harness source tree must be clean for native evidence");
    }
    if (harness_source.relative_path.empty() || harness_source.current_git_blob.empty() ||
        harness_source.head_git_blob.empty() ||
        harness_source.current_git_blob != harness_source.head_git_blob) {
        fail("harness source blob is not cleanly bound to its HEAD checkout");
    }
    const std::string resolved_source_path = canonical_path(source_path);
    if (resolved_source_path.empty() ||
        sha256_file(resolved_source_path) != source_sha256 ||
        harness_source.source_sha256 != source_sha256) {
        fail("harness source SHA-256 does not match the measured helper source");
    }
}

std::string env_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string observed_python_executable() {
    const std::string virtual_env = env_value("VIRTUAL_ENV");
    if (!virtual_env.empty()) {
#if defined(_WIN32)
        const std::filesystem::path candidate =
            std::filesystem::path(virtual_env) / "Scripts" / "python.exe";
#else
        const std::filesystem::path candidate =
            std::filesystem::path(virtual_env) / "bin" / "python";
#endif
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error) && !error) {
            return std::filesystem::absolute(candidate, error).lexically_normal().string();
        }
    }
#if defined(_WIN32)
    return command_output("where python");
#else
    std::string path = command_output("command -v python3");
    if (path.empty()) path = command_output("command -v python");
    return path;
#endif
}

std::string json_array(const std::vector<std::string>& values) {
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ", ";
        output << '"' << json_escape(values[index]) << '"';
    }
    output << ']';
    return output.str();
}

struct PayloadResolution {
    std::string status = "not_supplied";
    std::string abi_surface = "none";
    std::vector<std::string> symbols;
    std::string detail;
};

// These declarations intentionally stay local to this benchmark.  The helper
// is compiled from the candidate tree while it must also load the historical
// pre-freeze PR-70 typed baseline payload, whose typed ABI structs no longer
// exist in the candidate header.
// Keeping the layouts here makes the dlsym boundary explicit and prevents the
// benchmark from accidentally treating a generic route payload as typed ABI.
struct NativePrecisionCapabilities {
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t profile_mask;
    uint32_t storage_dtype_mask;
    uint32_t result_dtype_mask;
    uint32_t flags;
    uint64_t reserved[8];
};

struct NativePrecisionMatrixDesc {
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

struct NativePrecisionLaunchProtocol {
    uint32_t abi_version;
    uint32_t profile;
    const GafimeLaunchProtocol* base;
    uint64_t reserved[8];
};

struct NativeResultTableF64 {
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

struct NativeNumericRoute {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t route_id;
    uint32_t profile;
    uint32_t storage_dtype;
    uint32_t pointwise_dtype;
    uint32_t reduction_dtype;
    uint32_t result_dtype;
    uint32_t overflow_policy;
    uint32_t flags;
    uint64_t reserved[8];
};

struct NativeConstBufferView {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t flags;
    const void* data;
    uint64_t element_count;
    uint64_t byte_length;
    uint64_t byte_stride;
    uint64_t reserved[4];
};

struct NativeMutableBufferView {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t flags;
    void* data;
    uint64_t element_capacity;
    uint64_t byte_length;
    uint64_t byte_stride;
    uint64_t reserved[4];
};

struct NativeNumericMatrixDesc {
    uint32_t abi_version;
    uint32_t struct_size;
    NativeNumericRoute route;
    uint32_t layout;
    uint32_t flags;
    uint64_t rows;
    uint32_t cols;
    uint32_t row_stride;
    uint64_t bytes;
    uint64_t reserved[8];
};

struct NativeNumericLaunchProtocol {
    uint32_t abi_version;
    uint32_t struct_size;
    NativeNumericRoute route;
    const GafimeLaunchProtocol* base;
    uint64_t reserved[8];
};

struct NativeNumericResultTable {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t max_arity;
    uint32_t metric_count;
    uint32_t flags;
    uint32_t reserved32;
    uint64_t capacity;
    uint64_t row_count;
    uint32_t* combo_indices;
    NativeMutableBufferView metric_values;
    uint32_t* ranks;
    uint32_t* families;
    uint64_t* candidate_ids;
    uint32_t* row_flags;
    uint64_t reserved[8];
};

// These local mirrors are deliberately byte-pinned at the historical typed
// baseline boundary.  The executable is a
// common consumer built from the candidate tree, but it must be able to load
// the historical typed payload without including that payload's removed
// declarations.  A changed local layout must fail at compile time instead of
// turning dlsym calls into an ABI experiment.
static_assert(alignof(NativePrecisionCapabilities) == alignof(uint64_t));
static_assert(sizeof(NativePrecisionCapabilities) == 88);
static_assert(offsetof(NativePrecisionCapabilities, reserved) == 24);
static_assert(alignof(NativePrecisionMatrixDesc) == alignof(uint64_t));
static_assert(sizeof(NativePrecisionMatrixDesc) == 112);
static_assert(offsetof(NativePrecisionMatrixDesc, rows) == 24);
static_assert(offsetof(NativePrecisionMatrixDesc, cols) == 32);
static_assert(offsetof(NativePrecisionMatrixDesc, bytes) == 40);
static_assert(offsetof(NativePrecisionMatrixDesc, reserved) == 48);
static_assert(alignof(NativePrecisionLaunchProtocol) == alignof(uint64_t));
static_assert(sizeof(NativePrecisionLaunchProtocol) == 80);
static_assert(offsetof(NativePrecisionLaunchProtocol, base) == 8);
static_assert(offsetof(NativePrecisionLaunchProtocol, reserved) == 16);
static_assert(alignof(NativeResultTableF64) == alignof(uint64_t));
static_assert(sizeof(NativeResultTableF64) == 152);
static_assert(offsetof(NativeResultTableF64, capacity) == 16);
static_assert(offsetof(NativeResultTableF64, combo_indices) == 32);
static_assert(offsetof(NativeResultTableF64, backend_private) == 80);
static_assert(offsetof(NativeResultTableF64, reserved) == 88);

static_assert(alignof(NativeNumericRoute) == alignof(uint64_t));
static_assert(sizeof(NativeNumericRoute) == 104);
static_assert(offsetof(NativeNumericRoute, reserved) == 40);
static_assert(alignof(NativeConstBufferView) == alignof(uint64_t));
static_assert(sizeof(NativeConstBufferView) == 80);
static_assert(offsetof(NativeConstBufferView, data) == 16);
static_assert(offsetof(NativeConstBufferView, reserved) == 48);
static_assert(alignof(NativeMutableBufferView) == alignof(uint64_t));
static_assert(sizeof(NativeMutableBufferView) == 80);
static_assert(offsetof(NativeMutableBufferView, data) == 16);
static_assert(offsetof(NativeMutableBufferView, reserved) == 48);
static_assert(alignof(NativeNumericMatrixDesc) == alignof(uint64_t));
static_assert(sizeof(NativeNumericMatrixDesc) == 208);
static_assert(offsetof(NativeNumericMatrixDesc, route) == 8);
static_assert(offsetof(NativeNumericMatrixDesc, layout) == 112);
static_assert(offsetof(NativeNumericMatrixDesc, rows) == 120);
static_assert(offsetof(NativeNumericMatrixDesc, bytes) == 136);
static_assert(offsetof(NativeNumericMatrixDesc, reserved) == 144);
static_assert(alignof(NativeNumericLaunchProtocol) == alignof(uint64_t));
static_assert(sizeof(NativeNumericLaunchProtocol) == 184);
static_assert(offsetof(NativeNumericLaunchProtocol, route) == 8);
static_assert(offsetof(NativeNumericLaunchProtocol, base) == 112);
static_assert(offsetof(NativeNumericLaunchProtocol, reserved) == 120);
static_assert(alignof(NativeNumericResultTable) == alignof(uint64_t));
static_assert(sizeof(NativeNumericResultTable) == 224);
static_assert(offsetof(NativeNumericResultTable, capacity) == 24);
static_assert(offsetof(NativeNumericResultTable, metric_values) == 48);
static_assert(offsetof(NativeNumericResultTable, reserved) == 160);

enum class PayloadAbiSurface {
    None,
    TypedPrecision11,
    GenericNumericRoute11,
};

const char* payload_surface_name(PayloadAbiSurface surface) {
    switch (surface) {
    case PayloadAbiSurface::TypedPrecision11: return "precision-typed-v1.1";
    case PayloadAbiSurface::GenericNumericRoute11: return "numeric-route-v2";
    default: return "none";
    }
}

struct PayloadRoute {
    uint32_t profile = 0;
    uint32_t storage_dtype = 0;
    uint32_t result_dtype = 0;
    NativeNumericRoute generic{};
};

struct PayloadApi {
    using TypedCapabilities = int (*)(uint32_t, NativePrecisionCapabilities*);
    using TypedAlloc = int (*)(uint32_t, const NativePrecisionMatrixDesc*, GafimeGpuMatrix*);
    using TypedUploadF32 = int (*)(GafimeGpuMatrix, const float*, const float*, uint64_t, uint32_t);
    using TypedUploadF64 = int (*)(GafimeGpuMatrix, const double*, const double*, uint64_t, uint32_t);
    using TypedUpdateF32 = int (*)(GafimeGpuMatrix, const float*, uint64_t);
    using TypedUpdateF64 = int (*)(GafimeGpuMatrix, const double*, uint64_t);
    using TypedExecuteF32 = int (*)(GafimeGpuMatrix, const NativePrecisionLaunchProtocol*, GafimeResultTable*);
    using TypedExecuteF64 = int (*)(GafimeGpuMatrix, const NativePrecisionLaunchProtocol*, NativeResultTableF64*);
    using TypedMemory = int (*)(GafimeGpuMatrix, const NativePrecisionLaunchProtocol*, uint64_t*);
    using TypedFree = void (*)(GafimeGpuMatrix);

    using GenericRoutes = int (*)(uint32_t, uint32_t, uint32_t, NativeNumericRoute*, uint32_t, uint32_t*);
    using GenericAlloc = int (*)(uint32_t, const NativeNumericMatrixDesc*, GafimeGpuMatrix*);
    using GenericUpload = int (*)(GafimeGpuMatrix, const NativeNumericRoute*,
                                  const NativeConstBufferView*, const NativeConstBufferView*,
                                  uint64_t, uint32_t);
    using GenericUpdate = int (*)(GafimeGpuMatrix, const NativeNumericRoute*,
                                  const NativeConstBufferView*, uint64_t);
    using GenericExecute = int (*)(GafimeGpuMatrix, const NativeNumericLaunchProtocol*,
                                   NativeNumericResultTable*);
    using GenericMemory = int (*)(GafimeGpuMatrix, const NativeNumericLaunchProtocol*, uint64_t*);
    using GenericFree = int (*)(GafimeGpuMatrix);

    void* handle = nullptr;
    PayloadAbiSurface surface = PayloadAbiSurface::None;
    PayloadResolution resolution;

    TypedCapabilities typed_capabilities = nullptr;
    TypedAlloc typed_alloc = nullptr;
    TypedUploadF32 typed_upload_f32 = nullptr;
    TypedUploadF64 typed_upload_f64 = nullptr;
    TypedUpdateF32 typed_update_f32 = nullptr;
    TypedUpdateF64 typed_update_f64 = nullptr;
    TypedExecuteF32 typed_execute_f32 = nullptr;
    TypedExecuteF64 typed_execute_f64 = nullptr;
    TypedMemory typed_memory = nullptr;
    TypedFree typed_free = nullptr;

    GenericRoutes generic_routes = nullptr;
    GenericAlloc generic_alloc = nullptr;
    GenericUpload generic_upload = nullptr;
    GenericUpdate generic_update = nullptr;
    GenericExecute generic_execute = nullptr;
    GenericMemory generic_memory = nullptr;
    GenericFree generic_free = nullptr;

    ~PayloadApi() {
#if !defined(_WIN32)
        if (handle != nullptr) dlclose(handle);
#endif
    }

    PayloadApi() = default;
    PayloadApi(const PayloadApi&) = delete;
    PayloadApi& operator=(const PayloadApi&) = delete;
    PayloadApi(PayloadApi&& other) noexcept
        : handle(std::exchange(other.handle, nullptr)), surface(other.surface),
          resolution(std::move(other.resolution)), typed_capabilities(other.typed_capabilities),
          typed_alloc(other.typed_alloc), typed_upload_f32(other.typed_upload_f32),
          typed_upload_f64(other.typed_upload_f64), typed_update_f32(other.typed_update_f32),
          typed_update_f64(other.typed_update_f64), typed_execute_f32(other.typed_execute_f32),
          typed_execute_f64(other.typed_execute_f64), typed_memory(other.typed_memory),
          typed_free(other.typed_free), generic_routes(other.generic_routes),
          generic_alloc(other.generic_alloc), generic_upload(other.generic_upload),
          generic_update(other.generic_update), generic_execute(other.generic_execute),
          generic_memory(other.generic_memory), generic_free(other.generic_free) {}
    PayloadApi& operator=(PayloadApi&& other) noexcept {
        if (this == &other) return *this;
#if !defined(_WIN32)
        if (handle != nullptr) dlclose(handle);
#endif
        handle = std::exchange(other.handle, nullptr);
        surface = other.surface;
        resolution = std::move(other.resolution);
        typed_capabilities = other.typed_capabilities;
        typed_alloc = other.typed_alloc;
        typed_upload_f32 = other.typed_upload_f32;
        typed_upload_f64 = other.typed_upload_f64;
        typed_update_f32 = other.typed_update_f32;
        typed_update_f64 = other.typed_update_f64;
        typed_execute_f32 = other.typed_execute_f32;
        typed_execute_f64 = other.typed_execute_f64;
        typed_memory = other.typed_memory;
        typed_free = other.typed_free;
        generic_routes = other.generic_routes;
        generic_alloc = other.generic_alloc;
        generic_upload = other.generic_upload;
        generic_update = other.generic_update;
        generic_execute = other.generic_execute;
        generic_memory = other.generic_memory;
        generic_free = other.generic_free;
        return *this;
    }
};

#if !defined(_WIN32)
template <typename Function>
Function optional_payload_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    (void)dlerror();
    return reinterpret_cast<Function>(symbol);
}
#endif

PayloadApi open_payload(const Options& options) {
    PayloadApi api;
    const std::string path = options.payload_path.empty()
        ? env_value("GAFIME_CUDA_V1_LIB") : options.payload_path;
    if (path.empty()) return api;
    api.resolution.status = "missing";
    if (std::ifstream(path).fail()) {
        api.resolution.detail = "payload path is not a readable file";
        fail(api.resolution.detail);
    }
#if defined(_WIN32)
    api.resolution.status = "unsupported";
    api.resolution.detail = "CUDA native payload dlsym adapter is unavailable on Windows";
    fail(api.resolution.detail);
#else
    api.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr) {
        const char* error = dlerror();
        api.resolution.status = "unresolved";
        api.resolution.detail = error == nullptr ? "dlopen failed" : error;
        fail(api.resolution.detail);
    }

    const auto has = [&](const char* name) {
        return optional_payload_symbol<void*>(api.handle, name) != nullptr;
    };
    const std::array<const char*, 10> generic_names = {
        "gafime_gpu_numeric_routes_v2", "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_v2", "gafime_gpu_matrix_update_target_v2",
        "gafime_gpu_execute_v2", "gafime_gpu_execution_memory_peak_v2",
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_pvalues_v2",
        "gafime_gpu_interaction_diagnostics_v2", "gafime_gpu_matrix_free_v2",
    };
    const std::array<const char*, 11> typed_names = {
        "gafime_gpu_precision_capabilities", "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_f32_v2", "gafime_gpu_matrix_upload_f64_v2",
        "gafime_gpu_matrix_update_target_f32_v2", "gafime_gpu_matrix_update_target_f64_v2",
        "gafime_gpu_execute_f32_v2", "gafime_gpu_execute_f64_v2",
        "gafime_gpu_execution_memory_peak_v2", "gafime_gpu_interaction_diagnostics",
        "gafime_gpu_matrix_free",
    };
    const bool generic_marker = has(generic_names[0]);
    const bool generic_complete = std::all_of(
        generic_names.begin(), generic_names.end(), has);
    const bool typed_marker = has(typed_names[0]);
    const bool typed_complete = std::all_of(typed_names.begin(), typed_names.end(), has);
    if (generic_marker) {
        if (!generic_complete) {
            fail("payload advertises generic numeric ABI but its required CUDA symbols are incomplete");
        }
        api.surface = PayloadAbiSurface::GenericNumericRoute11;
        api.resolution.abi_surface = payload_surface_name(api.surface);
        api.resolution.symbols.assign(generic_names.begin(), generic_names.end());
        api.generic_routes = optional_payload_symbol<PayloadApi::GenericRoutes>(
            api.handle, generic_names[0]);
        api.generic_alloc = optional_payload_symbol<PayloadApi::GenericAlloc>(
            api.handle, generic_names[1]);
        api.generic_upload = optional_payload_symbol<PayloadApi::GenericUpload>(
            api.handle, generic_names[2]);
        api.generic_update = optional_payload_symbol<PayloadApi::GenericUpdate>(
            api.handle, generic_names[3]);
        api.generic_execute = optional_payload_symbol<PayloadApi::GenericExecute>(
            api.handle, generic_names[4]);
        api.generic_memory = optional_payload_symbol<PayloadApi::GenericMemory>(
            api.handle, generic_names[5]);
        api.generic_free = optional_payload_symbol<PayloadApi::GenericFree>(
            api.handle, generic_names[9]);
    } else {
        if (!typed_marker || !typed_complete) {
            fail("payload exposes neither a complete generic numeric ABI nor the complete typed precision ABI");
        }
        api.surface = PayloadAbiSurface::TypedPrecision11;
        api.resolution.abi_surface = payload_surface_name(api.surface);
        api.resolution.symbols.assign(typed_names.begin(), typed_names.end());
        api.typed_capabilities = optional_payload_symbol<PayloadApi::TypedCapabilities>(
            api.handle, typed_names[0]);
        api.typed_alloc = optional_payload_symbol<PayloadApi::TypedAlloc>(api.handle, typed_names[1]);
        api.typed_upload_f32 = optional_payload_symbol<PayloadApi::TypedUploadF32>(
            api.handle, typed_names[2]);
        api.typed_upload_f64 = optional_payload_symbol<PayloadApi::TypedUploadF64>(
            api.handle, typed_names[3]);
        api.typed_update_f32 = optional_payload_symbol<PayloadApi::TypedUpdateF32>(
            api.handle, typed_names[4]);
        api.typed_update_f64 = optional_payload_symbol<PayloadApi::TypedUpdateF64>(
            api.handle, typed_names[5]);
        api.typed_execute_f32 = optional_payload_symbol<PayloadApi::TypedExecuteF32>(
            api.handle, typed_names[6]);
        api.typed_execute_f64 = optional_payload_symbol<PayloadApi::TypedExecuteF64>(
            api.handle, typed_names[7]);
        api.typed_memory = optional_payload_symbol<PayloadApi::TypedMemory>(api.handle, typed_names[8]);
        api.typed_free = optional_payload_symbol<PayloadApi::TypedFree>(api.handle, typed_names[10]);
    }
    api.resolution.status = "resolved";
    return api;
#endif
}

std::vector<std::string> observed_environment() {
    const std::array<const char*, 13> keys = {
        "GAFIME_CUDA_V1_LIB", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "GAFIME_WHEEL_PATH",
        "GAFIME_NATIVE_AFFINITY",
    };
    std::vector<std::string> values;
    for (const char* key : keys) {
        const std::string value = env_value(key);
        if (!value.empty()) values.push_back(std::string(key) + "=" + value);
    }
    return values;
}

std::vector<int> observed_affinity() {
    std::vector<int> cpus;
#if !defined(_WIN32)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
        }
    }
#endif
    return cpus;
}

uint32_t parse_u32(const char* text, const char* option) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
        fail(std::string("invalid ") + option + " value");
    }
    return static_cast<uint32_t>(value);
}

uint64_t parse_u64(const char* text, const char* option) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') fail(std::string("invalid ") + option + " value");
    return static_cast<uint64_t>(value);
}

std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> result;
    std::string current;
    for (const char character : value) {
        if (character == ',') {
            if (!current.empty()) result.push_back(current);
            current.clear();
        } else if (character != ' ' && character != '\t' && character != '\n' && character != '\r') {
            current += character;
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

void validate_profiles(const std::vector<std::string>& profiles) {
    if (profiles.empty()) fail("at least one --profiles value is required");
    for (const std::string& profile : profiles) {
        if (profile != "fp32" && profile != "mixed" && profile != "fp64") {
            fail("unsupported precision profile: " + profile);
        }
    }
    if (std::set<std::string>(profiles.begin(), profiles.end()).size() != profiles.size()) {
        fail("duplicate precision profile in --profiles");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto value_for = [&](const char* option) -> const char* {
            if (index + 1 >= argc) fail(std::string("missing value for ") + option);
            return argv[++index];
        };
        if (argument == "--workload") {
            options.workload = value_for("--workload");
        } else if (argument == "--rows") {
            options.rows = parse_u32(value_for("--rows"), "--rows");
        } else if (argument == "--features") {
            options.features = parse_u32(value_for("--features"), "--features");
        } else if (argument == "--candidates") {
            options.candidates = parse_u32(value_for("--candidates"), "--candidates");
        } else if (argument == "--arity") {
            options.arity = parse_u32(value_for("--arity"), "--arity");
        } else if (argument == "--mi-bins") {
            options.mi_bins = parse_u32(value_for("--mi-bins"), "--mi-bins");
        } else if (argument == "--top-k") {
            options.top_k = parse_u32(value_for("--top-k"), "--top-k");
        } else if (argument == "--warmups") {
            options.warmups = parse_u32(value_for("--warmups"), "--warmups");
        } else if (argument == "--repeats") {
            options.repeats = parse_u32(value_for("--repeats"), "--repeats");
        } else if (argument == "--order-repetitions") {
            options.order_repetitions = parse_u32(value_for("--order-repetitions"), "--order-repetitions");
        } else if (argument == "--seed") {
            options.seed = parse_u64(value_for("--seed"), "--seed");
        } else if (argument == "--profiles" || argument == "--profile") {
            options.requested_profiles = split_csv(value_for("--profiles"));
        } else if (argument == "--json") {
            options.json_path = value_for("--json");
        } else if (argument == "--csv") {
            options.csv_path = value_for("--csv");
        } else if (argument == "--payload") {
            options.payload_path = value_for("--payload");
        } else if (argument == "--wheel") {
            options.wheel_path = value_for("--wheel");
        } else if (argument == "--source-root") {
            options.source_root = value_for("--source-root");
        } else if (argument == "--harness-source-root") {
            options.harness_source_root = value_for("--harness-source-root");
        } else if (argument == "--input-policy") {
            options.input_policy = value_for("--input-policy");
        } else if (argument == "--variant") {
            options.variant = value_for("--variant");
        } else if (argument == "--ab-block") {
            options.ab_block = static_cast<int64_t>(parse_u32(value_for("--ab-block"), "--ab-block"));
        } else if (argument == "--variant-sequence") {
            options.variant_sequence = split_csv(value_for("--variant-sequence"));
        } else if (argument == "--canonical-evidence") {
            options.canonical_evidence_path = value_for("--canonical-evidence");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: " << argv[0]
                << " [--workload NAME] [--rows N] [--features N] [--candidates N]"
                << " [--arity 1..5] [--mi-bins N] [--top-k N]"
                << " [--profiles fp32,mixed,fp64] [--order-repetitions N]"
                << " [--warmups N] [--repeats N] [--seed N]"
                << " [--payload PATH] [--wheel PATH] [--source-root PATH]"
                << " [--harness-source-root PATH]"
                << " [--input-policy common-f64|native] [--variant NAME]"
                << " [--ab-block N --variant-sequence baseline,candidate]"
                << " [--canonical-evidence PATH] [--json PATH] [--csv PATH]\n";
            std::exit(0);
        } else {
            fail("unknown option: " + std::string(argument));
        }
    }
    validate_profiles(options.requested_profiles);
    if (options.input_policy != "common-f64" && options.input_policy != "native") {
        fail("--input-policy must be common-f64 or native");
    }
    const bool any_schedule_field = !options.variant.empty() || options.ab_block >= 0 ||
        !options.variant_sequence.empty();
    if (any_schedule_field &&
        (options.variant.empty() || options.ab_block < 0 || options.variant_sequence.size() != 2 ||
         std::find(options.variant_sequence.begin(), options.variant_sequence.end(),
                   options.variant) == options.variant_sequence.end())) {
        fail("native A/B scheduling requires --variant, --ab-block, and a two-entry "
             "--variant-sequence containing that variant");
    }
    if (options.workload.empty() || options.rows == 0 || options.features == 0 ||
        options.candidates == 0 || options.arity == 0 || options.arity > 5 ||
        options.arity > options.features || options.mi_bins == 0 || options.mi_bins > 96 ||
        options.top_k == 0 || options.top_k > options.candidates ||
        options.order_repetitions < kMinimumOrderRepetitions || options.warmups < 10 ||
        options.repeats < 30) {
        fail("workload/features/candidates must be nonzero, arity must be 1..5 and <= features, "
             "mi-bins must be 1..96, top-k must be 1..candidates, and timing requires "
             "order-repetitions >= 5, warmups >= 10, repeats >= 30");
    }
    return options;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2u;
    if ((values.size() & 1u) != 0) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
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
    mix(record.profile);
    mix(record.operation);
    mix(record.metric);
    for (const std::string& value : record.profile_order) mix(value);
    hash ^= record.order_index;
    hash *= 1099511628211ULL;
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
SampledValues time_host_synchronized(
    uint32_t warmups, uint32_t repeats, Fn&& operation
) {
    check_cuda(cudaDeviceSynchronize(), "host timing initial synchronize");
    for (uint32_t index = 0; index < warmups; ++index) {
        check_cuda(cudaDeviceSynchronize(), "host timing warmup pre-synchronize");
        operation();
        check_cuda(cudaDeviceSynchronize(), "host timing warmup synchronize");
    }
    auto measure = [&](uint32_t loop_count) {
        check_cuda(cudaDeviceSynchronize(), "host timing pre-synchronize");
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t loop = 0; loop < loop_count; ++loop) operation();
        check_cuda(cudaDeviceSynchronize(), "host timing synchronize");
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
    SampledValues result;
    result.loop_count_per_sample = loop_count;
    result.samples_us.reserve(repeats);
    result.raw_samples_us.reserve(repeats);
    result.loop_counts_per_sample.reserve(repeats);
    for (uint32_t index = 0; index < repeats; ++index) {
        uint32_t sample_loop_count = loop_count;
        double raw_us = measure(sample_loop_count);
        while (raw_us < kSampleRegionTargetUs && sample_loop_count < kMaxLoopCount) {
            sample_loop_count = sample_loop_count > kMaxLoopCount / 2
                ? kMaxLoopCount : sample_loop_count * 2;
            raw_us = measure(sample_loop_count);
        }
        loop_count = std::max(loop_count, sample_loop_count);
        result.loop_counts_per_sample.push_back(sample_loop_count);
        result.raw_samples_us.push_back(raw_us);
        result.samples_us.push_back(std::max(1.0e-6, raw_us / sample_loop_count));
    }
    result.loop_count_per_sample = loop_count;
    return result;
}

template <typename Fn>
SampledValues time_cuda_events(
    cudaStream_t stream, uint32_t warmups, uint32_t repeats, Fn&& operation
) {
    for (uint32_t index = 0; index < warmups; ++index) {
        check_status(operation(), "CUDA event warmup operation");
    }
    check_cuda(cudaStreamSynchronize(stream), "CUDA event warmup synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    try {
        auto measure = [&](uint32_t loop_count) {
            check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
            for (uint32_t loop = 0; loop < loop_count; ++loop) {
                check_status(operation(), "CUDA event operation");
            }
            check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            float elapsed_ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
            return std::max(1.0e-6, static_cast<double>(elapsed_ms) * 1000.0);
        };
        uint32_t loop_count = 1;
        double calibration_us = measure(loop_count);
        while (calibration_us < kSampleRegionCalibrationTargetUs &&
               loop_count < kMaxLoopCount) {
            loop_count = loop_count > kMaxLoopCount / 2 ? kMaxLoopCount : loop_count * 2;
            calibration_us = measure(loop_count);
        }
        SampledValues result;
        result.loop_count_per_sample = loop_count;
        result.samples_us.reserve(repeats);
        result.raw_samples_us.reserve(repeats);
        result.loop_counts_per_sample.reserve(repeats);
        for (uint32_t index = 0; index < repeats; ++index) {
            uint32_t sample_loop_count = loop_count;
            double raw_us = measure(sample_loop_count);
            while (raw_us < kSampleRegionTargetUs && sample_loop_count < kMaxLoopCount) {
                sample_loop_count = sample_loop_count > kMaxLoopCount / 2
                    ? kMaxLoopCount : sample_loop_count * 2;
                raw_us = measure(sample_loop_count);
            }
            loop_count = std::max(loop_count, sample_loop_count);
            result.loop_counts_per_sample.push_back(sample_loop_count);
            result.raw_samples_us.push_back(raw_us);
            result.samples_us.push_back(std::max(1.0e-6, raw_us / sample_loop_count));
        }
        result.loop_count_per_sample = loop_count;
        check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
        check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        return result;
    } catch (...) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }
}

template <GafimePrecisionProfile Profile>
struct ProfileTypes;

template <>
struct ProfileTypes<GAFIME_PRECISION_FP32> {
    using Storage = float;
    using Accumulation = float;
    using Result = float;
    static constexpr const char* name = "fp32";
};

template <>
struct ProfileTypes<GAFIME_PRECISION_MIXED> {
    using Storage = float;
    using Accumulation = double;
    using Result = double;
    static constexpr const char* name = "mixed";
};

template <>
struct ProfileTypes<GAFIME_PRECISION_FP64> {
    using Storage = double;
    using Accumulation = double;
    using Result = double;
    static constexpr const char* name = "fp64";
};

GafimePrecisionProfile profile_id(const std::string& profile) {
    if (profile == "fp32") return GAFIME_PRECISION_FP32;
    if (profile == "mixed") return GAFIME_PRECISION_MIXED;
    if (profile == "fp64") return GAFIME_PRECISION_FP64;
    fail("unknown profile: " + profile);
}

template <typename Storage>
struct HostInputs;

template <typename Storage>
HostInputs<Storage> make_inputs(
    uint64_t rows, uint32_t features, uint32_t arity, uint32_t candidates);

template <typename Storage>
HostInputs<Storage> make_inputs_for_policy(const Options& options);

template <typename Storage>
void append_cuda_profile_identity(
    std::ostringstream& output,
    const Options& options,
    bool& first,
    const char* profile_name
) {
    const HostInputs<Storage> inputs = make_inputs_for_policy<Storage>(options);
    if (!first) output << ',';
    first = false;
    output << '"' << profile_name << "\":{\"storage_dtype\":\""
           << (std::is_same_v<Storage, float> ? "float32" : "float64")
           << "\",\"features_sha256\":\"" << sha256_vector(inputs.features)
           << "\",\"target_sha256\":\"" << sha256_vector(inputs.target)
           << "\",\"means_sha256\":\"" << sha256_vector(inputs.means)
           << "\",\"combos_sha256\":\"" << sha256_vector(inputs.combos)
           << "\",\"matrix_shape\":[" << options.rows << ',' << options.features
           << "],\"layout\":\"column_major\"}";
}

std::string cuda_dataset_identity_json(const Options& options);

std::vector<std::vector<std::string>> profile_orders(
    const std::vector<std::string>& requested, uint64_t seed
) {
    static constexpr std::array<std::string_view, 3> canonical_profiles = {
        "fp32", "mixed", "fp64",
    };
    std::vector<std::string> canonical_requested;
    for (const std::string_view profile : canonical_profiles) {
        if (std::find(requested.begin(), requested.end(), profile) != requested.end()) {
            canonical_requested.emplace_back(profile);
        }
    }
    if (canonical_requested.size() != requested.size()) {
        fail("requested precision set could not be canonicalized");
    }
    std::vector<std::vector<std::string>> orders;
    std::vector<size_t> permutation(canonical_requested.size());
    std::iota(permutation.begin(), permutation.end(), 0u);
    do {
        std::vector<std::string> order;
        order.reserve(permutation.size());
        for (const size_t index : permutation) order.push_back(canonical_requested[index]);
        orders.push_back(std::move(order));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    std::mt19937_64 generator(seed);
    std::shuffle(orders.begin(), orders.end(), generator);
    return orders;
}

template <GafimePrecisionProfile Profile>
struct Buffers {
    using Types = ProfileTypes<Profile>;
    using Storage = typename Types::Storage;
    using Result = typename Types::Result;

    Storage* features = nullptr;
    Storage* target = nullptr;
    Storage* means = nullptr;
    uint32_t* combos = nullptr;
    uint32_t* metric_ids = nullptr;
    Result* metric_values = nullptr;
    uint64_t* target_ranks = nullptr;
    void* target_stats = nullptr;
    void* feature_stats = nullptr;

    void allocate(
        uint64_t rows, uint32_t features_count, uint32_t candidates, uint32_t arity,
        const gafime_cuda_v1::CudaPrecisionKernelSet* set, uint32_t top_k,
        uint32_t partial_blocks
    ) {
        const size_t feature_count = static_cast<size_t>(rows) * features_count;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&features), feature_count * sizeof(Storage)), "cudaMalloc(features)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&target), static_cast<size_t>(rows) * sizeof(Storage)), "cudaMalloc(target)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&means), static_cast<size_t>(features_count) * sizeof(Storage)), "cudaMalloc(means)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&combos), static_cast<size_t>(candidates) * arity * sizeof(uint32_t)), "cudaMalloc(combos)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&metric_ids), sizeof(uint32_t)), "cudaMalloc(metric_ids)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&metric_values), static_cast<size_t>(candidates) * sizeof(Result)), "cudaMalloc(metric_values)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&target_ranks), static_cast<size_t>(rows) * sizeof(uint64_t)), "cudaMalloc(target_ranks)");
        check_cuda(cudaMalloc(&target_stats, set->target_stats_bytes), "cudaMalloc(target_stats)");
        check_cuda(cudaMalloc(&feature_stats, static_cast<size_t>(features_count) * set->feature_stats_bytes), "cudaMalloc(feature_stats)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&selected_indices), static_cast<size_t>(top_k) * sizeof(uint32_t)), "cudaMalloc(selected_indices)");
        check_cuda(cudaMalloc(&partial_scores, static_cast<size_t>(top_k) * partial_blocks * sizeof(Result)), "cudaMalloc(partial_scores)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&partial_indices), static_cast<size_t>(top_k) * partial_blocks * sizeof(uint32_t)), "cudaMalloc(partial_indices)");
        check_cuda(cudaMalloc(&selected_metric_values, static_cast<size_t>(top_k) * sizeof(Result)), "cudaMalloc(selected_metric_values)");
    }

    void release() {
        cudaFree(feature_stats);
        cudaFree(selected_metric_values);
        cudaFree(partial_indices);
        cudaFree(partial_scores);
        cudaFree(selected_indices);
        cudaFree(target_stats);
        cudaFree(target_ranks);
        cudaFree(metric_values);
        cudaFree(metric_ids);
        cudaFree(combos);
        cudaFree(means);
        cudaFree(target);
        cudaFree(features);
        features = nullptr;
        target = nullptr;
        means = nullptr;
        combos = nullptr;
        metric_ids = nullptr;
        metric_values = nullptr;
        target_ranks = nullptr;
        target_stats = nullptr;
        feature_stats = nullptr;
        selected_metric_values = nullptr;
        partial_indices = nullptr;
        partial_scores = nullptr;
        selected_indices = nullptr;
    }

    uint32_t* selected_indices = nullptr;
    void* partial_scores = nullptr;
    uint32_t* partial_indices = nullptr;
    void* selected_metric_values = nullptr;
};

template <typename Storage>
struct HostInputs {
    std::vector<Storage> features;
    std::vector<Storage> target;
    std::vector<Storage> means;
    std::vector<uint32_t> combos;
    std::vector<uint32_t> metric_ids{GAFIME_METRIC_PEARSON};
};

std::vector<std::vector<uint32_t>> make_combinations(uint32_t features, uint32_t arity) {
    std::vector<std::vector<uint32_t>> combinations;
    std::vector<uint32_t> current(arity);
    std::function<void(uint32_t, uint32_t)> visit = [&](uint32_t start, uint32_t depth) {
        if (depth == arity) {
            combinations.push_back(current);
            return;
        }
        for (uint32_t value = start; value <= features - (arity - depth); ++value) {
            current[depth] = value;
            visit(value + 1, depth + 1);
        }
    };
    visit(0, 0);
    return combinations;
}

template <typename Storage>
HostInputs<Storage> make_inputs(
    uint64_t rows, uint32_t features, uint32_t arity, uint32_t candidates
) {
    HostInputs<Storage> inputs;
    inputs.features.resize(static_cast<size_t>(rows) * features);
    inputs.target.resize(static_cast<size_t>(rows));
    inputs.means.resize(features);
    const auto all_combinations = make_combinations(features, arity);
    if (candidates > all_combinations.size()) fail("candidates exceeds combinations for features/arity");
    inputs.combos.reserve(static_cast<size_t>(candidates) * arity);
    for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
        inputs.combos.insert(
            inputs.combos.end(), all_combinations[candidate].begin(), all_combinations[candidate].end());
    }
    for (uint64_t row = 0; row < rows; ++row) {
        const double target_value = 0.25 +
            static_cast<double>((row * 104729u + row / 7u + 17u) % 100003u) / 100003.0;
        inputs.target[static_cast<size_t>(row)] = static_cast<Storage>(target_value);
        for (uint32_t column = 0; column < features; ++column) {
            const double value = 0.1 + static_cast<double>(
                (row * (8191u + 2u * column) + row / (3u + column % 11u) + 97u * column) %
                100019u) / 100019.0;
            inputs.features[static_cast<size_t>(column) * rows + row] = static_cast<Storage>(value);
        }
    }
    for (uint32_t column = 0; column < features; ++column) {
        double sum = 0.0;
        for (uint64_t row = 0; row < rows; ++row) {
            sum += static_cast<double>(inputs.features[static_cast<size_t>(column) * rows + row]);
        }
        inputs.means[column] = static_cast<Storage>(sum / static_cast<double>(rows));
    }
    return inputs;
}

template <typename Storage>
HostInputs<Storage> make_inputs_for_policy(const Options& options) {
    if (options.input_policy == "native") {
        return make_inputs<Storage>(
            options.rows, options.features, options.arity, options.candidates);
    }
    const HostInputs<double> common = make_inputs<double>(
        options.rows, options.features, options.arity, options.candidates);
    HostInputs<Storage> converted;
    converted.features.resize(common.features.size());
    converted.target.resize(common.target.size());
    converted.means.resize(common.means.size());
    converted.combos = common.combos;
    std::transform(
        common.features.begin(), common.features.end(), converted.features.begin(),
        [](double value) { return static_cast<Storage>(value); });
    std::transform(
        common.target.begin(), common.target.end(), converted.target.begin(),
        [](double value) { return static_cast<Storage>(value); });
    for (uint32_t column = 0; column < options.features; ++column) {
        double sum = 0.0;
        for (uint64_t row = 0; row < options.rows; ++row) {
            sum += static_cast<double>(
                converted.features[static_cast<size_t>(column) * options.rows + row]);
        }
        converted.means[column] = static_cast<Storage>(
            sum / static_cast<double>(options.rows));
    }
    return converted;
}

std::string cuda_dataset_identity_json(const Options& options) {
    const HostInputs<double> canonical = make_inputs<double>(
        options.rows, options.features, options.arity, options.candidates);
    std::ostringstream feature_names;
    for (uint32_t column = 0; column < options.features; ++column) {
        if (column != 0) feature_names << '\0';
        feature_names << 'x' << column;
    }
    const std::string feature_names_value = feature_names.str();
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"algorithm\":\"gafime.cuda.native_timing.inputs.v2\""
           << ",\"input_policy\":\"" << json_escape(options.input_policy) << "\""
           << ",\"policy_detail\":\""
           << (options.input_policy == "common-f64"
                ? "every profile begins from the same deterministic float64 source"
                : "fp32 and mixed begin from float32 while fp64 begins from float64")
           << "\""
           << ",\"generator\":\"make_inputs.v2\""
           << ",\"source_scalar_dtype\":\""
           << (options.input_policy == "common-f64" ? "float64" : "profile-native")
           << "\""
           << ",\"rows\":" << options.rows
           << ",\"features\":" << options.features
           << ",\"candidates\":" << options.candidates
           << ",\"arity\":" << options.arity
           << ",\"mi_bins\":" << options.mi_bins
           << ",\"combination_order\":\"lexicographic\""
           << ",\"feature_layout\":\"column_major\""
           << ",\"matrix_sha256\":\"" << sha256_vector(canonical.features)
           << "\",\"target_sha256\":\"" << sha256_vector(canonical.target)
           << "\",\"feature_names_sha256\":\""
           << sha256_bytes(feature_names_value.data(), feature_names_value.size())
           << "\",\"matrix_shape\":[" << options.rows << ',' << options.features
           << "],\"target_shape\":[" << options.rows
           << "],\"matrix_dtype\":\""
           << (options.input_policy == "common-f64" ? "float64" : "profile-native")
           << "\",\"target_dtype\":\""
           << (options.input_policy == "common-f64" ? "float64" : "profile-native")
           << "\""
           << ",\"profiles\":{";
    bool first = true;
    for (const std::string& profile : options.requested_profiles) {
        if (profile == "fp32") {
            append_cuda_profile_identity<float>(output, options, first, "fp32");
        } else if (profile == "mixed") {
            // Mixed stores the same profile-native float inputs as fp32 but
            // accumulates and returns double results.
            const HostInputs<float> inputs = make_inputs_for_policy<float>(options);
            if (!first) output << ',';
            first = false;
            output << "\"mixed\":{\"storage_dtype\":\"float32\","
                   << "\"features_sha256\":\"" << sha256_vector(inputs.features)
                   << "\",\"target_sha256\":\"" << sha256_vector(inputs.target)
                   << "\",\"means_sha256\":\"" << sha256_vector(inputs.means)
                   << "\",\"combos_sha256\":\"" << sha256_vector(inputs.combos)
                   << "\",\"matrix_shape\":[" << options.rows << ',' << options.features
                   << "],\"layout\":\"column_major\",\"accumulation_dtype\":\"float64\"}";
        } else if (profile == "fp64") {
            append_cuda_profile_identity<double>(output, options, first, "fp64");
        }
    }
    output << "}}";
    return output.str();
}

constexpr uint32_t kNativeAbi11 = GAFIME_PRECISION_ABI_VERSION;

void check_payload_status(int status, const char* operation) {
    if (status != GAFIME_STATUS_OK) {
        fail(std::string(operation) + " returned payload status " + std::to_string(status));
    }
}

uint64_t payload_dtype_size(uint32_t dtype) {
    if (dtype == GAFIME_DTYPE_F32) return sizeof(float);
    if (dtype == GAFIME_DTYPE_F64) return sizeof(double);
    fail("unknown payload route dtype " + std::to_string(dtype));
}

NativeConstBufferView payload_const_view(uint32_t dtype, const void* data, uint64_t count) {
    NativeConstBufferView view{};
    view.abi_version = kNativeAbi11;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_count = count;
    view.byte_length = count * payload_dtype_size(dtype);
    view.byte_stride = payload_dtype_size(dtype);
    return view;
}

NativeMutableBufferView payload_mutable_view(uint32_t dtype, void* data, uint64_t count) {
    NativeMutableBufferView view{};
    view.abi_version = kNativeAbi11;
    view.struct_size = sizeof(view);
    view.dtype = dtype;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_capacity = count;
    view.byte_length = count * payload_dtype_size(dtype);
    view.byte_stride = payload_dtype_size(dtype);
    return view;
}

PayloadRoute typed_payload_route(uint32_t profile) {
    PayloadRoute route;
    route.profile = profile;
    route.storage_dtype = profile == GAFIME_PRECISION_FP64 ? GAFIME_DTYPE_F64 : GAFIME_DTYPE_F32;
    route.result_dtype = profile == GAFIME_PRECISION_FP32 ? GAFIME_DTYPE_F32 : GAFIME_DTYPE_F64;
    route.generic.abi_version = kNativeAbi11;
    route.generic.struct_size = sizeof(route.generic);
    route.generic.route_id = profile;
    route.generic.profile = profile;
    route.generic.storage_dtype = route.storage_dtype;
    route.generic.pointwise_dtype = route.storage_dtype;
    route.generic.reduction_dtype = route.result_dtype;
    route.generic.result_dtype = route.result_dtype;
    route.generic.overflow_policy = GAFIME_OVERFLOW_IEEE;
    return route;
}

const char* payload_profile_name(uint32_t profile) {
    switch (profile) {
    case GAFIME_PRECISION_FP32: return "fp32";
    case GAFIME_PRECISION_MIXED: return "mixed";
    case GAFIME_PRECISION_FP64: return "fp64";
    default: return "unknown";
    }
}

void validate_generic_route(const NativeNumericRoute& observed_route) {
    if (observed_route.abi_version != kNativeAbi11) {
        fail("generic CUDA payload route ABI version is not canonical 1.1");
    }
    // The route ABI promises a stable prefix through flags.  A newer minor
    // may append fields, but it may not truncate any required declaration or
    // claim a record larger than the frozen local layout.
    if (observed_route.struct_size < offsetof(NativeNumericRoute, reserved) ||
        observed_route.struct_size > sizeof(NativeNumericRoute)) {
        fail("generic CUDA payload route has an invalid struct_size prefix");
    }
    if (observed_route.struct_size >= sizeof(NativeNumericRoute) &&
        std::any_of(std::begin(observed_route.reserved), std::end(observed_route.reserved),
                    [](uint64_t value) { return value != 0; })) {
        fail("generic CUDA payload route has nonzero reserved fields");
    }
    if (observed_route.flags != 0) {
        fail("generic CUDA payload route has unsupported flags");
    }
    if (observed_route.profile < GAFIME_PRECISION_FP32 ||
        observed_route.profile > GAFIME_PRECISION_FP64) {
        fail("generic CUDA payload returned an unknown precision profile");
    }
    const PayloadRoute expected = typed_payload_route(observed_route.profile);
    if (observed_route.route_id != observed_route.profile ||
        observed_route.storage_dtype != expected.storage_dtype ||
        observed_route.pointwise_dtype != expected.generic.pointwise_dtype ||
        observed_route.reduction_dtype != expected.generic.reduction_dtype ||
        observed_route.result_dtype != expected.result_dtype ||
        observed_route.overflow_policy != GAFIME_OVERFLOW_IEEE) {
        fail(std::string("generic CUDA payload route tuple contradicts ") +
             payload_profile_name(observed_route.profile));
    }
}

std::array<PayloadRoute, 3> payload_routes(PayloadApi& api) {
    std::array<PayloadRoute, 3> routes = {
        typed_payload_route(GAFIME_PRECISION_FP32),
        typed_payload_route(GAFIME_PRECISION_MIXED),
        typed_payload_route(GAFIME_PRECISION_FP64),
    };
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        NativePrecisionCapabilities capabilities{};
        check_payload_status(api.typed_capabilities(0, &capabilities),
                             "gafime_gpu_precision_capabilities");
        if (capabilities.abi_version != GAFIME_PRECISION_ABI_VERSION ||
            capabilities.backend_kind != GAFIME_BACKEND_CUDA ||
            (capabilities.profile_mask & (GAFIME_PRECISION_PROFILE_MASK_FP32 |
                                          GAFIME_PRECISION_PROFILE_MASK_MIXED |
                                          GAFIME_PRECISION_PROFILE_MASK_FP64)) !=
                (GAFIME_PRECISION_PROFILE_MASK_FP32 |
                 GAFIME_PRECISION_PROFILE_MASK_MIXED |
                 GAFIME_PRECISION_PROFILE_MASK_FP64) ||
            (capabilities.storage_dtype_mask &
                 (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64)) !=
                (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64) ||
            (capabilities.result_dtype_mask &
                 (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64)) !=
                (GAFIME_DTYPE_MASK_F32 | GAFIME_DTYPE_MASK_F64)) {
            fail("typed CUDA payload capabilities contradict the three precision profiles");
        }
        return routes;
    }
    if (api.surface != PayloadAbiSurface::GenericNumericRoute11) {
        return {};
    }
    uint32_t count = 0;
    check_payload_status(api.generic_routes(
        0, kNativeAbi11, sizeof(NativeNumericRoute), nullptr, 0, &count),
        "gafime_gpu_numeric_routes_v2 count query");
    if (count != 3) {
        fail("generic CUDA payload advertised " + std::to_string(count) +
             " routes, expected fp32/mixed/fp64");
    }
    std::array<NativeNumericRoute, 3> observed{};
    check_payload_status(api.generic_routes(
        0, kNativeAbi11, sizeof(NativeNumericRoute), observed.data(), count, &count),
        "gafime_gpu_numeric_routes_v2 enumeration");
    if (count != observed.size()) {
        fail("generic CUDA payload returned an incomplete route enumeration");
    }
    std::array<bool, 3> seen{};
    for (const NativeNumericRoute& observed_route : observed) {
        validate_generic_route(observed_route);
        PayloadRoute& route = routes[observed_route.profile - GAFIME_PRECISION_FP32];
        if (seen[observed_route.profile - GAFIME_PRECISION_FP32]) {
            fail("generic CUDA payload returned duplicate precision routes");
        }
        seen[observed_route.profile - GAFIME_PRECISION_FP32] = true;
        route.profile = observed_route.profile;
        route.storage_dtype = observed_route.storage_dtype;
        route.result_dtype = observed_route.result_dtype;
        route.generic = observed_route;
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
        fail("generic CUDA payload did not advertise all three precision profiles");
    }
    return routes;
}

struct PayloadCallState {
    PayloadRoute route;
    std::vector<float> features_f32;
    std::vector<float> target_f32;
    std::vector<double> features_f64;
    std::vector<double> target_f64;
    std::vector<uint32_t> combos;
    std::array<uint32_t, 1> metrics{GAFIME_METRIC_PEARSON};
    GafimeShapeHint hint{};
    GafimeArityChunk chunk{};
    GafimeLaunchProtocol base{};
    NativeNumericLaunchProtocol numeric{};
    NativePrecisionLaunchProtocol typed{};
    std::vector<uint32_t> result_combos;
    std::vector<uint32_t> result_ranks;
    std::vector<uint32_t> result_families;
    std::vector<uint64_t> result_candidate_ids;
    std::vector<uint32_t> result_flags;
    std::vector<float> result_values_f32;
    std::vector<double> result_values_f64;
    GafimeResultTable typed_result_f32{};
    NativeResultTableF64 typed_result_f64{};
    NativeNumericResultTable numeric_result{};

    PayloadCallState(const Options& options, const PayloadRoute& selected_route)
        : route(selected_route) {
        if (route.storage_dtype == GAFIME_DTYPE_F32) {
            const HostInputs<float> inputs = make_inputs<float>(
                options.rows, options.features, options.arity, options.candidates);
            features_f32.resize(inputs.features.size());
            for (uint64_t row = 0; row < options.rows; ++row) {
                for (uint32_t column = 0; column < options.features; ++column) {
                    features_f32[static_cast<size_t>(row) * options.features + column] =
                        inputs.features[static_cast<size_t>(column) * options.rows + row];
                }
            }
            target_f32 = inputs.target;
            combos = inputs.combos;
        } else {
            const HostInputs<double> inputs = make_inputs_for_policy<double>(options);
            features_f64.resize(inputs.features.size());
            for (uint64_t row = 0; row < options.rows; ++row) {
                for (uint32_t column = 0; column < options.features; ++column) {
                    features_f64[static_cast<size_t>(row) * options.features + column] =
                        inputs.features[static_cast<size_t>(column) * options.rows + row];
                }
            }
            target_f64 = inputs.target;
            combos = inputs.combos;
        }
        result_combos.resize(static_cast<size_t>(options.candidates) * options.arity);
        result_ranks.resize(options.candidates);
        result_families.resize(options.candidates);
        result_candidate_ids.resize(options.candidates);
        result_flags.resize(options.candidates);
        if (route.result_dtype == GAFIME_DTYPE_F32) {
            result_values_f32.resize(options.candidates);
        } else {
            result_values_f64.resize(options.candidates);
        }

        hint.vendor_hint = options.mi_bins;
        chunk.arity = options.arity;
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.metric_mask = 1u << (GAFIME_METRIC_PEARSON - 1u);
        chunk.shape_hint_index = 0;
        chunk.combo_row_offset = 0;
        chunk.combo_count = options.candidates;
        chunk.local_chunk_id = 0;
        chunk.flags = 0;
        chunk.descriptor_offset = 0;
        chunk.descriptor_count = options.candidates;
        base.abi_version = GAFIME_ABI_VERSION;
        base.backend_kind = GAFIME_BACKEND_CUDA;
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
        base.rank.top_k = options.top_k;
        base.rank.primary_metric = GAFIME_METRIC_PEARSON;
        base.rank.descending = 1;
        base.rank.include_ties = 0;
        base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] =
            0x60000000ULL + route.profile;

        typed.abi_version = GAFIME_PRECISION_ABI_VERSION;
        typed.profile = route.profile;
        typed.base = &base;
        numeric.abi_version = kNativeAbi11;
        numeric.struct_size = sizeof(numeric);
        numeric.route = route.generic;
        numeric.base = &base;

        if (route.result_dtype == GAFIME_DTYPE_F32) {
            typed_result_f32.abi_version = GAFIME_ABI_VERSION;
            typed_result_f32.max_arity = options.arity;
            typed_result_f32.metric_count = 1;
            typed_result_f32.capacity = options.candidates;
            typed_result_f32.combo_indices = result_combos.data();
            typed_result_f32.metric_values = result_values_f32.data();
            typed_result_f32.ranks = result_ranks.data();
            typed_result_f32.families = result_families.data();
            typed_result_f32.candidate_ids = result_candidate_ids.data();
            typed_result_f32.row_flags = result_flags.data();
        } else {
            typed_result_f64.abi_version = GAFIME_PRECISION_ABI_VERSION;
            typed_result_f64.max_arity = options.arity;
            typed_result_f64.metric_count = 1;
            typed_result_f64.capacity = options.candidates;
            typed_result_f64.combo_indices = result_combos.data();
            typed_result_f64.metric_values = result_values_f64.data();
            typed_result_f64.ranks = result_ranks.data();
            typed_result_f64.families = result_families.data();
            typed_result_f64.candidate_ids = result_candidate_ids.data();
            typed_result_f64.row_flags = result_flags.data();
        }
        numeric_result.abi_version = kNativeAbi11;
        numeric_result.struct_size = sizeof(numeric_result);
        numeric_result.max_arity = options.arity;
        numeric_result.metric_count = 1;
        numeric_result.capacity = options.candidates;
        numeric_result.combo_indices = result_combos.data();
        numeric_result.metric_values = payload_mutable_view(
            route.result_dtype,
            route.result_dtype == GAFIME_DTYPE_F32
                ? static_cast<void*>(result_values_f32.data())
                : static_cast<void*>(result_values_f64.data()),
            options.candidates);
        numeric_result.ranks = result_ranks.data();
        numeric_result.families = result_families.data();
        numeric_result.candidate_ids = result_candidate_ids.data();
        numeric_result.row_flags = result_flags.data();
        reset_result();
    }

    void reset_result() {
        typed_result_f32.row_count = 0;
        typed_result_f32.flags = 0;
        typed_result_f64.row_count = 0;
        typed_result_f64.flags = 0;
        numeric_result.row_count = 0;
        numeric_result.flags = 0;
    }

    void set_metric(uint32_t metric) {
        if (metric < GAFIME_METRIC_PEARSON || metric > GAFIME_METRIC_R2) {
            fail("unsupported payload metric " + std::to_string(metric));
        }
        metrics[0] = metric;
        chunk.metric_mask = 1u << (metric - 1u);
        base.rank.primary_metric = metric;
        reset_result();
    }
};

const char* payload_metric_name(uint32_t metric) {
    switch (metric) {
    case GAFIME_METRIC_PEARSON: return "pearson";
    case GAFIME_METRIC_SPEARMAN: return "spearman";
    case GAFIME_METRIC_MUTUAL_INFO: return "mutual_info";
    case GAFIME_METRIC_R2: return "r2";
    default: return "unknown";
    }
}

NativePrecisionMatrixDesc typed_matrix_desc(
    const Options& options, const PayloadRoute& route
) {
    NativePrecisionMatrixDesc desc{};
    desc.abi_version = GAFIME_PRECISION_ABI_VERSION;
    desc.profile = route.profile;
    desc.dtype = route.storage_dtype;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = options.rows;
    desc.cols = options.features;
    desc.row_stride = options.features;
    desc.bytes = options.rows * static_cast<uint64_t>(options.features) *
        payload_dtype_size(route.storage_dtype);
    return desc;
}

NativeNumericMatrixDesc generic_matrix_desc(
    const Options& options, const PayloadRoute& route
) {
    NativeNumericMatrixDesc desc{};
    desc.abi_version = kNativeAbi11;
    desc.struct_size = sizeof(desc);
    desc.route = route.generic;
    desc.layout = GAFIME_MATRIX_ROW_MAJOR;
    desc.rows = options.rows;
    desc.cols = options.features;
    desc.row_stride = options.features;
    desc.bytes = options.rows * static_cast<uint64_t>(options.features) *
        payload_dtype_size(route.storage_dtype);
    return desc;
}

int payload_alloc(
    PayloadApi& api, const Options& options, const PayloadRoute& route, GafimeGpuMatrix* matrix
) {
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        const NativePrecisionMatrixDesc desc = typed_matrix_desc(options, route);
        return api.typed_alloc(0, &desc, matrix);
    }
    const NativeNumericMatrixDesc desc = generic_matrix_desc(options, route);
    return api.generic_alloc(0, &desc, matrix);
}

int payload_upload(PayloadApi& api, const Options& options, PayloadCallState& call,
                   GafimeGpuMatrix matrix) {
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        if (call.route.storage_dtype == GAFIME_DTYPE_F32) {
            return api.typed_upload_f32(
                matrix, call.features_f32.data(), call.target_f32.data(), options.rows,
                options.features);
        }
        return api.typed_upload_f64(
            matrix, call.features_f64.data(), call.target_f64.data(), options.rows,
            options.features);
    }
    const uint64_t feature_count = options.rows * static_cast<uint64_t>(options.features);
    const NativeConstBufferView features = call.route.storage_dtype == GAFIME_DTYPE_F32
        ? payload_const_view(call.route.storage_dtype, call.features_f32.data(), feature_count)
        : payload_const_view(call.route.storage_dtype, call.features_f64.data(), feature_count);
    const NativeConstBufferView target = call.route.storage_dtype == GAFIME_DTYPE_F32
        ? payload_const_view(call.route.storage_dtype, call.target_f32.data(), options.rows)
        : payload_const_view(call.route.storage_dtype, call.target_f64.data(), options.rows);
    return api.generic_upload(
        matrix, &call.route.generic, &features, &target, options.rows, options.features);
}

int payload_update_target(PayloadApi& api, const Options& options, PayloadCallState& call,
                          GafimeGpuMatrix matrix) {
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        if (call.route.storage_dtype == GAFIME_DTYPE_F32) {
            return api.typed_update_f32(matrix, call.target_f32.data(), options.rows);
        }
        return api.typed_update_f64(matrix, call.target_f64.data(), options.rows);
    }
    const NativeConstBufferView target = call.route.storage_dtype == GAFIME_DTYPE_F32
        ? payload_const_view(call.route.storage_dtype, call.target_f32.data(), options.rows)
        : payload_const_view(call.route.storage_dtype, call.target_f64.data(), options.rows);
    return api.generic_update(matrix, &call.route.generic, &target, options.rows);
}

int payload_execute(PayloadApi& api, PayloadCallState& call, GafimeGpuMatrix matrix) {
    call.reset_result();
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        if (call.route.result_dtype == GAFIME_DTYPE_F32) {
            return api.typed_execute_f32(matrix, &call.typed, &call.typed_result_f32);
        }
        return api.typed_execute_f64(matrix, &call.typed, &call.typed_result_f64);
    }
    return api.generic_execute(matrix, &call.numeric, &call.numeric_result);
}

int payload_memory_peak(PayloadApi& api, PayloadCallState& call, GafimeGpuMatrix matrix,
                        uint64_t* peak_bytes) {
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        return api.typed_memory(matrix, &call.typed, peak_bytes);
    }
    return api.generic_memory(matrix, &call.numeric, peak_bytes);
}

int payload_free(PayloadApi& api, GafimeGpuMatrix matrix) {
    if (matrix == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    if (api.surface == PayloadAbiSurface::TypedPrecision11) {
        api.typed_free(matrix);
        return GAFIME_STATUS_OK;
    }
    return api.generic_free(matrix);
}

template <GafimePrecisionProfile Profile>
void upload_inputs(
    const HostInputs<typename ProfileTypes<Profile>::Storage>& inputs,
    Buffers<Profile>& buffers, uint64_t rows, uint32_t features, cudaStream_t stream
) {
    using Storage = typename ProfileTypes<Profile>::Storage;
    check_cuda(cudaMemcpyAsync(
        buffers.features, inputs.features.data(),
        inputs.features.size() * sizeof(Storage), cudaMemcpyHostToDevice, stream), "H2D features");
    check_cuda(cudaMemcpyAsync(
        buffers.target, inputs.target.data(),
        static_cast<size_t>(rows) * sizeof(Storage), cudaMemcpyHostToDevice, stream), "H2D target");
    check_cuda(cudaMemcpyAsync(
        buffers.means, inputs.means.data(),
        static_cast<size_t>(features) * sizeof(Storage), cudaMemcpyHostToDevice, stream), "H2D means");
    check_cuda(cudaMemcpyAsync(
        buffers.combos, inputs.combos.data(),
        inputs.combos.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "H2D combos");
}

template <typename Storage>
__global__ void supplemental_candidate_materialization_kernel(
    const Storage* features,
    const uint32_t* combos,
    Storage* materialized,
    uint64_t rows,
    uint32_t candidates,
    uint32_t arity
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = rows * candidates;
    if (index >= total) return;
    const uint32_t candidate = static_cast<uint32_t>(index / rows);
    const uint64_t row = index % rows;
    Storage value = static_cast<Storage>(1);
    for (uint32_t slot = 0; slot < arity; ++slot) {
        value *= features[static_cast<uint64_t>(combos[candidate * arity + slot]) * rows + row];
    }
    materialized[index] = value;
}

template <typename Fn>
SampledValues time_host_only(uint32_t warmups, uint32_t repeats, Fn&& operation) {
    for (uint32_t index = 0; index < warmups; ++index) operation();
    auto measure = [&](uint32_t loop_count) {
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t loop = 0; loop < loop_count; ++loop) operation();
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
    SampledValues result;
    result.loop_count_per_sample = loop_count;
    result.samples_us.reserve(repeats);
    result.raw_samples_us.reserve(repeats);
    result.loop_counts_per_sample.reserve(repeats);
    for (uint32_t index = 0; index < repeats; ++index) {
        uint32_t sample_loop_count = loop_count;
        double raw_us = measure(sample_loop_count);
        while (raw_us < kSampleRegionTargetUs && sample_loop_count < kMaxLoopCount) {
            sample_loop_count = sample_loop_count > kMaxLoopCount / 2
                ? kMaxLoopCount : sample_loop_count * 2;
            raw_us = measure(sample_loop_count);
        }
        loop_count = std::max(loop_count, sample_loop_count);
        result.loop_counts_per_sample.push_back(sample_loop_count);
        result.raw_samples_us.push_back(raw_us);
        result.samples_us.push_back(std::max(1.0e-6, raw_us / sample_loop_count));
    }
    result.loop_count_per_sample = loop_count;
    return result;
}

template <GafimePrecisionProfile Profile>
void add_record(
    std::vector<TimingRecord>& records,
    const std::string& profile,
    uint32_t order_index,
    const std::vector<std::string>& profile_order,
    std::string operation,
    std::string metric,
    std::string clock,
    std::string timing_scope,
    bool supplemental,
    SampledValues samples,
    std::string evidence_lane = "supplemental_internal_kernel",
    std::string comparability = "direct_kernel_only",
    std::string note = {}
) {
    records.push_back(TimingRecord{
        profile, order_index, profile_order, std::move(operation), std::move(metric),
        std::move(clock), std::move(timing_scope), supplemental, std::move(evidence_lane),
        std::move(comparability), std::move(note), std::move(samples.samples_us),
        std::move(samples.raw_samples_us), std::move(samples.loop_counts_per_sample),
        samples.loop_count_per_sample,
    });
}

template <GafimePrecisionProfile Profile>
void run_profile(
    const Options& options,
    const gafime_cuda_v1::CudaKernelLaunchPolicy& policy,
    std::vector<TimingRecord>& records,
    uint32_t order_index,
    const std::vector<std::string>& profile_order
) {
    using Types = ProfileTypes<Profile>;
    using Storage = typename Types::Storage;
    using Result = typename Types::Result;
    const uint64_t rows = options.rows;
    const uint32_t features = options.features;
    const uint32_t candidates = options.candidates;
    const uint32_t arity = options.arity;
    const auto* set = gafime_cuda_v1::cuda_precision_kernel_set(Profile);
    if (set == nullptr) fail("precision kernel set is unavailable");
    const auto conversion_samples = time_host_only(
        options.warmups, options.repeats, [&]() {
            const auto converted = make_inputs_for_policy<Storage>(options);
            if (converted.features.empty()) fail("input conversion produced no features");
        });
    add_record<Profile>(records, Types::name, order_index, profile_order,
        "ingest_conversion", "none", "host_steady_clock", "host_only", true,
        conversion_samples, "supplemental_internal_kernel", "direct_kernel_only",
        options.input_policy == "common-f64"
            ? "common f64 source converted to the route storage dtype"
            : "profile-native fp32/fp64 source ownership preparation; no cross-dtype conversion is claimed");
    const auto planning_samples = time_host_only(
        options.warmups, options.repeats, [&]() {
            const auto combinations = make_combinations(features, arity);
            if (combinations.size() < candidates) fail("planning produced too few combinations");
        });
    add_record<Profile>(records, Types::name, order_index, profile_order,
        "planning", "none", "host_steady_clock", "host_only", true,
        planning_samples);
    const auto inputs = make_inputs_for_policy<Storage>(options);
    const uint32_t partial_blocks = std::min<uint32_t>(
        gafime_cuda_v1::kTopKMaxPartialBlocks,
        std::min<uint32_t>(
            1u + (candidates - 1u) / policy.threads_per_block,
            1u + (candidates - 1u) / options.top_k));
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    Buffers<Profile> buffers;
    Storage* supplemental_materialized = nullptr;

    try {
        add_record<Profile>(records, Types::name, order_index, profile_order,
            "allocation", "none", "host_chrono_cuda_sync", "host_synchronized", true,
            time_host_synchronized(options.warmups, options.repeats, [&]() {
                Buffers<Profile> temporary;
                temporary.allocate(rows, features, candidates, arity, set, options.top_k, partial_blocks);
                temporary.release();
            }));

        buffers.allocate(rows, features, candidates, arity, set, options.top_k, partial_blocks);
        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&supplemental_materialized),
            static_cast<size_t>(rows) * candidates * sizeof(Storage)),
            "cudaMalloc(supplemental_materialized)");
        add_record<Profile>(records, Types::name, order_index, profile_order,
            "h2d_upload", "none", "host_chrono_stream_sync", "host_synchronized", true,
            time_host_synchronized(options.warmups, options.repeats, [&]() {
                upload_inputs<Profile>(inputs, buffers, rows, features, stream);
                check_cuda(cudaStreamSynchronize(stream), "H2D synchronize");
            }));

        add_record<Profile>(records, Types::name, order_index, profile_order,
            "target_stat_preparation", "none", "cuda_event_stream", "device_event", true,
            time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                return set->target_stats(
                    buffers.target, rows, buffers.target_stats, policy, stream);
            }),
            "supplemental_internal_kernel", "direct_kernel_only",
            "direct target-stat preparation kernel; canonical payload-private preparation remains unobservable");
        add_record<Profile>(records, Types::name, order_index, profile_order,
            "feature_stat_preparation", "none", "cuda_event_stream", "device_event", true,
            time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                return set->feature_stats(
                    buffers.features, rows, features, buffers.feature_stats, policy, stream);
            }),
            "supplemental_internal_kernel", "direct_kernel_only",
            "direct feature-stat preparation kernel; canonical payload-private preparation remains unobservable");

        const uint32_t blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(rows) * candidates + policy.threads_per_block - 1u) /
            policy.threads_per_block);
        add_record<Profile>(records, Types::name, order_index, profile_order,
            "candidate_materialization", "none", "cuda_event_stream",
            "supplemental_internal_kernel_not_production_launch_synchronized", true,
            time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                supplemental_candidate_materialization_kernel<Storage><<<
                    blocks, policy.threads_per_block, 0, stream
                >>>(buffers.features, buffers.combos, supplemental_materialized,
                    rows, candidates, arity);
                return cudaGetLastError();
            }));

        const auto time_metric = [&](uint32_t metric_id, const char* metric_name) {
            check_cuda(cudaMemcpyAsync(
                buffers.metric_ids, &metric_id, sizeof(metric_id), cudaMemcpyHostToDevice, stream),
                "metric id upload");
            check_cuda(cudaStreamSynchronize(stream), "metric id synchronize");
            if ((metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2) && arity == 1) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", true,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->continuous_unary(
                            buffers.features, buffers.target, buffers.target_stats,
                            buffers.feature_stats, buffers.combos, rows, 0, candidates,
                            buffers.metric_ids, 1, buffers.metric_values, policy, stream);
                    }));
            } else if (metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", true,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->continuous(
                            buffers.features, buffers.target, buffers.means, buffers.combos,
                            rows, arity, 0, candidates, 1, buffers.metric_ids, 1,
                            buffers.metric_values, policy, stream);
                    }));
            } else if (metric_id == GAFIME_METRIC_MUTUAL_INFO) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", true,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->mutual_info(
                            buffers.features, buffers.target, buffers.means, buffers.combos,
                            rows, arity, 0, candidates, 1, 0, options.mi_bins, buffers.metric_values,
                            policy, stream);
                    }));
            } else if (metric_id == GAFIME_METRIC_SPEARMAN) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", true,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->spearman(
                            buffers.features, buffers.target, buffers.means, buffers.target_ranks,
                            buffers.combos, rows, arity, 0, candidates, 1, 0,
                            buffers.metric_values, policy, stream);
                    }));
            }
            add_record<Profile>(records, Types::name, order_index, profile_order,
                "ranking_topk", metric_name, "cuda_event_stream", "device_event", true,
                time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                    return set->select_topk(
                        buffers.metric_values, candidates, 1, 0, options.top_k, 1,
                        buffers.selected_indices, buffers.partial_scores, buffers.partial_indices,
                        partial_blocks, policy, stream);
                }));
            check_cuda(cudaStreamSynchronize(stream), "ranking synchronize");
            std::vector<uint32_t> host_selected(options.top_k);
            std::vector<Result> host_scores(candidates);
            add_record<Profile>(records, Types::name, order_index, profile_order,
                "d2h_transfer", metric_name, "host_chrono_stream_sync", "host_synchronized", true,
                time_host_synchronized(options.warmups, options.repeats, [&]() {
                    check_cuda(cudaMemcpyAsync(
                        host_scores.data(), buffers.metric_values,
                        static_cast<size_t>(candidates) * sizeof(Result),
                        cudaMemcpyDeviceToHost, stream), "D2H metric values");
                    check_cuda(cudaMemcpyAsync(
                        host_selected.data(), buffers.selected_indices,
                        static_cast<size_t>(options.top_k) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream), "D2H result");
                    check_cuda(cudaStreamSynchronize(stream), "D2H synchronize");
                }));
            add_record<Profile>(records, Types::name, order_index, profile_order,
                "selected_row_gather", metric_name, "cuda_event_stream", "device_event", true,
                time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                    return set->copy_selected_rows(
                        buffers.metric_values, buffers.selected_indices, options.top_k, 1,
                        buffers.selected_metric_values, policy, stream);
                }));
            std::vector<Result> selected_scores(options.top_k);
            check_cuda(cudaMemcpyAsync(
                host_scores.data(), buffers.metric_values,
                static_cast<size_t>(candidates) * sizeof(Result), cudaMemcpyDeviceToHost, stream),
                "D2H metric values");
            check_cuda(cudaMemcpyAsync(
                selected_scores.data(), buffers.selected_metric_values,
                static_cast<size_t>(options.top_k) * sizeof(Result), cudaMemcpyDeviceToHost, stream),
                "D2H selected values");
            check_cuda(cudaStreamSynchronize(stream), "D2H report values synchronize");
            add_record<Profile>(records, Types::name, order_index, profile_order,
                "report_construction", metric_name, "host_steady_clock", "host_only", true,
                time_host_only(options.warmups, options.repeats, [&]() {
                    std::ostringstream report;
                    report << metric_name << ':' << host_scores.size() << ':' << selected_scores.size();
                    for (const Result value : selected_scores) report << ':' << value;
                    const volatile size_t report_size = report.str().size();
                    if (report_size == 0) fail("empty report construction");
                }));
        };

        add_record<Profile>(records, Types::name, order_index, profile_order,
            "ranking_kernel", "target_ranks", "cuda_event_stream", "device_event", true,
            time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                return set->build_target_ranks(
                    buffers.target, rows, buffers.target_ranks, policy, stream);
            }));
        time_metric(GAFIME_METRIC_PEARSON, "pearson");
        time_metric(GAFIME_METRIC_R2, "r2");
        time_metric(GAFIME_METRIC_MUTUAL_INFO, "mutual_info");
        time_metric(GAFIME_METRIC_SPEARMAN, "spearman");
        check_cuda(cudaFree(supplemental_materialized), "cudaFree(supplemental_materialized)");
        supplemental_materialized = nullptr;
        buffers.release();
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    } catch (...) {
        cudaFree(supplemental_materialized);
        buffers.release();
        cudaStreamDestroy(stream);
        throw;
    }
}

void add_payload_record(
    std::vector<TimingRecord>& records,
    const std::string& profile,
    uint32_t order_index,
    const std::vector<std::string>& profile_order,
    std::string operation,
    std::string metric,
    SampledValues samples,
    std::string note
) {
    records.push_back(TimingRecord{
        profile, order_index, profile_order, std::move(operation), std::move(metric),
        "host_steady_clock", "host_synchronized_payload_api", false,
        "canonical_payload_api", "within_abi_surface_only", std::move(note),
        std::move(samples.samples_us),
        std::move(samples.raw_samples_us), std::move(samples.loop_counts_per_sample),
        samples.loop_count_per_sample,
    });
}

void run_payload_profile(
    const Options& options,
    PayloadApi& api,
    const PayloadRoute& route,
    std::vector<TimingRecord>& records,
    uint32_t order_index,
    const std::vector<std::string>& profile_order
) {
    const std::string profile = route.profile == GAFIME_PRECISION_FP32
        ? "fp32" : route.profile == GAFIME_PRECISION_MIXED ? "mixed" : "fp64";
    PayloadCallState call(options, route);

    // The payload owns its stream and its execute entry point is synchronous at
    // the ABI boundary.  Host steady-clock samples are therefore explicitly
    // bracketed by cudaDeviceSynchronize; CUDA events on this helper's stream
    // would not prove timing of a payload-private stream.
    const auto allocation = time_host_synchronized(
        options.warmups, options.repeats, [&]() {
            GafimeGpuMatrix temporary = nullptr;
            check_payload_status(
                payload_alloc(api, options, route, &temporary), "payload matrix allocation");
            check_payload_status(payload_free(api, temporary), "payload matrix allocation teardown");
        });
    add_payload_record(
        records, profile, order_index, profile_order, "payload_allocation", "", std::move(allocation),
        "each synchronized sample performs the actual payload alloc/free pair; ABI wrapper and device allocation are included");

    GafimeGpuMatrix matrix = nullptr;
    try {
        check_payload_status(payload_alloc(api, options, route, &matrix), "payload matrix allocation");

        const auto upload = time_host_synchronized(
            options.warmups, options.repeats, [&]() {
                check_payload_status(
                    payload_upload(api, options, call, matrix), "payload matrix upload");
            });
        add_payload_record(
            records, profile, order_index, profile_order, "payload_h2d_upload", "", std::move(upload),
            "synchronous payload upload; host timing includes the payload's resident-stat preparation and device completion");

        const auto update = time_host_synchronized(
            options.warmups, options.repeats, [&]() {
                check_payload_status(
                    payload_update_target(api, options, call, matrix), "payload target update");
            });
        add_payload_record(
            records, profile, order_index, profile_order, "payload_update_target", "", std::move(update),
            "synchronous target replacement through the selected ABI surface");

        uint64_t peak_bytes = 0;
        const auto memory_peak = time_host_synchronized(
            options.warmups, options.repeats, [&]() {
                check_payload_status(
                    payload_memory_peak(api, call, matrix, &peak_bytes),
                    "payload execution memory peak");
            });
        add_payload_record(
            records, profile, order_index, profile_order,
            "payload_execution_memory_peak", "", std::move(memory_peak),
            "state-aware payload memory forecast; no arithmetic timing is inferred from this query");

        const std::array<uint32_t, 4> payload_metrics = {
            GAFIME_METRIC_PEARSON,
            GAFIME_METRIC_SPEARMAN,
            GAFIME_METRIC_MUTUAL_INFO,
            GAFIME_METRIC_R2,
        };
        for (const uint32_t metric : payload_metrics) {
            // The protocol carries one metric at a time so every record is an
            // actual payload execution for a named arithmetic metric.  This
            // resets both the chunk mask and primary-rank declaration; a
            // payload rejecting one metric therefore fails with its ABI return
            // code instead of silently disappearing from the A/B lane.
            call.set_metric(metric);
            const auto execute = time_host_synchronized(
                options.warmups, options.repeats, [&]() {
                    check_payload_status(
                        payload_execute(api, call, matrix),
                        (std::string("payload execute ") + payload_metric_name(metric)).c_str());
                });
            add_payload_record(
                records, profile, order_index, profile_order, "payload_execute",
                payload_metric_name(metric), std::move(execute),
                "synchronized host API boundary around the actual payload execute call; includes ABI validation, payload-private launches, device completion, and caller-owned result visibility; this is not pure kernel timing");
        }

        check_payload_status(payload_free(api, matrix), "payload matrix free");
        matrix = nullptr;
    } catch (...) {
        if (matrix != nullptr) static_cast<void>(payload_free(api, matrix));
        throw;
    }
}

void write_identity(
    std::ostream& output,
    const char* label,
    const std::string& path,
    bool comma,
    bool resolve_symlinks = true
) {
    std::error_code path_error;
    const std::string resolved = resolve_symlinks
        ? canonical_path(path)
        : std::filesystem::absolute(path, path_error).lexically_normal().string();
    std::error_code error;
    const bool exists = !resolved.empty() && std::filesystem::is_regular_file(resolved, error);
    const uint64_t size = exists ? std::filesystem::file_size(resolved, error) : 0ull;
    const bool readable = exists && !error;
    output << "    \"" << label << "\": {\"path\": \"" << json_escape(resolved)
           << "\", \"size_bytes\": " << (readable ? static_cast<unsigned long long>(size) : 0ull)
           << ", \"sha256\": \"" << (readable ? sha256_file(resolved) : std::string()) << "\"}"
           << (comma ? "," : "") << "\n";
}

void append_command_snapshot(std::ostream& output, const CommandSnapshot& snapshot) {
    output << "{\"command\":\"" << json_escape(snapshot.command)
           << "\",\"status\":\"" << snapshot.status
           << "\",\"output\":\"" << json_escape(snapshot.output) << "\"";
    if (!snapshot.detail.empty()) {
        output << ",\"detail\":\"" << json_escape(snapshot.detail) << "\"";
    }
    output << '}';
}

void append_cpu_governor_snapshot(std::ostream& output, const CpuGovernorSnapshot& snapshot) {
    output << "{\"status\":\"" << snapshot.status << "\",\"values\":[";
    for (size_t index = 0; index < snapshot.values.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << json_escape(snapshot.values[index]) << '"';
    }
    output << ']';
    if (!snapshot.detail.empty()) {
        output << ",\"detail\":\"" << json_escape(snapshot.detail) << "\"";
    }
    output << '}';
}

void append_clock_power_state(std::ostream& output, const ClockPowerState& state) {
    output << "{\"cpu_governor\":";
    append_cpu_governor_snapshot(output, state.cpu_governor);
    output << ",\"nvidia_smi\":";
    append_command_snapshot(output, state.nvidia_smi);
    output << '}';
}

void write_json(
    const std::string& path,
    const char* binary_path,
    const std::string& source_sha256,
    const std::string& binary_sha256,
    const std::string& source_commit,
    const SourceBinding& product_source_binding,
    const SourceBinding& harness_source_binding,
    const cudaDeviceProp& props,
    const gafime_cuda_v1::CudaKernelLaunchPolicy& policy,
    const Options& options,
    const PayloadResolution& payload_resolution,
    const ClockPowerState& clock_power_before,
    const ClockPowerState& clock_power_after,
    const std::vector<TimingRecord>& records
) {
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    const std::string payload_path = options.payload_path.empty()
        ? env_value("GAFIME_CUDA_V1_LIB") : options.payload_path;
    const std::string wheel_path = options.wheel_path.empty()
        ? env_value("GAFIME_WHEEL_PATH") : options.wheel_path;
    const std::string canonical_path = options.canonical_evidence_path;
    const std::string python_executable = observed_python_executable();
    const bool canonical_exists = !canonical_path.empty() && std::ifstream(canonical_path).good();
    const std::string dataset_identity = cuda_dataset_identity_json(options);
    const ToolIdentity nvcc = identify_tool("nvcc", "--version");
    const ToolIdentity host_cxx = identify_tool("c++", "--version");
    const ToolIdentity linker = identify_tool("ld", "--version");
    std::set<std::vector<std::string>> distinct_orders;
    for (const auto& record : records) distinct_orders.insert(record.profile_order);
    std::ostringstream output;
    output << std::setprecision(12);
    output << "{\n"
           << "  \"schema\": \"gafime.cuda.native_timing.v2\",\n"
           << "  \"status\": \"pass\",\n"
           << "  \"backend\": \"cuda\",\n"
           << "  \"profiles\": " << json_array(options.requested_profiles) << ",\n"
           << "  \"process_isolation\": \"fresh_helper_process_per_variant_trial\",\n"
           << "  \"variant\": "
           << (options.variant.empty()
                ? "null"
                : "\"" + json_escape(options.variant) + "\"") << ",\n"
           << "  \"ab_block\": ";
    if (options.ab_block < 0) {
        output << "null";
    } else {
        output << options.ab_block;
    }
    output << ",\n  \"variant_sequence\": " << json_array(options.variant_sequence) << ",\n"
           << "  \"source_commit\": \"" << json_escape(source_commit) << "\",\n"
           << "  \"source_file\": \"" << json_escape(__FILE__) << "\",\n"
           << "  \"source_sha256\": \"" << source_sha256 << "\",\n"
           << "  \"source_root\": \"" << json_escape(product_source_binding.root) << "\",\n"
           << "  \"source_tree_state\": ";
    append_source_tree_state(output, product_source_binding.tree);
    output << ",\n"
           << "  \"source_blob\": ";
    append_source_binding(output, product_source_binding);
    output << ",\n"
           << "  \"product_source_root\": \""
           << json_escape(product_source_binding.root) << "\",\n"
           << "  \"product_source_commit\": \""
           << json_escape(product_source_binding.commit) << "\",\n"
           << "  \"product_source_tree_state\": ";
    append_source_tree_state(output, product_source_binding.tree);
    output << ",\n"
           << "  \"harness_source_root\": \""
           << json_escape(harness_source_binding.root) << "\",\n"
           << "  \"harness_source_commit\": \""
           << json_escape(harness_source_binding.commit) << "\",\n"
           << "  \"harness_source_tree_state\": ";
    append_source_tree_state(output, harness_source_binding.tree);
    output << ",\n"
           << "  \"harness_source_sha256\": \""
           << json_escape(harness_source_binding.source_sha256) << "\",\n"
           << "  \"harness_source_blob\": ";
    append_source_binding(output, harness_source_binding);
    output << ",\n"
           << "  \"binary_path\": \"" << json_escape(binary_path) << "\",\n"
           << "  \"binary_sha256\": \"" << binary_sha256 << "\",\n"
           << "  \"device\": {\"name\": \"" << json_escape(props.name)
           << "\", \"compute_major\": " << props.major
           << ", \"compute_minor\": " << props.minor
           << ", \"runtime_version\": " << runtime_version
           << ", \"driver_version\": " << driver_version << "},\n"
           << "  \"compiler\": {\"nvcc_major\": "
#if defined(__CUDACC_VER_MAJOR__)
           << __CUDACC_VER_MAJOR__
#else
           << 0
#endif
           << ", \"nvcc_minor\": "
#if defined(__CUDACC_VER_MINOR__)
           << __CUDACC_VER_MINOR__
#else
           << 0
#endif
           << ", \"nvcc_build\": "
#if defined(__CUDACC_VER_BUILD__)
           << __CUDACC_VER_BUILD__
#else
           << 0
#endif
           << ", \"nvcc\": ";
    append_tool_identity(output, nvcc);
    output << ", \"host_cxx\": ";
    append_tool_identity(output, host_cxx);
    output << ", \"linker\": ";
    append_tool_identity(output, linker);
    output << "},\n"
           << "  \"input_policy\": \"" << json_escape(options.input_policy) << "\",\n"
           << "  \"input_identity\": " << dataset_identity << ",\n"
           << "  \"dataset_identity\": " << dataset_identity << ",\n"
           << "  \"launch_threads_per_block\": " << policy.threads_per_block << ",\n"
           << "  \"workload\": {\"name\": \"" << json_escape(options.workload)
           << "\", \"rows\": " << options.rows
           << ", \"features\": " << options.features
           << ", \"candidates\": " << options.candidates
           << ", \"arity\": " << options.arity
           << ", \"mi_bins\": " << options.mi_bins
           << ", \"top_k\": " << options.top_k << "},\n"
           << "  \"rows\": " << options.rows << ",\n"
           << "  \"candidates\": " << options.candidates << ",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"repeats\": " << options.repeats << ",\n"
           << "  \"sample_region_target_us\": " << kSampleRegionTargetUs << ",\n"
           << "  \"sample_region_calibration_target_us\": "
           << kSampleRegionCalibrationTargetUs << ",\n"
           << "  \"bootstrap_resamples\": " << kBootstrapResamples << ",\n"
           << "  \"bootstrap_seed\": " << kBootstrapSeed << ",\n"
           << "  \"order_repetitions\": " << options.order_repetitions << ",\n"
           << "  \"profile_orders\": [\n";
    size_t order_index = 0;
    for (const auto& order : distinct_orders) {
        output << "    " << json_array(order)
               << (order_index + 1 == distinct_orders.size() ? "\n" : ",\n");
        ++order_index;
    }
    output << "  ],\n"
           << "  \"execution_mode\": \"supplemental_internal_kernel\",\n"
           << "  \"payload_execution_mode\": \""
           << (payload_resolution.status == "resolved" ? "canonical_payload_api" : "not_collected")
           << "\",\n"
           << "  \"abi_surface\": \"" << json_escape(payload_resolution.abi_surface)
           << "\",\n"
           << "  \"payload_comparability\": {\"status\": \"within_surface_only\","
           << "\"cross_surface_abi_overhead\": \"not_comparable\","
           << "\"execute_boundary\": \"host steady clock with cudaDeviceSynchronize before and after; payload-private streams are not event-proven\","
           << "\"arithmetic_claim\": \"payload execute/device completion, not pure kernel timing\"},\n"
           << "  \"measurement_categories\": {"
           << "\"canonical_payload_api\": \"host-synchronized ABI wrapper plus payload-private execution; not pure kernel time\","
           << "\"supplemental_internal_kernel\": \"direct helper-owned CUDA event lane; not canonical payload-wrapper overhead\"},\n"
           << "  \"unobservable_phases\": ["
           << "\"payload-private target/feature-stat preparation separated from execute\", "
           << "\"payload-private per-kernel launch decomposition\", "
           << "\"payload-internal D2H separated from synchronous execute\", "
           << "\"ABI validation separated from the selected payload operation\"],\n"
           << "  \"canonical_payload_resolution\": {\"status\": \""
           << json_escape(payload_resolution.status) << "\", \"detail\": \""
           << json_escape(payload_resolution.detail) << "\", \"abi_surface\": \""
           << json_escape(payload_resolution.abi_surface) << "\", \"symbols\": "
           << json_array(payload_resolution.symbols) << "},\n"
           << "  \"canonical_payload_lifecycle\": {\"status\": \""
           << (canonical_exists ? "validated" : "missing")
           << "\", \"schema\": \"gafime.native-decomposition.v1\""
           << (canonical_exists
                ? ", \"path\": \"" + json_escape(canonical_path)
                    + "\", \"sha256\": \"" + sha256_file(canonical_path) + "\""
                : "")
           << "},\n"
           << "  \"decomposition_boundaries\": {\n"
           << "    \"ingest_conversion\": \""
           << (options.input_policy == "common-f64"
                ? "host conversion from deterministic f64 source"
                : "profile-native fp32/fp64 host ownership preparation without cross-dtype conversion")
           << "\",\n"
           << "    \"planning\": \"host combination descriptor generation\",\n"
           << "    \"candidate_materialization\": \"supplemental direct kernel; synchronized event region; not a production payload launch\",\n"
           << "    \"ranking\": \"actual device top-k partial selection and merge\",\n"
           << "    \"selected_row_gather\": \"device gather of selected metric rows\",\n"
           << "    \"d2h\": \"synchronized metric, selected-index, and selected-row copies\"\n"
           << "  },\n"
           << "  \"provenance\": {\n";
    output << "    \"source_root\": ";
    append_source_binding(output, product_source_binding);
    output << ",\n    \"product_source_root\": ";
    append_source_binding(output, product_source_binding);
    output << ",\n    \"harness_source_root\": ";
    append_source_binding(output, harness_source_binding);
    output << ",\n";
    write_identity(output, "benchmark_source", __FILE__, true);
    write_identity(output, "harness_source", __FILE__, true);
    write_identity(output, "benchmark_binary", binary_path, true);
    write_identity(output, "payload", payload_path, true);
    write_identity(output, "wheel", wheel_path, true);
    write_identity(output, "python_executable", python_executable, false, false);
    output << "  },\n"
           << "  \"environment\": " << json_array(observed_environment()) << ",\n"
           << "  \"process_affinity\": " << '[';
    const auto affinity = observed_affinity();
    for (size_t index = 0; index < affinity.size(); ++index) {
        if (index != 0) output << ", ";
        output << affinity[index];
    }
    output << "],\n"
           << "  \"clock_and_power_capture_point\": \"before and after all timed benchmark regions\",\n"
           << "  \"clock_and_power_state\": {\"before\": ";
    append_clock_power_state(output, clock_power_before);
    output << ",\"after\": ";
    append_clock_power_state(output, clock_power_after);
    output << "},\n"
           << "  \"clock\": {\"host\": \"std::chrono::steady_clock\", \"device\": \"cudaEvent elapsed time on the recorded stream\"},\n"
           << "  \"records\": [\n";
    for (size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        const auto ci = bootstrap_median_ci(record.samples_us, stable_seed(record));
        const auto& raw_samples = record.raw_samples_us.empty()
            ? record.samples_us : record.raw_samples_us;
        output << "    {\"profile\": \"" << record.profile
               << "\", \"order_index\": " << record.order_index
               << ", \"profile_order\": " << json_array(record.profile_order)
               << ", \"operation\": \"" << record.operation
               << "\", \"metric\": \"" << record.metric
               << "\", \"clock\": \"" << record.clock
               << "\", \"timing_scope\": \"" << record.timing_scope
               << "\", \"supplemental\": " << (record.supplemental ? "true" : "false")
               << ", \"evidence_lane\": \"" << record.evidence_lane
               << "\", \"comparability\": \"" << record.comparability
               << "\", \"note\": \"" << json_escape(record.note)
               << "\", \"samples_us\": [";
        for (size_t sample = 0; sample < record.samples_us.size(); ++sample) {
            if (sample != 0) output << ", ";
            output << record.samples_us[sample];
        }
        output << "], \"raw_samples_us\": [";
        for (size_t sample = 0; sample < raw_samples.size(); ++sample) {
            if (sample != 0) output << ", ";
            output << raw_samples[sample];
        }
        output << "], \"median_us\": " << median(record.samples_us)
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
               << ", \"loop_counts_per_sample\": [";
        for (size_t sample = 0; sample < record.loop_counts_per_sample.size(); ++sample) {
            if (sample != 0) output << ", ";
            output << record.loop_counts_per_sample[sample];
        }
        output << ']'
               << ", \"sample_region_target_us\": " << kSampleRegionTargetUs
               << ", \"sample_region_min_observed_us\": "
               << *std::min_element(raw_samples.begin(), raw_samples.end())
               << ", \"sample_region_target_met\": "
               << (*std::min_element(raw_samples.begin(), raw_samples.end()) >=
                       kSampleRegionTargetUs ? "true" : "false")
               << ", \"bootstrap_resamples\": " << kBootstrapResamples
               << ", \"bootstrap_seed\": " << stable_seed(record)
               << "}" << (index + 1 == records.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (path.empty()) {
        std::cout << output.str();
    } else {
        std::ofstream file(path);
        if (!file) fail("cannot open JSON output: " + path);
        file << output.str();
    }
}

void write_csv(
    const std::string& path,
    const char* binary_path,
    const std::string& source_sha256,
    const std::string& binary_sha256,
    const ClockPowerState& clock_power_before,
    const ClockPowerState& clock_power_after,
    const std::vector<TimingRecord>& records
) {
    if (path.empty()) return;
    std::ofstream file(path);
    if (!file) fail("cannot open CSV output: " + path);
    file << "# schema=gafime.cuda.native_timing.v2\n"
         << "# source_file=" << __FILE__ << "\n"
         << "# source_sha256=" << source_sha256 << "\n"
         << "# binary_path=" << binary_path << "\n"
         << "# binary_sha256=" << binary_sha256 << "\n"
         << "# clock_and_power_capture_point=before and after all timed benchmark regions\n";
    std::ostringstream clock_before_json;
    std::ostringstream clock_after_json;
    append_clock_power_state(clock_before_json, clock_power_before);
    append_clock_power_state(clock_after_json, clock_power_after);
    file << "# clock_and_power_state_before=" << clock_before_json.str() << "\n"
         << "# clock_and_power_state_after=" << clock_after_json.str() << "\n"
         << "profile,order_index,operation,metric,clock,supplemental,evidence_lane,comparability,note,samples,raw_samples,"
            "median_us,mad_us,p05_us,p95_us,bootstrap_median_ci_low_us,"
            "bootstrap_median_ci_high_us,mean_us,min_us,max_us,loop_count_per_sample,"
            "loop_counts_per_sample\n";
    file << std::setprecision(12);
    for (const auto& record : records) {
        file << csv_escape(record.profile) << ',' << record.order_index << ','
             << csv_escape(record.operation) << ',' << csv_escape(record.metric) << ','
             << csv_escape(record.clock) << ',' << (record.supplemental ? 1 : 0) << ','
             << csv_escape(record.evidence_lane) << ',' << csv_escape(record.comparability)
             << ',' << csv_escape(record.note) << ',';
        for (size_t index = 0; index < record.samples_us.size(); ++index) {
            if (index != 0) file << ';';
            file << record.samples_us[index];
        }
        const auto ci = bootstrap_median_ci(record.samples_us, stable_seed(record));
        const auto& raw_samples = record.raw_samples_us.empty()
            ? record.samples_us : record.raw_samples_us;
        file << ',';
        for (size_t index = 0; index < raw_samples.size(); ++index) {
            if (index != 0) file << ';';
            file << raw_samples[index];
        }
        file << ',' << median(record.samples_us) << ','
             << median_absolute_deviation(record.samples_us) << ','
             << percentile(record.samples_us, 0.05) << ','
             << percentile(record.samples_us, 0.95) << ',' << ci[0] << ',' << ci[1] << ','
             << mean(record.samples_us) << ','
             << *std::min_element(record.samples_us.begin(), record.samples_us.end()) << ','
             << *std::max_element(record.samples_us.begin(), record.samples_us.end()) << ','
             << record.loop_count_per_sample << ',';
        for (size_t index = 0; index < record.loop_counts_per_sample.size(); ++index) {
            if (index != 0) file << ';';
            file << record.loop_counts_per_sample[index];
        }
        file << '\n';
    }
}

void validate_records(const std::vector<TimingRecord>& records, uint32_t repeats) {
    if (records.empty()) fail("no CUDA timing records produced");
    for (const TimingRecord& record : records) {
        if (record.samples_us.size() != repeats ||
            record.raw_samples_us.size() != repeats ||
            record.loop_counts_per_sample.size() != repeats) {
            fail("CUDA timing record does not contain the requested raw sample count: " +
                 record.operation);
        }
        for (size_t index = 0; index < repeats; ++index) {
            if (!std::isfinite(record.samples_us[index]) || record.samples_us[index] <= 0.0 ||
                !std::isfinite(record.raw_samples_us[index]) ||
                record.raw_samples_us[index] < kSampleRegionTargetUs ||
                record.loop_counts_per_sample[index] == 0) {
                fail("CUDA timing record has an invalid calibrated sample: " + record.operation);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
        const auto policy = gafime_cuda_v1::cuda_kernel_launch_policy_for_device(
            static_cast<uint32_t>(props.major), static_cast<uint32_t>(props.maxThreadsPerBlock));
        if (!gafime_cuda_v1::cuda_kernel_launch_policy_supported(policy)) {
            fail("CUDA launch policy is unsupported for the selected device");
        }

        // Resolve both provenance domains before any payload discovery or
        // timed work.  A dirty product/harness checkout must fail quickly,
        // rather than spending minutes calibrating a result that cannot be
        // admitted as evidence.
        const std::string source_path = canonical_path(__FILE__);
        const std::string binary_path = canonical_path(argv[0]);
        const std::string source_sha256 = sha256_file(source_path);
        const std::string binary_sha256 = sha256_file(binary_path);
        const SourceBinding product_source_binding = identify_product_source(options);
        const SourceBinding harness_source_binding = identify_harness_source(options, source_sha256);
        validate_source_provenance(
            product_source_binding, harness_source_binding, source_path, source_sha256);

        // Resolve the payload surface and routes before the clock/power
        // boundary.  The direct-kernel lane remains runnable without a
        // payload, while a supplied payload must expose one complete ABI
        // surface or the helper fails closed.
        PayloadApi payload_api = open_payload(options);
        std::array<PayloadRoute, 3> payload_route_by_profile{};
        if (payload_api.surface != PayloadAbiSurface::None) {
            payload_route_by_profile = payload_routes(payload_api);
        }

        // Capture outside every timed region.  Payload discovery and source
        // identity work happen outside the matching timed regions so they can
        // never be charged to the native records.
        const ClockPowerState clock_power_before = capture_clock_power_state();
        std::vector<TimingRecord> records;
        const auto orders = profile_orders(options.requested_profiles, options.seed);
        if (options.requested_profiles.size() == 3) {
            const std::set<std::vector<std::string>> observed_orders(
                orders.begin(), orders.end());
            if (observed_orders.size() != 6) {
                fail("CUDA native timing must exercise all six canonical profile orders");
            }
        }
        for (uint32_t repeat = 0; repeat < options.order_repetitions; ++repeat) {
            for (size_t order_index = 0; order_index < orders.size(); ++order_index) {
                const auto& order = orders[order_index];
                for (const std::string& profile : order) {
                    switch (profile_id(profile)) {
                    case GAFIME_PRECISION_FP32:
                        run_profile<GAFIME_PRECISION_FP32>(
                            options, policy, records, static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        if (payload_api.surface != PayloadAbiSurface::None) {
                            run_payload_profile(
                                options, payload_api, payload_route_by_profile[0], records,
                                static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        }
                        break;
                    case GAFIME_PRECISION_MIXED:
                        run_profile<GAFIME_PRECISION_MIXED>(
                            options, policy, records, static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        if (payload_api.surface != PayloadAbiSurface::None) {
                            run_payload_profile(
                                options, payload_api, payload_route_by_profile[1], records,
                                static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        }
                        break;
                    case GAFIME_PRECISION_FP64:
                        run_profile<GAFIME_PRECISION_FP64>(
                            options, policy, records, static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        if (payload_api.surface != PayloadAbiSurface::None) {
                            run_payload_profile(
                                options, payload_api, payload_route_by_profile[2], records,
                                static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        }
                        break;
                    default:
                        fail("unsupported profile dispatch");
                    }
                }
            }
        }
        validate_records(records, options.repeats);
        const ClockPowerState clock_power_after = capture_clock_power_state();
        const PayloadResolution& payload_resolution = payload_api.resolution;
        const std::string source_commit = product_source_binding.commit;
        write_json(
            options.json_path, binary_path.c_str(), source_sha256, binary_sha256,
            source_commit, product_source_binding, harness_source_binding,
            props, policy, options, payload_resolution,
            clock_power_before, clock_power_after, records);
        write_csv(
            options.csv_path, binary_path.c_str(), source_sha256, binary_sha256,
            clock_power_before, clock_power_after, records);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_precision_native_timing: " << error.what() << '\n';
        return 1;
    }
}
