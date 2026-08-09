#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
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
    uint32_t order_repetitions = 1;
    uint64_t seed = 20260809;
    std::vector<std::string> requested_profiles{"fp32", "mixed", "fp64"};
    std::string json_path;
    std::string csv_path;
    std::string payload_path;
    std::string wheel_path;
    std::string source_root;
    std::string canonical_evidence_path;
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

SourceBinding identify_source(const Options& options, const std::string& source_sha256) {
    SourceBinding binding;
    binding.root = source_root_path(options);
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
    std::vector<std::string> symbols;
    std::string detail;
};

PayloadResolution resolve_payload(const Options& options) {
    PayloadResolution resolution;
    if (options.payload_path.empty()) return resolution;
    if (std::ifstream(options.payload_path).fail()) {
        resolution.status = "missing";
        resolution.detail = "payload path is not a readable file";
        return resolution;
    }
#if !defined(_WIN32)
    void* handle = dlopen(options.payload_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        resolution.status = "unresolved";
        const char* error = dlerror();
        resolution.detail = error == nullptr ? "dlopen failed" : error;
        return resolution;
    }
    for (const char* symbol : {
        "gafime_gpu_device_info", "gafime_gpu_matrix_alloc_v2", "gafime_gpu_matrix_upload_v2",
        "gafime_gpu_execute_v2", "gafime_gpu_matrix_free_v2"
    }) {
        if (dlsym(handle, symbol) != nullptr) resolution.symbols.emplace_back(symbol);
    }
    dlclose(handle);
    resolution.status = resolution.symbols.size() >= 3 ? "resolved" : "partial";
    if (resolution.status == "partial") {
        resolution.detail = "canonical ABI symbol resolution found " +
            std::to_string(resolution.symbols.size()) + " of 5 required symbols";
    }
#else
    resolution.status = "path_verified_no_loader_api";
#endif
    return resolution;
}

std::vector<std::string> observed_environment() {
    const std::array<const char*, 12> keys = {
        "GAFIME_CUDA_V1_LIB", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "PYTHONPATH",
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
                << " [--canonical-evidence PATH] [--json PATH] [--csv PATH]\n";
            std::exit(0);
        } else {
            fail("unknown option: " + std::string(argument));
        }
    }
    validate_profiles(options.requested_profiles);
    if (options.workload.empty() || options.rows == 0 || options.features == 0 ||
        options.candidates == 0 || options.arity == 0 || options.arity > 5 ||
        options.arity > options.features || options.mi_bins == 0 || options.mi_bins > 96 ||
        options.top_k == 0 || options.top_k > options.candidates ||
        options.order_repetitions == 0 || options.warmups < 10 || options.repeats < 30) {
        fail("workload/features/candidates must be nonzero, arity must be 1..5 and <= features, "
             "mi-bins must be 1..96, top-k must be 1..candidates, and timing requires "
             "warmups >= 10, repeats >= 30");
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
    for (uint32_t index = 0; index < warmups; ++index) {
        operation();
        check_cuda(cudaDeviceSynchronize(), "host timing warmup synchronize");
    }
    auto measure = [&](uint32_t loop_count) {
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
void append_cuda_profile_identity(
    std::ostringstream& output,
    const Options& options,
    bool& first,
    const char* profile_name
) {
    const HostInputs<Storage> inputs = make_inputs<Storage>(
        options.rows, options.features, options.arity, options.candidates);
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
    std::vector<std::vector<std::string>> orders;
    std::vector<size_t> permutation(requested.size());
    std::iota(permutation.begin(), permutation.end(), 0u);
    do {
        std::vector<std::string> order;
        order.reserve(permutation.size());
        for (const size_t index : permutation) order.push_back(requested[index]);
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
           << "{\"algorithm\":\"gafime.cuda.native_timing.inputs.v1\""
           << ",\"input_policy\":\"native\""
           << ",\"policy_detail\":\"profile-native storage from deterministic f64 formula\""
           << ",\"generator\":\"make_inputs.v1\""
           << ",\"source_scalar_dtype\":\"float64\""
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
           << "],\"matrix_dtype\":\"float64\",\"target_dtype\":\"float64\""
           << ",\"profiles\":{";
    bool first = true;
    for (const std::string& profile : options.requested_profiles) {
        if (profile == "fp32") {
            append_cuda_profile_identity<float>(output, options, first, "fp32");
        } else if (profile == "mixed") {
            // Mixed stores the same profile-native float inputs as fp32 but
            // accumulates and returns double results.
            const HostInputs<float> inputs = make_inputs<float>(
                options.rows, options.features, options.arity, options.candidates);
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
    SampledValues samples
) {
    records.push_back({profile, order_index, profile_order, std::move(operation),
                       std::move(metric), std::move(clock), std::move(timing_scope),
                       supplemental, std::move(samples.samples_us),
                       std::move(samples.raw_samples_us),
                       std::move(samples.loop_counts_per_sample),
                       samples.loop_count_per_sample});
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
            const auto converted = make_inputs<Storage>(rows, features, arity, candidates);
            if (converted.features.empty()) fail("input conversion produced no features");
        });
    add_record<Profile>(records, Types::name, order_index, profile_order,
        "ingest_conversion", "none", "host_steady_clock", "host_only", false,
        conversion_samples);
    const auto planning_samples = time_host_only(
        options.warmups, options.repeats, [&]() {
            const auto combinations = make_combinations(features, arity);
            if (combinations.size() < candidates) fail("planning produced too few combinations");
        });
    add_record<Profile>(records, Types::name, order_index, profile_order,
        "planning", "none", "host_steady_clock", "host_only", false,
        planning_samples);
    const auto inputs = make_inputs<Storage>(rows, features, arity, candidates);
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
            "allocation", "none", "host_chrono_cuda_sync", "host_synchronized", false,
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
            "h2d_upload", "none", "host_chrono_stream_sync", "host_synchronized", false,
            time_host_synchronized(options.warmups, options.repeats, [&]() {
                upload_inputs<Profile>(inputs, buffers, rows, features, stream);
                check_cuda(cudaStreamSynchronize(stream), "H2D synchronize");
            }));

        check_status(set->target_stats(
            buffers.target, rows, buffers.target_stats, policy, stream), "target stats");
        check_status(set->feature_stats(
            buffers.features, rows, features, buffers.feature_stats, policy, stream), "feature stats");
        check_cuda(cudaStreamSynchronize(stream), "stats synchronize");

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
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", false,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->continuous_unary(
                            buffers.features, buffers.target, buffers.target_stats,
                            buffers.feature_stats, buffers.combos, rows, 0, candidates,
                            buffers.metric_ids, 1, buffers.metric_values, policy, stream);
                    }));
            } else if (metric_id == GAFIME_METRIC_PEARSON || metric_id == GAFIME_METRIC_R2) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", false,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->continuous(
                            buffers.features, buffers.target, buffers.means, buffers.combos,
                            rows, arity, 0, candidates, 1, buffers.metric_ids, 1,
                            buffers.metric_values, policy, stream);
                    }));
            } else if (metric_id == GAFIME_METRIC_MUTUAL_INFO) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", false,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->mutual_info(
                            buffers.features, buffers.target, buffers.means, buffers.combos,
                            rows, arity, 0, candidates, 1, 0, options.mi_bins, buffers.metric_values,
                            policy, stream);
                    }));
            } else if (metric_id == GAFIME_METRIC_SPEARMAN) {
                add_record<Profile>(records, Types::name, order_index, profile_order,
                    "metric_kernel", metric_name, "cuda_event_stream", "device_event", false,
                    time_cuda_events(stream, options.warmups, options.repeats, [&]() {
                        return set->spearman(
                            buffers.features, buffers.target, buffers.means, buffers.target_ranks,
                            buffers.combos, rows, arity, 0, candidates, 1, 0,
                            buffers.metric_values, policy, stream);
                    }));
            }
            add_record<Profile>(records, Types::name, order_index, profile_order,
                "ranking_topk", metric_name, "cuda_event_stream", "device_event", false,
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
                "d2h_transfer", metric_name, "host_chrono_stream_sync", "host_synchronized", false,
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
                "selected_row_gather", metric_name, "cuda_event_stream", "device_event", false,
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
                "report_construction", metric_name, "host_steady_clock", "host_only", false,
                time_host_only(options.warmups, options.repeats, [&]() {
                    std::ostringstream report;
                    report << metric_name << ':' << host_scores.size() << ':' << selected_scores.size();
                    for (const Result value : selected_scores) report << ':' << value;
                    const volatile size_t report_size = report.str().size();
                    if (report_size == 0) fail("empty report construction");
                }));
        };

        add_record<Profile>(records, Types::name, order_index, profile_order,
            "ranking_kernel", "target_ranks", "cuda_event_stream", "device_event", false,
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
    const SourceBinding& source_binding,
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
           << "  \"source_commit\": \"" << json_escape(source_commit) << "\",\n"
           << "  \"source_file\": \"" << json_escape(__FILE__) << "\",\n"
           << "  \"source_sha256\": \"" << source_sha256 << "\",\n"
           << "  \"source_root\": \"" << json_escape(source_binding.root) << "\",\n"
           << "  \"source_tree_state\": ";
    append_source_tree_state(output, source_binding.tree);
    output << ",\n"
           << "  \"source_blob\": ";
    append_source_binding(output, source_binding);
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
           << "  \"input_policy\": \"native\",\n"
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
           << "  \"canonical_payload_resolution\": {\"status\": \""
           << json_escape(payload_resolution.status) << "\", \"detail\": \""
           << json_escape(payload_resolution.detail) << "\", \"symbols\": "
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
           << "    \"ingest_conversion\": \"host conversion from deterministic f64 source\",\n"
           << "    \"planning\": \"host combination descriptor generation\",\n"
           << "    \"candidate_materialization\": \"supplemental direct kernel; synchronized event region; not a production payload launch\",\n"
           << "    \"ranking\": \"actual device top-k partial selection and merge\",\n"
           << "    \"selected_row_gather\": \"device gather of selected metric rows\",\n"
           << "    \"d2h\": \"synchronized metric, selected-index, and selected-row copies\"\n"
           << "  },\n"
           << "  \"provenance\": {\n";
    output << "    \"source_root\": ";
    append_source_binding(output, source_binding);
    output << ",\n";
    write_identity(output, "benchmark_source", __FILE__, true);
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
               << ", \"samples_us\": [";
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
         << "profile,order_index,operation,metric,clock,supplemental,samples,raw_samples,"
            "median_us,mad_us,p05_us,p95_us,bootstrap_median_ci_low_us,"
            "bootstrap_median_ci_high_us,mean_us,min_us,max_us,loop_count_per_sample,"
            "loop_counts_per_sample\n";
    file << std::setprecision(12);
    for (const auto& record : records) {
        file << record.profile << ',' << record.order_index << ',' << record.operation << ','
             << record.metric << ',' << record.clock << ',' << (record.supplemental ? 1 : 0) << ',';
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

        // Capture outside every timed region.  Payload discovery and source
        // identity work happen after the matching after-snapshot so they can
        // never be charged to the native arithmetic records.
        const ClockPowerState clock_power_before = capture_clock_power_state();
        std::vector<TimingRecord> records;
        const auto orders = profile_orders(options.requested_profiles, options.seed);
        for (uint32_t repeat = 0; repeat < options.order_repetitions; ++repeat) {
            for (size_t order_index = 0; order_index < orders.size(); ++order_index) {
                const auto& order = orders[order_index];
                for (const std::string& profile : order) {
                    switch (profile_id(profile)) {
                    case GAFIME_PRECISION_FP32:
                        run_profile<GAFIME_PRECISION_FP32>(
                            options, policy, records, static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        break;
                    case GAFIME_PRECISION_MIXED:
                        run_profile<GAFIME_PRECISION_MIXED>(
                            options, policy, records, static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        break;
                    case GAFIME_PRECISION_FP64:
                        run_profile<GAFIME_PRECISION_FP64>(
                            options, policy, records, static_cast<uint32_t>(order_index + repeat * orders.size()), order);
                        break;
                    default:
                        fail("unsupported profile dispatch");
                    }
                }
            }
        }
        validate_records(records, options.repeats);
        const ClockPowerState clock_power_after = capture_clock_power_state();
        const std::string source_path = canonical_path(__FILE__);
        const std::string binary_path = canonical_path(argv[0]);
        const std::string source_sha256 = sha256_file(source_path);
        const std::string binary_sha256 = sha256_file(binary_path);
        const PayloadResolution payload_resolution = resolve_payload(options);
        const SourceBinding source_binding = identify_source(options, source_sha256);
        const std::string source_commit = source_binding.commit;
        if (source_commit.size() != 40 ||
            !std::all_of(source_commit.begin(), source_commit.end(), [](unsigned char value) {
                return std::isxdigit(value) != 0;
            })) {
            fail("could not resolve a full source commit for provenance");
        }
        write_json(
            options.json_path, binary_path.c_str(), source_sha256, binary_sha256,
            source_commit, source_binding, props, policy, options, payload_resolution,
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
