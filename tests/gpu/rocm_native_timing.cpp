/*
 * ROCm native timing evidence for the published ABI 1.1 precision routes.
 *
 * This executable deliberately loads the payload named by --payload instead
 * of linking against the build-tree target.  The resulting artifact is a
 * machine-readable decomposition of the canonical host lifecycle and the HIP
 * event time of each metric/ranking route.  It is a benchmark helper, not a
 * second product execution path.
 */

#include <hip/hip_runtime.h>

#include "gafime_gpu_abi.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
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
#include <memory>
#include <numeric>
#include <random>
#include <sched.h>
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
constexpr uint32_t kMaxLoopCount = 1u << 20;
constexpr uint32_t kBootstrapResamples = 2000;
constexpr uint64_t kBootstrapSeed = 20260809ULL;

struct BenchmarkError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string payload;
    std::string output;
    std::string wheel;
    std::string source_root;
    std::string workload = "custom";
    uint32_t device = 0;
    uint64_t rows = 4096;
    uint32_t features = 8;
    uint32_t candidates = 8;
    uint32_t arity = 1;
    uint32_t mi_bins = 64;
    uint32_t top_k = 2;
    uint32_t warmups = 10;
    uint32_t repeats = 30;
    uint64_t order_seed = 0x475346494d45524fULL;
    uint64_t dataset_seed = 0x524f434d31312d31ULL;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw BenchmarkError(
        message +
        "\nusage: gafime_rocm_native_timing --payload PATH --json PATH "
        "[--workload NAME] [--rows N] [--features N] [--candidates N] [--arity 1..5] "
        "[--mi-bins N] [--top-k N] [--warmups N] [--repeats N] "
        "[--order-seed N] [--dataset-seed N] [--device N] [--source-root PATH] [--wheel PATH]");
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
        } else if (argument == "--workload") {
            options.workload = value;
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
    if (options.warmups < 10 || options.repeats < 30) {
        usage_error("warmups must be at least 10 and repeats at least 30");
    }
    if (options.top_k > options.candidates) {
        usage_error("top-k cannot exceed candidates");
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
    if (!source_path.is_absolute()) source_path = std::filesystem::absolute(source_path, error);
    if (error) return {};
    return canonical_path(source_path.parent_path().parent_path().parent_path().string());
}

std::string source_root_path(const Options& options) {
    return canonical_path(options.source_root.empty() ? inferred_source_root() : options.source_root);
}

std::string git_head(const std::string& source_root) {
    if (source_root.empty()) return {};
    return command_output("git -C " + shell_quote(source_root) + " rev-parse HEAD 2>/dev/null");
}

struct SourceTreeState {
    std::string status = "not_supplied";
    std::vector<std::string> entries;
    size_t entry_count = 0;
    std::string detail;
};

SourceTreeState source_tree_state(const std::string& source_root) {
    SourceTreeState state;
    if (source_root.empty()) return state;
    if (command_output("git -C " + shell_quote(source_root) +
                       " rev-parse --is-inside-work-tree 2>/dev/null") != "true") {
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
};

struct FileIdentity;
SourceBinding identify_source(const Options& options, const FileIdentity& benchmark_source);

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

FileIdentity identify_file(const std::string& input) {
    const std::string path = absolute_path(input);
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

SourceBinding identify_source(const Options& options, const FileIdentity& benchmark_source) {
    SourceBinding binding;
    binding.root = source_root_path(options);
    binding.relative_path = source_relative_path(binding.root);
    binding.commit = git_head(binding.root);
    binding.source_sha256 = benchmark_source.sha256;
    binding.current_git_blob = git_blob(binding.root, binding.relative_path, false);
    binding.head_git_blob = git_blob(binding.root, binding.relative_path, true);
    binding.tree = source_tree_state(binding.root);
    return binding;
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
           << ",\"relative_source\":" << json_escape(binding.relative_path)
           << ",\"source_path\":" << json_escape(binding.relative_path)
           << ",\"commit\":" << json_escape(binding.commit)
           << ",\"sha256\":" << json_escape(binding.source_sha256)
           << ",\"source_sha256\":" << json_escape(binding.source_sha256)
           << ",\"git_blob\":" << json_escape(binding.current_git_blob)
           << ",\"current_git_blob\":" << json_escape(binding.current_git_blob)
           << ",\"head_git_blob\":" << json_escape(binding.head_git_blob)
           << ",\"tree_state\":";
    append_source_tree_state(stream, binding.tree);
    stream << '}';
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

struct Api {
    using Routes = int (*)(uint32_t, uint32_t, uint32_t, GafimeNumericRoute*, uint32_t, uint32_t*);
    using Alloc = int (*)(uint32_t, const GafimeNumericMatrixDesc*, GafimeGpuMatrix*);
    using Upload = int (*)(GafimeGpuMatrix, const GafimeNumericRoute*,
                           const GafimeConstBufferView*, const GafimeConstBufferView*,
                           uint64_t, uint32_t);
    using UpdateTarget = int (*)(GafimeGpuMatrix, const GafimeNumericRoute*,
                                 const GafimeConstBufferView*, uint64_t);
    using Execute = int (*)(GafimeGpuMatrix, const GafimeNumericLaunchProtocol*,
                            GafimeNumericResultTable*);
    using Free = int (*)(GafimeGpuMatrix);

    void* handle = nullptr;
    Routes routes = nullptr;
    Alloc alloc = nullptr;
    Upload upload = nullptr;
    UpdateTarget update_target = nullptr;
    Execute execute = nullptr;
    Free free_matrix = nullptr;

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
          routes(other.routes),
          alloc(other.alloc),
          upload(other.upload),
          update_target(other.update_target),
          execute(other.execute),
          free_matrix(other.free_matrix) {}
    Api& operator=(Api&& other) noexcept {
        if (this == &other) return *this;
        if (handle != nullptr) dlclose(handle);
        handle = std::exchange(other.handle, nullptr);
        routes = other.routes;
        alloc = other.alloc;
        upload = other.upload;
        update_target = other.update_target;
        execute = other.execute;
        free_matrix = other.free_matrix;
        return *this;
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

Api open_payload(const std::string& input) {
    Api api;
    const std::string path = absolute_path(input);
    api.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr) {
        const char* error = dlerror();
        throw BenchmarkError("cannot load payload " + path +
                             (error == nullptr ? "" : std::string(": ") + error));
    }
    api.routes = load_symbol<Api::Routes>(api.handle, "gafime_gpu_numeric_routes_v2");
    api.alloc = load_symbol<Api::Alloc>(api.handle, "gafime_gpu_matrix_alloc_v2");
    api.upload = load_symbol<Api::Upload>(api.handle, "gafime_gpu_matrix_upload_v2");
    api.update_target = load_symbol<Api::UpdateTarget>(
        api.handle, "gafime_gpu_matrix_update_target_v2");
    api.execute = load_symbol<Api::Execute>(api.handle, "gafime_gpu_execute_v2");
    api.free_matrix = load_symbol<Api::Free>(api.handle, "gafime_gpu_matrix_free_v2");
    return api;
}

uint64_t dtype_size(uint32_t dtype) {
    if (dtype == GAFIME_DTYPE_F32) return sizeof(float);
    if (dtype == GAFIME_DTYPE_F64) return sizeof(double);
    throw BenchmarkError("unknown route dtype " + std::to_string(dtype));
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
};

Dataset make_dataset(const Options& options) {
    Dataset dataset;
    const uint64_t count = options.rows * static_cast<uint64_t>(options.features);
    dataset.features.resize(static_cast<size_t>(count));
    dataset.target.resize(static_cast<size_t>(options.rows));
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
            target += (0.21 / frequency) * value;
        }
        target += 0.07 * std::sin(row_index * 0.031 + phase * 0.7);
        dataset.target[static_cast<size_t>(row)] = target;
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
    stream << "{\"algorithm\":\"gafime.rocm.native_timing.dataset.v1\""
           << ",\"input_policy\":\"common-f64\""
           << ",\"generator\":\"make_dataset.v1\""
           << ",\"dataset_seed\":" << options.dataset_seed
           << ",\"matrix_sha256\":" << json_escape(sha256_vector(dataset.features))
           << ",\"target_sha256\":" << json_escape(sha256_vector(dataset.target))
           << ",\"feature_names_sha256\":" << json_escape(
               sha256_bytes(feature_names.data(), feature_names.size()))
           << ",\"matrix_shape\":[" << options.rows << ',' << options.features << ']'
           << ",\"target_shape\":[" << options.rows << ']'
           << ",\"matrix_dtype\":\"float64\",\"target_dtype\":\"float64\""
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

    Result(const GafimeNumericRoute& route, uint32_t max_arity, uint64_t capacity) {
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
        reset();
    }

    void reset() {
        table.row_count = 0;
        table.flags = 0;
    }

    double first_value(uint32_t dtype) const {
        if (dtype == GAFIME_DTYPE_F32) return static_cast<double>(values_f32[0]);
        return values_f64[0];
    }
};

struct SampledValues {
    // samples_us is normalized to one operation; raw_samples_us retains the
    // complete calibrated region measured by the selected clock.
    std::vector<double> samples_us;
    std::vector<double> raw_samples_us;
    std::vector<uint32_t> loop_counts_per_sample;
    uint32_t loop_count_per_sample = 1;
};

struct EventSamples {
    SampledValues gpu;
    SampledValues host;
};

template <typename Function>
SampledValues host_samples(uint32_t warmups, uint32_t repeats, Function function) {
    for (uint32_t index = 0; index < warmups; ++index) {
        require_status(function(), "host timing warmup");
    }

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

template <typename Prepare, typename Execute>
EventSamples event_samples(
    uint32_t warmups,
    uint32_t repeats,
    Prepare prepare,
    Execute execute
) {
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
            require_status(prepare(), "event timing preparation");
            require_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(before event)");
            require_hip(hipEventRecord(start, nullptr), "hipEventRecord(start)");
            const auto host_start = Clock::now();
            for (uint32_t loop = 0; loop < loop_count; ++loop) {
                require_status(execute(), "event timing execute");
            }
            const auto host_stop = Clock::now();
            require_hip(hipEventRecord(stop, nullptr), "hipEventRecord(stop)");
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
        for (uint32_t index = 0; index < warmups; ++index) {
            static_cast<void>(one(1));
        }
        uint32_t loop_count = 1;
        auto calibration = one(loop_count);
        while (calibration.first < kSampleRegionCalibrationTargetUs &&
               loop_count < kMaxLoopCount) {
            loop_count = loop_count > kMaxLoopCount / 2 ? kMaxLoopCount : loop_count * 2;
            calibration = one(loop_count);
        }
        EventSamples samples;
        samples.gpu.loop_count_per_sample = loop_count;
        samples.host.loop_count_per_sample = loop_count;
        samples.gpu.samples_us.reserve(repeats);
        samples.gpu.raw_samples_us.reserve(repeats);
        samples.gpu.loop_counts_per_sample.reserve(repeats);
        samples.host.samples_us.reserve(repeats);
        samples.host.raw_samples_us.reserve(repeats);
        samples.host.loop_counts_per_sample.reserve(repeats);
        for (uint32_t index = 0; index < repeats; ++index) {
            uint32_t sample_loop_count = loop_count;
            auto sample = one(sample_loop_count);
            while (sample.first < kSampleRegionTargetUs &&
                   sample_loop_count < kMaxLoopCount) {
                sample_loop_count = sample_loop_count > kMaxLoopCount / 2
                    ? kMaxLoopCount : sample_loop_count * 2;
                sample = one(sample_loop_count);
            }
            loop_count = std::max(loop_count, sample_loop_count);
            samples.gpu.loop_counts_per_sample.push_back(sample_loop_count);
            samples.host.loop_counts_per_sample.push_back(sample_loop_count);
            samples.gpu.raw_samples_us.push_back(sample.first);
            samples.gpu.samples_us.push_back(
                std::max(1.0e-6, sample.first / sample_loop_count));
            samples.host.raw_samples_us.push_back(sample.second);
            samples.host.samples_us.push_back(
                std::max(1.0e-6, sample.second / sample_loop_count));
        }
        samples.gpu.loop_count_per_sample = loop_count;
        samples.host.loop_count_per_sample = loop_count;
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
    std::string operation;
    std::string metric;
    std::vector<double> samples;
    std::vector<double> raw_samples;
    std::vector<uint32_t> loop_counts_per_sample;
    uint32_t loop_count_per_sample = 1;
    std::string clock;
    std::string synchronization;
    std::string note;
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
    const std::string& note
) {
    if (samples.samples_us.empty()) {
        throw BenchmarkError("timing record has no samples: " + operation);
    }
    records.push_back(Record{
        profile, order_index, operation, metric, std::move(samples.samples_us),
        std::move(samples.raw_samples_us), std::move(samples.loop_counts_per_sample),
        samples.loop_count_per_sample,
        clock, synchronization, note,
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
           << ",\"operation\":" << json_escape(record.operation)
           << ",\"metric\":" << json_escape(record.metric)
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
    static constexpr std::array<const char*, 14> keys = {
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "ROCR_MEM_THRASH_LIMIT",
        "HIP_FORCE_DEV_KERNARG", "HIP_LAUNCH_BLOCKING", "GAFIME_ROCM_V1_LIB",
        "LD_LIBRARY_PATH", "PATH", "SHELL", "HOSTNAME", "TERM",
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
    uint64_t rows,
    uint32_t features
) {
    ProfileData converted;
    if (route.storage_dtype == GAFIME_DTYPE_F32) {
        converted.features_f32.resize(source.features.size());
        converted.target_f32.resize(source.target.size());
        for (size_t index = 0; index < source.features.size(); ++index) {
            converted.features_f32[index] = static_cast<float>(source.features[index]);
        }
        for (size_t index = 0; index < source.target.size(); ++index) {
            converted.target_f32[index] = static_cast<float>(source.target[index]);
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

struct MatrixGuard {
    const Api* api = nullptr;
    GafimeGpuMatrix handle = nullptr;
    ~MatrixGuard() {
        if (api != nullptr && handle != nullptr) {
            static_cast<void>(api->free_matrix(handle));
        }
    }
};

void run_profile(
    const Options& options,
    const Api& api,
    const Dataset& source,
    const GafimeNumericRoute& route,
    const std::string& name,
    uint32_t order_index,
    std::vector<Record>& records
) {
    const auto ingest = host_samples(options.warmups, options.repeats, [&]() -> int {
        ProfileData ignored = convert_dataset(source, route, options.rows, options.features);
        if (route.storage_dtype == GAFIME_DTYPE_F32 && ignored.features_f32.empty()) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        if (route.storage_dtype == GAFIME_DTYPE_F64 && ignored.features_f64.empty()) {
            return GAFIME_STATUS_DEVICE_ERROR;
        }
        return GAFIME_STATUS_OK;
    });
    append_record(
        records, name, order_index, "ingest_conversion", "", std::move(ingest),
        "host_steady_clock",
        "steady_clock around common-f64 to canonical route storage conversion",
        "common deterministic f64 source converted to route storage dtype");

    ProfileData converted = convert_dataset(source, route, options.rows, options.features);
    std::vector<uint32_t> descriptors;
    const auto candidate_materialization = host_samples(options.warmups, options.repeats, [&]() -> int {
        materialize_candidates(options.features, options.arity, options.candidates, descriptors);
        return GAFIME_STATUS_OK;
    });
    append_record(
        records, name, order_index, "candidate_materialization", "", std::move(candidate_materialization),
        "host_steady_clock",
        "steady_clock around caller-owned combination descriptor materialization",
        "candidate descriptors are materialized separately from route planning");
    materialize_candidates(options.features, options.arity, options.candidates, descriptors);

    Protocol planning_protocol(
        route, options, descriptors, GAFIME_METRIC_PEARSON, 0,
        0x10000000ULL + static_cast<uint64_t>(order_index) * 16ULL + route.profile);
    const auto planning = host_samples(options.warmups, options.repeats, [&]() -> int {
        planning_protocol.numeric.route = route;
        planning_protocol.numeric.base = &planning_protocol.base;
        planning_protocol.base.rank.top_k = 0;
        planning_protocol.base.rank.primary_metric = GAFIME_METRIC_PEARSON;
        planning_protocol.base.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT]++;
        return GAFIME_STATUS_OK;
    });
    append_record(
        records, name, order_index, "planning", "", std::move(planning),
        "host_steady_clock",
        "steady_clock around canonical ABI1.1 route/base/chunk/result descriptor assembly",
        "planning and descriptor materialization are host-owned; no product call is included");

    const GafimeNumericMatrixDesc desc = matrix_desc(route, options);
    const auto allocation = host_samples(options.warmups, options.repeats, [&]() -> int {
        GafimeGpuMatrix temporary = nullptr;
        const int status = api.alloc(options.device, &desc, &temporary);
        if (status != GAFIME_STATUS_OK) return status;
        return api.free_matrix(temporary);
    });
    append_record(
        records, name, order_index, "allocation", "", std::move(allocation),
        "host_steady_clock",
        "steady_clock around canonical gafime_gpu_matrix_alloc_v2 plus matrix_free_v2",
        "fresh device allocation/free pair per sample");

    MatrixGuard matrix{&api, nullptr};
    require_status(api.alloc(options.device, &desc, &matrix.handle), "matrix_alloc_v2");
    const auto upload = host_samples(options.warmups, options.repeats, [&]() -> int {
        if (route.storage_dtype == GAFIME_DTYPE_F32) {
            const auto features = const_view(route.storage_dtype, converted.features_f32.data(),
                                             converted.features_f32.size());
            const auto target = const_view(route.storage_dtype, converted.target_f32.data(),
                                           converted.target_f32.size());
            return api.upload(matrix.handle, &route, &features, &target, options.rows, options.features);
        }
        const auto features = const_view(route.storage_dtype, converted.features_f64.data(),
                                         converted.features_f64.size());
        const auto target = const_view(route.storage_dtype, converted.target_f64.data(),
                                       converted.target_f64.size());
        return api.upload(matrix.handle, &route, &features, &target, options.rows, options.features);
    });
    append_record(
        records, name, order_index, "h2d_upload", "", std::move(upload),
        "host_steady_clock",
        "steady_clock around canonical synchronous matrix_upload_v2",
        "upload includes route-typed host/device transfer and payload-side resident statistics");

    const auto target_view = [&]() {
        if (route.storage_dtype == GAFIME_DTYPE_F32) {
            return const_view(route.storage_dtype, converted.target_f32.data(), converted.target_f32.size());
        }
        return const_view(route.storage_dtype, converted.target_f64.data(), converted.target_f64.size());
    };
    const GafimeConstBufferView target = target_view();

    auto run_execute = [&](Protocol& protocol, Result& result, auto&& prepare) {
        result.reset();
        return event_samples(
            options.warmups, options.repeats,
            [&]() -> int {
                result.reset();
                return prepare();
            },
            [&]() -> int {
                result.reset();
                return api.execute(matrix.handle, &protocol.numeric, &result.table);
            });
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
        auto result = std::make_unique<Result>(route, options.arity, options.candidates);
        const auto timing = run_execute(protocol, *result, []() -> int {
            return GAFIME_STATUS_OK;
        });
        const std::string metric_name = metric == GAFIME_METRIC_PEARSON
            ? "pearson"
            : metric == GAFIME_METRIC_R2
                ? "r2"
                : metric == GAFIME_METRIC_MUTUAL_INFO ? "mutual_info" : "spearman";
        append_record(
            records, name, order_index, "metric_kernel", metric_name,
            std::move(timing.gpu),
            "hip_event_elapsed_after_synchronized_execute",
            "hipDeviceSynchronize before start; hipEventRecord/hipEventSynchronize around canonical execute_v2",
            "canonical payload metric route; execute_v2 synchronously materializes result buffers");
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
    auto ranking_result = std::make_unique<Result>(route, options.arity, options.candidates);
    const auto ranking_timing = run_execute(ranking_protocol, *ranking_result, [&]() -> int {
        return api.update_target(matrix.handle, &route, &target, options.rows);
    });
    append_record(
        records, name, order_index, "ranking_target_ranks", "spearman",
        std::move(ranking_timing.gpu),
        "hip_event_elapsed_after_synchronized_execute",
        "target update invalidates the cache outside the event; synchronized HIP events bracket the first Spearman execute_v2",
        "the first Spearman execution includes canonical target-rank construction when cache eligibility holds");

    Protocol spearman_protocol(
        route, options, descriptors, GAFIME_METRIC_SPEARMAN, 0,
        0x31000000ULL + static_cast<uint64_t>(order_index) * 32ULL + route.profile);
    auto spearman_result = std::make_unique<Result>(route, options.arity, options.candidates);
    const auto spearman_timing = run_execute(spearman_protocol, *spearman_result, []() -> int {
        return GAFIME_STATUS_OK;
    });
    append_record(
        records, name, order_index, "metric_kernel", "spearman",
        std::move(spearman_timing.gpu),
        "hip_event_elapsed_after_synchronized_execute",
        "hipDeviceSynchronize before start; hipEventRecord/hipEventSynchronize around cached canonical execute_v2",
        "Spearman metric timing follows the explicit target-rank record");

    if (options.top_k != 0) {
        Protocol topk_protocol(
            route, options, descriptors, GAFIME_METRIC_PEARSON, options.top_k,
            0x32000000ULL + static_cast<uint64_t>(order_index) * 32ULL + route.profile);
        auto topk_result = std::make_unique<Result>(route, options.arity, options.top_k);
        const auto topk_timing = run_execute(topk_protocol, *topk_result, []() -> int {
            return GAFIME_STATUS_OK;
        });
        append_record(
            records, name, order_index, "ranking_topk_and_gather", "pearson",
            std::move(topk_timing.gpu),
            "hip_event_elapsed_after_synchronized_execute",
            "hipDeviceSynchronize before start; HIP events bracket canonical execute_v2 top-k path",
            "top-k selection, selected-row gather, and result materialization are exercised by the public payload");
    }

    if (pearson_result_for_report == nullptr ||
        pearson_host_samples.samples_us.size() != options.repeats) {
        throw BenchmarkError("Pearson result was not captured for D2H/report decomposition");
    }
    append_record(
        records, name, order_index, "d2h_transfer", "pearson",
        std::move(pearson_host_samples),
        "host_steady_clock",
        "host steady clock around synchronous canonical execute_v2 return and caller-owned result visibility",
        "ABI1.1 execute_v2 bundles device synchronization and result readback; this boundary is explicitly documented");

    std::vector<double> report_values;
    report_values.reserve(options.candidates);
    if (route.result_dtype == GAFIME_DTYPE_F32) {
        report_values.assign(pearson_result->values_f32.begin(), pearson_result->values_f32.end());
    } else {
        report_values = pearson_result->values_f64;
    }
    const auto report = host_samples(options.warmups, options.repeats, [&]() -> int {
        std::ostringstream summary;
        summary << std::setprecision(17) << "rows=" << options.rows
                << ";features=" << options.features << ";candidates=" << options.candidates;
        for (const double value : report_values) summary << ';' << value;
        if (summary.str().empty()) return GAFIME_STATUS_DEVICE_ERROR;
        return GAFIME_STATUS_OK;
    });
    append_record(
        records, name, order_index, "report_construction", "", std::move(report),
        "host_steady_clock",
        "steady_clock around host-only result summary construction",
        "report construction is not represented as device-kernel time");
}

void append_u32_array(std::ostringstream& stream, const std::vector<uint32_t>& values) {
    stream << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) stream << ',';
        stream << values[index];
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

void write_json(
    const Options& options,
    const std::string& source_commit,
    const SourceBinding& source_binding,
    const std::string& dataset_identity,
    const FileIdentity& benchmark_source,
    const FileIdentity& benchmark_binary,
    const FileIdentity& payload,
    const FileIdentity* wheel,
    const std::vector<std::array<uint32_t, 3>>& orders,
    const std::vector<Record>& records
) {
    const ToolIdentity hipcc = identify_tool("hipcc", "--version");
    const ToolIdentity clangxx = identify_tool("clang++", "--version");
    const ToolIdentity linker = identify_tool("ld", "--version");
    std::ostringstream stream;
    stream << "{\n"
           << "  \"schema\":\"gafime.rocm.native_timing.v2\",\n"
           << "  \"status\":\"pass\",\n"
           << "  \"backend\":\"rocm\",\n"
           << "  \"execution_mode\":\"canonical_payload\",\n"
           << "  \"source_commit\":" << json_escape(source_commit) << ",\n"
           << "  \"source_root\":" << json_escape(source_binding.root) << ",\n"
           << "  \"source_tree_state\":";
    append_source_tree_state(stream, source_binding.tree);
    stream << ",\n  \"source_blob\":";
    append_source_binding(stream, source_binding);
    stream << ",\n"
           << "  \"benchmark\":\"canonical ABI1.1 ROCm payload lifecycle and HIP-event timing\",\n"
           << "  \"profiles\":[\"fp32\",\"mixed\",\"fp64\"],\n"
           << "  \"profile_orders\":";
    append_orders_json(stream, orders);
    stream << ",\n"
           << "  \"order_seed\":" << options.order_seed << ",\n"
           << "  \"dataset_seed\":" << options.dataset_seed << ",\n"
           << "  \"input_policy\":\"common-f64\",\n"
           << "  \"input_identity\":" << dataset_identity << ",\n"
           << "  \"dataset_identity\":" << dataset_identity << ",\n"
           << "  \"workload\":" << json_escape(options.workload) << ",\n"
           << "  \"rows\":" << options.rows << ",\n"
           << "  \"features\":" << options.features << ",\n"
           << "  \"candidates\":" << options.candidates << ",\n"
           << "  \"arity\":" << options.arity << ",\n"
           << "  \"mi_bins\":" << options.mi_bins << ",\n"
           << "  \"top_k\":" << options.top_k << ",\n"
           << "  \"warmups\":" << options.warmups << ",\n"
           << "  \"repeats\":" << options.repeats << ",\n"
           << "  \"sample_region_target_us\":" << kSampleRegionTargetUs << ",\n"
           << "  \"sample_region_calibration_target_us\":"
           << kSampleRegionCalibrationTargetUs << ",\n"
           << "  \"bootstrap_resamples\":" << kBootstrapResamples << ",\n"
           << "  \"bootstrap_seed\":" << kBootstrapSeed << ",\n"
           << "  \"gpu_timing_supported\":true,\n"
           << "  \"timing_clock\":\"hipEventElapsedTime\",\n"
           << "  \"decomposition_boundaries\":{\n"
           << "    \"ingest_conversion\":\"common f64 host dataset converted to route storage dtype\",\n"
           << "    \"candidate_materialization\":\"caller-owned combination descriptors materialized before ABI planning\",\n"
           << "    \"planning\":\"caller-owned ABI1.1 route/base/chunk/result descriptor assembly\",\n"
           << "    \"allocation\":\"gafime_gpu_matrix_alloc_v2 and matrix_free_v2\",\n"
           << "    \"h2d_upload\":\"synchronous gafime_gpu_matrix_upload_v2 including resident-stat preparation\",\n"
           << "    \"metric_kernel\":\"canonical gafime_gpu_execute_v2 metric route\",\n"
           << "    \"ranking_target_ranks\":\"synchronized first canonical Spearman execute after target update invalidates cache\",\n"
           << "    \"ranking_topk_and_gather\":\"canonical top-k selection and selected-row gather inside execute_v2\",\n"
           << "    \"d2h_transfer\":\"execute_v2 synchronous return exposes caller-owned result buffers after readback\",\n"
           << "    \"report_construction\":\"host-only summary construction after result visibility\"\n"
           << "  },\n"
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
    stream << ",\n  \"provenance\":{\n"
           << "    \"source_root\":";
    append_source_binding(stream, source_binding);
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
    stream << "\n  },\n"
           << "  \"self_checks\":{\n"
           << "    \"canonical_routes\":true,\n"
           << "    \"all_profiles_exercised\":true,\n"
           << "    \"all_six_profile_orders\":true,\n"
           << "    \"hip_events_synchronized\":true,\n"
           << "    \"raw_sample_counts_valid\":true,\n"
           << "    \"finite_results_observed\":true\n"
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
            throw BenchmarkError("record sampled region stayed below 5 ms after loop scaling: " +
                                 record.operation);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::string payload_path = absolute_path(options.payload);
        Api api = open_payload(payload_path);
        require_hip(hipSetDevice(static_cast<int>(options.device)), "hipSetDevice");

        uint32_t route_count = 0;
        require_status(
            api.routes(options.device, kAbi11, sizeof(GafimeNumericRoute), nullptr, 0, &route_count),
            "numeric_routes_v2 count query");
        if (route_count != 3) {
            throw BenchmarkError("canonical ROCm payload advertised " + std::to_string(route_count) +
                                 " routes, expected fp32/mixed/fp64");
        }
        std::vector<GafimeNumericRoute> routes(route_count);
        require_status(
            api.routes(options.device, kAbi11, sizeof(GafimeNumericRoute),
                       routes.data(), route_count, &route_count),
            "numeric_routes_v2 enumeration");
        std::array<GafimeNumericRoute, 3> by_profile{};
        std::array<bool, 3> seen{};
        for (const auto& route : routes) {
            for (size_t index = 0; index < kProfileIds.size(); ++index) {
                if (route.profile == kProfileIds[index]) {
                    if (seen[index]) throw BenchmarkError("duplicate canonical route profile");
                    by_profile[index] = route;
                    seen[index] = true;
                }
            }
        }
        if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
            throw BenchmarkError("payload did not advertise all canonical precision profiles");
        }

        const Dataset dataset = make_dataset(options);
        std::vector<std::array<uint32_t, 3>> orders;
        std::array<uint32_t, 3> order = kProfileIds;
        do {
            orders.push_back(order);
        } while (std::next_permutation(order.begin(), order.end()));
        std::mt19937_64 order_generator(options.order_seed);
        std::shuffle(orders.begin(), orders.end(), order_generator);
        if (orders.size() != 6) throw BenchmarkError("six profile permutations were not generated");

        std::vector<Record> records;
        records.reserve(6 * 3 * 16);
        for (size_t order_index = 0; order_index < orders.size(); ++order_index) {
            for (const uint32_t profile : orders[order_index]) {
                const size_t profile_index = static_cast<size_t>(
                    std::find(kProfileIds.begin(), kProfileIds.end(), profile) - kProfileIds.begin());
                run_profile(
                    options, api, dataset, by_profile[profile_index], profile_name(profile),
                    static_cast<uint32_t>(order_index), records);
            }
        }
        verify_result_finiteness(records);

        const FileIdentity benchmark_source = identify_file(__FILE__);
        const std::string executable = "/proc/self/exe";
        const FileIdentity benchmark_binary = identify_file(executable);
        const FileIdentity payload = identify_file(payload_path);
        FileIdentity wheel;
        const FileIdentity* wheel_pointer = nullptr;
        if (!options.wheel.empty()) {
            wheel = identify_file(options.wheel);
            wheel_pointer = &wheel;
        }
        const SourceBinding source_binding = identify_source(options, benchmark_source);
        const std::string source_commit = source_binding.commit;
        if (source_commit.size() != 40 ||
            !std::all_of(source_commit.begin(), source_commit.end(), [](unsigned char value) {
                return std::isxdigit(value) != 0;
            })) {
            throw BenchmarkError("could not resolve a full source commit for provenance");
        }
        const std::string dataset_identity = rocm_dataset_identity_json(options, dataset);
        write_json(
            options, source_commit, source_binding, dataset_identity, benchmark_source,
            benchmark_binary, payload, wheel_pointer, orders, records);
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
