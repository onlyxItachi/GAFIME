#pragma once

// Strict, backend-independent consumer for native-loop-plan.v1.
//
// The calibration/plan generator is Python, but the CUDA and ROCm helpers are
// intentionally self-contained native consumers.  This parser keeps their
// acceptance boundary identical: JSON objects have no duplicate or unknown
// members, the document is canonical JSON, and all identity bindings are
// checked before a timing loop can be selected.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gafime_native_loop_plan {

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct JsonValue {
    enum class Kind { Null, Boolean, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    std::string number;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    static JsonValue null() { return {}; }
    static JsonValue boolean_value(bool value) {
        JsonValue result;
        result.kind = Kind::Boolean;
        result.boolean = value;
        return result;
    }
    static JsonValue number_value(std::string value) {
        JsonValue result;
        result.kind = Kind::Number;
        result.number = std::move(value);
        return result;
    }
    static JsonValue string_value(std::string value) {
        JsonValue result;
        result.kind = Kind::String;
        result.string = std::move(value);
        return result;
    }
    static JsonValue array_value(std::vector<JsonValue> value) {
        JsonValue result;
        result.kind = Kind::Array;
        result.array = std::move(value);
        return result;
    }
    static JsonValue object_value(std::map<std::string, JsonValue> value) {
        JsonValue result;
        result.kind = Kind::Object;
        result.object = std::move(value);
        return result;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    JsonValue parse_document() {
        skip_space();
        JsonValue value = parse_value();
        skip_space();
        if (position_ != input_.size()) {
            fail("trailing data after native loop plan JSON document");
        }
        return value;
    }

private:
    [[noreturn]] void fail(std::string message) const {
        throw ParseError(
            std::move(message) + " at byte " + std::to_string(position_));
    }

    void skip_space() {
        while (position_ < input_.size()) {
            const unsigned char value = static_cast<unsigned char>(input_[position_]);
            if (value != ' ' && value != '\n' && value != '\r' && value != '\t') return;
            ++position_;
        }
    }

    char take() {
        if (position_ >= input_.size()) fail("unexpected end of native loop plan JSON");
        return input_[position_++];
    }

    void expect(char expected) {
        if (take() != expected) fail(std::string("expected '") + expected + "'");
    }

    JsonValue parse_value() {
        skip_space();
        if (position_ >= input_.size()) fail("missing JSON value");
        switch (input_[position_]) {
        case 'n':
            parse_literal("null");
            return JsonValue::null();
        case 't':
            parse_literal("true");
            return JsonValue::boolean_value(true);
        case 'f':
            parse_literal("false");
            return JsonValue::boolean_value(false);
        case '"':
            return JsonValue::string_value(parse_string());
        case '[':
            return parse_array();
        case '{':
            return parse_object();
        default:
            if (input_[position_] == '-' ||
                std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                return JsonValue::number_value(parse_number());
            }
            fail("invalid JSON value");
        }
    }

    void parse_literal(std::string_view literal) {
        if (input_.substr(position_, literal.size()) != literal) {
            fail("invalid JSON literal");
        }
        position_ += literal.size();
    }

    std::string parse_number() {
        const size_t start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) fail("incomplete JSON number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() &&
                std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                fail("JSON numbers may not contain a leading zero");
            }
        } else {
            if (std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
                fail("invalid JSON number");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const size_t fraction_start = position_;
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
            if (fraction_start == position_) fail("JSON number has an empty fraction");
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            const size_t exponent_start = position_;
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
            if (exponent_start == position_) fail("JSON number has an empty exponent");
        }
        return std::string(input_.substr(start, position_ - start));
    }

    static uint32_t hex_digit(char value) {
        if (value >= '0' && value <= '9') return static_cast<uint32_t>(value - '0');
        if (value >= 'a' && value <= 'f') return static_cast<uint32_t>(value - 'a' + 10);
        if (value >= 'A' && value <= 'F') return static_cast<uint32_t>(value - 'A' + 10);
        throw ParseError("invalid JSON unicode escape");
    }

    uint32_t unicode_escape() {
        if (position_ + 4 > input_.size()) fail("truncated JSON unicode escape");
        uint32_t codepoint = 0;
        for (uint32_t index = 0; index < 4; ++index) {
            codepoint = (codepoint << 4u) | hex_digit(input_[position_++]);
        }
        return codepoint;
    }

    static void append_utf8(std::string& output, uint32_t codepoint) {
        if (codepoint <= 0x7fu) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffu) {
            output.push_back(static_cast<char>(0xc0u | (codepoint >> 6u)));
            output.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
        } else if (codepoint <= 0xffffu) {
            output.push_back(static_cast<char>(0xe0u | (codepoint >> 12u)));
            output.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
            output.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
        } else {
            output.push_back(static_cast<char>(0xf0u | (codepoint >> 18u)));
            output.push_back(static_cast<char>(0x80u | ((codepoint >> 12u) & 0x3fu)));
            output.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
            output.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char value = static_cast<unsigned char>(input_[position_++]);
            if (value == '"') return result;
            if (value < 0x20u) fail("unescaped control character in JSON string");
            if (value != '\\') {
                result.push_back(static_cast<char>(value));
                continue;
            }
            if (position_ >= input_.size()) fail("truncated JSON string escape");
            const char escaped = input_[position_++];
            switch (escaped) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': {
                uint32_t codepoint = unicode_escape();
                if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
                    if (position_ + 6 > input_.size() || input_[position_] != '\\' ||
                        input_[position_ + 1] != 'u') {
                        fail("high surrogate is missing its low surrogate");
                    }
                    position_ += 2;
                    const uint32_t low = unicode_escape();
                    if (low < 0xdc00u || low > 0xdfffu) {
                        fail("invalid JSON surrogate pair");
                    }
                    codepoint = 0x10000u + ((codepoint - 0xd800u) << 10u) +
                        (low - 0xdc00u);
                } else if (codepoint >= 0xdc00u && codepoint <= 0xdfffu) {
                    fail("unexpected low surrogate in JSON string");
                }
                append_utf8(result, codepoint);
                break;
            }
            default: fail("unsupported JSON string escape");
            }
        }
        fail("unterminated JSON string");
    }

    JsonValue parse_array() {
        expect('[');
        std::vector<JsonValue> values;
        skip_space();
        if (position_ < input_.size() && input_[position_] == ']') {
            ++position_;
            return JsonValue::array_value(std::move(values));
        }
        while (true) {
            values.push_back(parse_value());
            skip_space();
            const char separator = take();
            if (separator == ']') break;
            if (separator != ',') fail("expected ',' or ']' in JSON array");
            skip_space();
            if (position_ < input_.size() && input_[position_] == ']') {
                fail("trailing comma in JSON array");
            }
        }
        return JsonValue::array_value(std::move(values));
    }

    JsonValue parse_object() {
        expect('{');
        std::map<std::string, JsonValue> values;
        skip_space();
        if (position_ < input_.size() && input_[position_] == '}') {
            ++position_;
            return JsonValue::object_value(std::move(values));
        }
        while (true) {
            skip_space();
            if (position_ >= input_.size() || input_[position_] != '"') {
                fail("JSON object key must be a string");
            }
            const std::string key = parse_string();
            skip_space();
            expect(':');
            JsonValue value = parse_value();
            if (!values.emplace(key, std::move(value)).second) {
                fail("duplicate JSON object key: " + key);
            }
            skip_space();
            const char separator = take();
            if (separator == '}') break;
            if (separator != ',') fail("expected ',' or '}' in JSON object");
            skip_space();
            if (position_ < input_.size() && input_[position_] == '}') {
                fail("trailing comma in JSON object");
            }
        }
        return JsonValue::object_value(std::move(values));
    }

    std::string_view input_;
    size_t position_ = 0;
};

inline void append_json_string(std::string& output, std::string_view value) {
    static constexpr char hex[] = "0123456789abcdef";
    output.push_back('"');
    for (size_t index = 0; index < value.size();) {
        const unsigned char byte = static_cast<unsigned char>(value[index++]);
        switch (byte) {
        case '"': output += "\\\""; continue;
        case '\\': output += "\\\\"; continue;
        case '\b': output += "\\b"; continue;
        case '\f': output += "\\f"; continue;
        case '\n': output += "\\n"; continue;
        case '\r': output += "\\r"; continue;
        case '\t': output += "\\t"; continue;
        default: break;
        }
        if (byte < 0x20u) {
            output += "\\u00";
            output.push_back(hex[byte >> 4u]);
            output.push_back(hex[byte & 0x0fu]);
        } else if (byte < 0x80u) {
            output.push_back(static_cast<char>(byte));
        } else {
            // Decode UTF-8 so the serializer matches Python's ensure_ascii=True
            // canonical JSON rather than preserving a noncanonical raw byte.
            uint32_t codepoint = 0;
            uint32_t continuation = 0;
            if ((byte & 0xe0u) == 0xc0u) {
                codepoint = byte & 0x1fu;
                continuation = 1;
            } else if ((byte & 0xf0u) == 0xe0u) {
                codepoint = byte & 0x0fu;
                continuation = 2;
            } else if ((byte & 0xf8u) == 0xf0u) {
                codepoint = byte & 0x07u;
                continuation = 3;
            } else {
                throw ParseError("invalid UTF-8 in JSON string");
            }
            if (index + continuation > value.size()) {
                throw ParseError("truncated UTF-8 in JSON string");
            }
            for (uint32_t item = 0; item < continuation; ++item) {
                const unsigned char next = static_cast<unsigned char>(value[index++]);
                if ((next & 0xc0u) != 0x80u) throw ParseError("invalid UTF-8 continuation");
                codepoint = (codepoint << 6u) | (next & 0x3fu);
            }
            if ((continuation == 1 && codepoint < 0x80u) ||
                (continuation == 2 && codepoint < 0x800u) ||
                (continuation == 3 && codepoint < 0x10000u) ||
                codepoint > 0x10ffffu || (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
                throw ParseError("invalid UTF-8 code point");
            }
            const auto append_u16 = [&](uint32_t unit) {
                output += "\\u";
                output.push_back(hex[(unit >> 12u) & 0xfu]);
                output.push_back(hex[(unit >> 8u) & 0xfu]);
                output.push_back(hex[(unit >> 4u) & 0xfu]);
                output.push_back(hex[unit & 0xfu]);
            };
            if (codepoint <= 0xffffu) {
                append_u16(codepoint);
            } else {
                codepoint -= 0x10000u;
                append_u16(0xd800u + (codepoint >> 10u));
                append_u16(0xdc00u + (codepoint & 0x3ffu));
            }
        }
    }
    output.push_back('"');
}

inline void append_canonical_json(std::string& output, const JsonValue& value) {
    switch (value.kind) {
    case JsonValue::Kind::Null:
        output += "null";
        break;
    case JsonValue::Kind::Boolean:
        output += value.boolean ? "true" : "false";
        break;
    case JsonValue::Kind::Number:
        output += value.number;
        break;
    case JsonValue::Kind::String:
        append_json_string(output, value.string);
        break;
    case JsonValue::Kind::Array:
        output.push_back('[');
        for (size_t index = 0; index < value.array.size(); ++index) {
            if (index != 0) output.push_back(',');
            append_canonical_json(output, value.array[index]);
        }
        output.push_back(']');
        break;
    case JsonValue::Kind::Object:
        output.push_back('{');
        size_t index = 0;
        for (const auto& [key, child] : value.object) {
            if (index++ != 0) output.push_back(',');
            append_json_string(output, key);
            output.push_back(':');
            append_canonical_json(output, child);
        }
        output.push_back('}');
        break;
    }
}

inline std::string canonical_json(const JsonValue& value) {
    std::string output;
    append_canonical_json(output, value);
    output.push_back('\n');
    return output;
}

inline const JsonValue& member(
    const JsonValue& object,
    std::string_view key,
    std::string_view label
) {
    if (object.kind != JsonValue::Kind::Object) {
        throw ParseError(std::string(label) + " must be an object");
    }
    const auto found = object.object.find(std::string(key));
    if (found == object.object.end()) {
        throw ParseError(std::string(label) + " is missing " + std::string(key));
    }
    return found->second;
}

inline void exact_keys(
    const JsonValue& object,
    std::initializer_list<std::string_view> allowed,
    std::string_view label
) {
    if (object.kind != JsonValue::Kind::Object) {
        throw ParseError(std::string(label) + " must be an object");
    }
    std::set<std::string> expected;
    for (const std::string_view key : allowed) expected.emplace(key);
    if (object.object.size() != expected.size()) {
        throw ParseError(std::string(label) + " has missing or unknown members");
    }
    for (const auto& [key, value] : object.object) {
        static_cast<void>(value);
        if (!expected.contains(key)) {
            throw ParseError(std::string(label) + " has unknown member: " + key);
        }
    }
}

inline const std::string& string_member(
    const JsonValue& object,
    std::string_view key,
    std::string_view label
) {
    const JsonValue& value = member(object, key, label);
    if (value.kind != JsonValue::Kind::String) {
        throw ParseError(std::string(label) + "." + std::string(key) + " must be a string");
    }
    return value.string;
}

inline uint64_t uint_member(
    const JsonValue& object,
    std::string_view key,
    std::string_view label,
    bool positive = false
) {
    const JsonValue& value = member(object, key, label);
    if (value.kind != JsonValue::Kind::Number || value.number.empty() ||
        value.number.front() == '-' || value.number.find_first_not_of("0123456789") !=
            std::string::npos) {
        throw ParseError(std::string(label) + "." + std::string(key) + " must be an integer");
    }
    uint64_t result = 0;
    try {
        size_t end = 0;
        result = std::stoull(value.number, &end, 10);
        if (end != value.number.size()) throw std::out_of_range("integer");
    } catch (...) {
        throw ParseError(std::string(label) + "." + std::string(key) + " is outside uint64 range");
    }
    if (positive && result == 0) {
        throw ParseError(std::string(label) + "." + std::string(key) + " must be positive");
    }
    return result;
}

inline std::string canonical_member(const JsonValue& object, std::string_view key, std::string_view label) {
    return canonical_json(member(object, key, label));
}

struct Binding {
    std::string path;
    std::string relative_path;
    std::string sha256;
    std::string variant;
    std::string source_commit;
    std::string product_source_commit;
    std::string harness_source_commit;
    std::string backend;
    std::string workload_json;
    std::string input_policy;
    std::string input_identity_json;
    std::string device_json;
};

struct Plan {
    std::map<std::string, uint32_t> entries;
    std::set<std::string> lookups;
    std::string path;
    std::string semantic_sha256;
    std::string file_sha256;
    std::string backend;
    std::string workload;
    uint64_t rows = 0;
    uint64_t features = 0;
    uint64_t candidates = 0;
    uint64_t arity = 0;
    uint64_t mi_bins = 0;
    uint64_t top_k = 0;
    std::string input_policy;
    std::string evidence_lane;
    std::string artifact_kind;
    std::string device_json;
    std::string scope_id;
    std::vector<Binding> bindings;
};

struct ExpectedScope {
    std::string backend;
    std::string workload;
    uint64_t rows = 0;
    uint64_t features = 0;
    uint64_t candidates = 0;
    uint64_t arity = 0;
    uint64_t mi_bins = 0;
    uint64_t top_k = 0;
    std::string input_policy;
    std::string evidence_lane;
    std::string artifact_kind;
    std::string device_json;
    std::string scope_id;
};

inline bool hex_string(std::string_view value, size_t length) {
    if (value.size() != length) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return std::isxdigit(byte) != 0;
    });
}

inline bool safe_relative_path(std::string_view value) {
    const std::filesystem::path path{std::string(value)};
    if (path.empty() || path.is_absolute()) return false;
    for (const auto& component : path) {
        if (component == "..") return false;
    }
    return true;
}

inline void validate_identity_object(const JsonValue& value, std::string_view label) {
    if (value.kind != JsonValue::Kind::Object) {
        throw ParseError(std::string(label) + " must be an object");
    }
    const bool has_size = value.object.contains("size_bytes");
    exact_keys(value, has_size ? std::initializer_list<std::string_view>{"path", "sha256", "size_bytes"}
                               : std::initializer_list<std::string_view>{"path", "sha256"}, label);
    if (string_member(value, "path", label).empty() ||
        !hex_string(string_member(value, "sha256", label), 64)) {
        throw ParseError(std::string(label) + " has an invalid file identity");
    }
    if (has_size) static_cast<void>(uint_member(value, "size_bytes", label));
}

inline Binding parse_binding(const JsonValue& value, std::string_view label) {
    if (value.kind != JsonValue::Kind::Object) {
        throw ParseError(std::string(label) + " must be an object");
    }
    const std::set<std::string> allowed = {
        "backend", "command_line", "device", "harness_source_commit", "input_identity",
        "input_policy", "path", "product_source_commit", "provenance", "relative_path", "sha256",
        "source_blob", "harness_source_blob", "source_commit", "source_root",
        "product_source_root", "harness_source_root", "source_tree_state",
        "product_source_tree_state", "harness_source_tree_state", "git", "git_identity",
        "variant", "workload",
    };
    for (const auto& [key, child] : value.object) {
        static_cast<void>(child);
        if (!allowed.contains(key)) throw ParseError(std::string(label) + " has unknown member: " + key);
    }
    for (const std::string_view key : {
             std::string_view("backend"), std::string_view("command_line"),
             std::string_view("device"), std::string_view("input_identity"),
             std::string_view("input_policy"), std::string_view("path"),
             std::string_view("provenance"), std::string_view("sha256"),
             std::string_view("source_commit"),
             std::string_view("product_source_commit"),
             std::string_view("harness_source_commit"),
             std::string_view("variant"),
             std::string_view("workload")}) {
        static_cast<void>(member(value, key, label));
    }
    Binding binding;
    binding.path = string_member(value, "path", label);
    binding.relative_path = string_member(value, "relative_path", label);
    binding.sha256 = string_member(value, "sha256", label);
    if (binding.path.empty() || !safe_relative_path(binding.relative_path) ||
        !hex_string(binding.sha256, 64)) {
        throw ParseError(std::string(label) + " has an invalid calibration identity");
    }
    binding.variant = string_member(value, "variant", label);
    binding.source_commit = string_member(value, "source_commit", label);
    binding.product_source_commit = string_member(value, "product_source_commit", label);
    binding.harness_source_commit = string_member(value, "harness_source_commit", label);
    if (!hex_string(binding.source_commit, 40) ||
        !hex_string(binding.product_source_commit, 40) ||
        !hex_string(binding.harness_source_commit, 40) ||
        binding.source_commit != binding.product_source_commit) {
        throw ParseError(
            std::string(label) +
            " must bind matching full source/product commits and a full harness commit");
    }
    binding.backend = string_member(value, "backend", label);
    binding.workload_json = canonical_member(value, "workload", label);
    binding.input_policy = string_member(value, "input_policy", label);
    const JsonValue& input_identity = member(value, "input_identity", label);
    if (input_identity.kind != JsonValue::Kind::Object || input_identity.object.empty()) {
        throw ParseError(std::string(label) + ".input_identity must be a non-empty object");
    }
    binding.input_identity_json = canonical_json(input_identity);
    binding.device_json = canonical_member(value, "device", label);
    const JsonValue& command_line = member(value, "command_line", label);
    if (command_line.kind != JsonValue::Kind::Array || command_line.array.empty()) {
        throw ParseError(std::string(label) + ".command_line must be a non-empty array");
    }
    for (const JsonValue& item : command_line.array) {
        if (item.kind != JsonValue::Kind::String) {
            throw ParseError(std::string(label) + ".command_line contains a non-string");
        }
    }
    const JsonValue& provenance = member(value, "provenance", label);
    exact_keys(provenance, {"benchmark_binary", "payload", "wheel"}, std::string(label) + ".provenance");
    validate_identity_object(
        member(provenance, "benchmark_binary", std::string(label) + ".provenance"),
        std::string(label) + ".provenance.benchmark_binary");
    for (const std::string_view key : {std::string_view("payload"), std::string_view("wheel")}) {
        const JsonValue& identity = member(provenance, key, std::string(label) + ".provenance");
        if (identity.kind != JsonValue::Kind::Null) {
            validate_identity_object(identity, std::string(label) + ".provenance." + std::string(key));
        }
    }
    return binding;
}

template <typename DigestFn>
Plan parse_plan(std::string_view text, uint32_t max_loop_count, DigestFn&& digest_fn) {
    JsonValue root = JsonParser(text).parse_document();
    exact_keys(
        root,
        {"bindings", "entries", "entry_count", "headroom_factor", "max_loop_count",
         "plan_sha256", "policy", "schema", "scope", "source_commits", "source_count",
         "variants", "version"},
        "plan");
    if (root.kind != JsonValue::Kind::Object) throw ParseError("plan root must be an object");
    if (canonical_json(root) != text) throw ParseError("plan JSON is not canonical");
    const std::string& schema = string_member(root, "schema", "plan");
    const std::string& policy = string_member(root, "policy", "plan");
    if (schema != "gafime.native-loop-plan.v1") throw ParseError("unsupported loop-plan schema");
    if (policy != "max_calibration_count_times_fixed_headroom_factor") {
        throw ParseError("unsupported loop-plan policy");
    }
    if (uint_member(root, "version", "plan") != 1 || uint_member(root, "source_count", "plan") != 2) {
        throw ParseError("loop-plan version/source_count is unsupported");
    }
    const std::string& declared_digest = string_member(root, "plan_sha256", "plan");
    if (!hex_string(declared_digest, 64)) throw ParseError("loop-plan SHA-256 is invalid");
    JsonValue unsigned_root = root;
    unsigned_root.object.at("plan_sha256").string.assign(64, '0');
    const std::string unsigned_text = canonical_json(unsigned_root);
    if (digest_fn(unsigned_text.data(), unsigned_text.size()) != declared_digest) {
        throw ParseError("loop-plan SHA-256 does not match canonical contents");
    }

    const JsonValue& scope = member(root, "scope", "plan");
    exact_keys(scope, {"artifact_kind", "backend", "device", "evidence_lane", "input_policy", "scope_id", "workload"}, "plan.scope");
    const JsonValue& workload = member(scope, "workload", "plan.scope");
    exact_keys(workload, {"arity", "candidates", "features", "mi_bins", "name", "rows", "top_k"}, "plan.scope.workload");
    Plan plan;
    plan.semantic_sha256 = declared_digest;
    // ``plan_sha256`` authenticates canonical unsigned contents.  The raw
    // serialized-file digest is a separate, external identity and must never
    // be embedded in the plan (that would create a self-referential hash).
    plan.file_sha256 = digest_fn(text.data(), text.size());
    plan.backend = string_member(scope, "backend", "plan.scope");
    plan.workload = string_member(workload, "name", "plan.scope.workload");
    plan.rows = uint_member(workload, "rows", "plan.scope.workload", true);
    plan.features = uint_member(workload, "features", "plan.scope.workload", true);
    plan.candidates = uint_member(workload, "candidates", "plan.scope.workload", true);
    plan.arity = uint_member(workload, "arity", "plan.scope.workload", true);
    plan.mi_bins = uint_member(workload, "mi_bins", "plan.scope.workload", true);
    plan.top_k = uint_member(workload, "top_k", "plan.scope.workload", true);
    plan.input_policy = string_member(scope, "input_policy", "plan.scope");
    plan.evidence_lane = string_member(scope, "evidence_lane", "plan.scope");
    plan.artifact_kind = string_member(scope, "artifact_kind", "plan.scope");
    plan.scope_id = string_member(scope, "scope_id", "plan.scope");
    plan.device_json = canonical_member(scope, "device", "plan.scope");
    if (plan.backend.empty() || plan.workload.empty() || plan.input_policy.empty() ||
        plan.evidence_lane.empty() || plan.artifact_kind.empty() || plan.scope_id.empty()) {
        throw ParseError("loop-plan scope contains an empty binding");
    }

    const uint64_t plan_cap = uint_member(root, "max_loop_count", "plan", true);
    const uint64_t helper_cap = std::min<uint64_t>(plan_cap, max_loop_count);
    if (plan_cap > max_loop_count) throw ParseError("loop-plan max_loop_count exceeds helper cap");
    static_cast<void>(uint_member(root, "headroom_factor", "plan", true));

    const JsonValue& variants = member(root, "variants", "plan");
    if (variants.kind != JsonValue::Kind::Array || variants.array.size() != 2 ||
        variants.array[0].kind != JsonValue::Kind::String || variants.array[1].kind != JsonValue::Kind::String ||
        variants.array[0].string != "baseline" || variants.array[1].string != "candidate") {
        throw ParseError("loop-plan variants must be [baseline,candidate]");
    }
    const JsonValue& source_commits = member(root, "source_commits", "plan");
    if (source_commits.kind != JsonValue::Kind::Array || source_commits.array.size() != 2) {
        throw ParseError("loop-plan source_commits must contain two entries");
    }
    std::set<std::string> commit_set;
    for (const JsonValue& item : source_commits.array) {
        if (item.kind != JsonValue::Kind::String || !hex_string(item.string, 40) ||
            !commit_set.insert(item.string).second) {
            throw ParseError("loop-plan source_commits must contain distinct full SHAs");
        }
    }

    const JsonValue& bindings = member(root, "bindings", "plan");
    if (bindings.kind != JsonValue::Kind::Array || bindings.array.size() != 2) {
        throw ParseError("loop-plan bindings must contain baseline and candidate");
    }
    for (size_t index = 0; index < bindings.array.size(); ++index) {
        plan.bindings.push_back(parse_binding(bindings.array[index], "plan.bindings[" + std::to_string(index) + "]"));
    }
    std::set<std::string> binding_variants;
    std::set<std::string> binding_commits;
    std::string input_identity;
    std::string harness_source_commit;
    for (const Binding& binding : plan.bindings) {
        if (binding.backend != plan.backend || binding.workload_json != canonical_member(scope, "workload", "plan.scope") ||
            binding.input_policy != plan.input_policy || binding.device_json != plan.device_json) {
            throw ParseError("loop-plan binding does not match its scope");
        }
        if (!binding_variants.insert(binding.variant).second || !binding_commits.insert(binding.source_commit).second) {
            throw ParseError("loop-plan bindings contain duplicate variant or commit");
        }
        if (input_identity.empty()) input_identity = binding.input_identity_json;
        if (input_identity != binding.input_identity_json) {
            throw ParseError("baseline and candidate input identities differ");
        }
        if (harness_source_commit.empty()) {
            harness_source_commit = binding.harness_source_commit;
        }
        if (harness_source_commit != binding.harness_source_commit) {
            throw ParseError("baseline and candidate calibration harness commits differ");
        }
    }
    if (binding_variants != std::set<std::string>{"baseline", "candidate"} || binding_commits != commit_set) {
        throw ParseError("loop-plan binding variants/commits do not match root declarations");
    }

    const JsonValue& entries = member(root, "entries", "plan");
    if (entries.kind != JsonValue::Kind::Array || entries.array.empty()) {
        throw ParseError("loop-plan entries are required");
    }
    std::string previous_key;
    for (size_t index = 0; index < entries.array.size(); ++index) {
        const JsonValue& entry = entries.array[index];
        exact_keys(entry, {"key", "loop_count"}, "plan.entries[" + std::to_string(index) + "]");
        const std::string& key = string_member(entry, "key", "plan.entry");
        if (key.empty() || key.find('\n') != std::string::npos || (!previous_key.empty() && key <= previous_key)) {
            throw ParseError("loop-plan entries must be unique and sorted");
        }
        previous_key = key;
        const uint64_t count = uint_member(entry, "loop_count", "plan.entry", true);
        if (count > helper_cap || count > std::numeric_limits<uint32_t>::max()) {
            throw ParseError("loop-plan entry loop_count exceeds helper cap");
        }
        plan.entries.emplace(key, static_cast<uint32_t>(count));
    }
    if (uint_member(root, "entry_count", "plan") != plan.entries.size()) {
        throw ParseError("loop-plan entry_count does not match entries");
    }
    return plan;
}

inline void validate_scope(const Plan& plan, const ExpectedScope& expected) {
    if (plan.backend != expected.backend || plan.workload != expected.workload ||
        plan.rows != expected.rows || plan.features != expected.features ||
        plan.candidates != expected.candidates || plan.arity != expected.arity ||
        plan.mi_bins != expected.mi_bins || plan.top_k != expected.top_k ||
        plan.input_policy != expected.input_policy || plan.evidence_lane != expected.evidence_lane ||
        plan.artifact_kind != expected.artifact_kind || plan.device_json != expected.device_json ||
        plan.scope_id != expected.scope_id) {
        throw ParseError("immutable native loop plan scope does not exactly match helper bindings");
    }
}

inline void validate_variant_binding(
    const Plan& plan,
    std::string_view variant,
    std::string_view product_source_commit,
    std::string_view harness_source_commit
) {
    const auto match = std::find_if(
        plan.bindings.begin(), plan.bindings.end(), [&](const Binding& binding) {
            return binding.variant == variant &&
                binding.source_commit == product_source_commit &&
                binding.product_source_commit == product_source_commit &&
                binding.harness_source_commit == harness_source_commit;
        });
    if (match == plan.bindings.end()) {
        throw ParseError(
            "immutable native loop plan does not bind this variant/product/harness commit");
    }
}

inline std::string expected_cuda_device_json(
    std::string_view name,
    uint64_t compute_major,
    uint64_t compute_minor
) {
    JsonValue device = JsonValue::object_value({
        {"compute_major", JsonValue::number_value(std::to_string(compute_major))},
        {"compute_minor", JsonValue::number_value(std::to_string(compute_minor))},
        {"name", JsonValue::string_value(std::string(name))},
    });
    return canonical_json(device);
}

inline std::string expected_rocm_device_json(uint64_t ordinal) {
    JsonValue device = JsonValue::object_value({
        {"ordinal", JsonValue::number_value(std::to_string(ordinal))},
    });
    return canonical_json(device);
}

}  // namespace gafime_native_loop_plan
