#include "native_loop_plan_parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: native_loop_plan_parser_test PLAN EXPECTED_SEMANTIC_SHA256 EXPECTED_FILE_SHA256\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    std::ostringstream contents;
    contents << input.rdbuf();
    try {
        const std::string unsigned_marker =
            "\"plan_sha256\":\"" + std::string(64, '0') + "\"";
        const auto plan = gafime_native_loop_plan::parse_plan(
            contents.str(), 1u << 21,
            [semantic = std::string(argv[2]), file = std::string(argv[3]), unsigned_marker](
                const void* data, size_t size) {
                const std::string_view text(static_cast<const char*>(data), size);
                return text.find(unsigned_marker) != std::string_view::npos ? semantic : file;
            });
        if (plan.entries.empty() || plan.bindings.size() != 2 ||
            plan.semantic_sha256 != argv[2] || plan.file_sha256 != argv[3] ||
            plan.bindings[0].relative_path.empty()) {
            return 4;
        }
        gafime_native_loop_plan::validate_variant_binding(
            plan, "baseline", std::string(40, '1'), std::string(40, '3'));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
