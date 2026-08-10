#pragma once

// Git provenance for native benchmark helpers is deliberately independent of
// PATH and of the caller's Git configuration.  The CMake target supplies an
// absolute, validated executable and its build-time identity.  All Git
// commands below run with inherited GIT_* variables removed before the shell
// is started, so a linked worktree or alternate object database cannot change
// the repository being authenticated.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
extern char** _environ;
#else
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#ifndef GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE
#define GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE ""
#endif
#ifndef GAFIME_NATIVE_TRUSTED_GIT_SHA256
#define GAFIME_NATIVE_TRUSTED_GIT_SHA256 ""
#endif
#ifndef GAFIME_NATIVE_TRUSTED_GIT_VERSION
#define GAFIME_NATIVE_TRUSTED_GIT_VERSION ""
#endif

namespace gafime_native_trusted_git {

inline std::string shell_quote(std::string_view value) {
    std::string quoted("'");
    for (const char character : value) {
        if (character == '\'') quoted += "'\\''";
        else quoted += character;
    }
    quoted += '\'';
    return quoted;
}

inline std::string trim(std::string value) {
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                              value.back() == ' ' || value.back() == '\t')) {
        value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && (value[first] == ' ' || value[first] == '\t')) ++first;
    return value.substr(first);
}

inline std::string canonical_path(const std::string& input) {
    std::error_code error;
    std::filesystem::path path(input);
    if (path.empty()) return {};
    if (!path.is_absolute()) path = std::filesystem::absolute(path, error);
    if (error) return path.string();
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    return error ? path.string() : canonical.string();
}

inline void put_environment(const std::string& name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

inline void unset_environment(const std::string& name) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), "");
#else
    unsetenv(name.c_str());
#endif
}

struct SavedEnvironmentValue {
    std::string name;
    std::string value;
};

class ScrubbedGitEnvironment {
public:
    ScrubbedGitEnvironment() {
#if defined(_WIN32)
        char** entries = _environ;
#else
        char** entries = environ;
#endif
        if (entries != nullptr) {
            for (char** entry = entries; *entry != nullptr; ++entry) {
                const std::string value(*entry);
                const size_t separator = value.find('=');
                if (separator == std::string::npos) continue;
                const std::string name = value.substr(0, separator);
                if (name.starts_with("GIT_")) {
                    saved_.push_back({name, value.substr(separator + 1)});
                }
            }
        }
        // Remove every inherited GIT_* name, including redirection, object
        // database, worktree, index and config-injection variables.
        for (const auto& saved : saved_) unset_environment(saved.name);
        put_environment("GIT_CONFIG_NOSYSTEM", "1");
#if defined(_WIN32)
        put_environment("GIT_CONFIG_GLOBAL", "NUL");
        put_environment("GIT_CONFIG_SYSTEM", "NUL");
#else
        put_environment("GIT_CONFIG_GLOBAL", "/dev/null");
        put_environment("GIT_CONFIG_SYSTEM", "/dev/null");
#endif
        put_environment("GIT_CONFIG_COUNT", "0");
        put_environment("GIT_TERMINAL_PROMPT", "0");
    }

    ~ScrubbedGitEnvironment() {
        unset_environment("GIT_CONFIG_NOSYSTEM");
        unset_environment("GIT_CONFIG_GLOBAL");
        unset_environment("GIT_CONFIG_SYSTEM");
        unset_environment("GIT_CONFIG_COUNT");
        unset_environment("GIT_TERMINAL_PROMPT");
        for (const auto& saved : saved_) put_environment(saved.name, saved.value);
    }

    ScrubbedGitEnvironment(const ScrubbedGitEnvironment&) = delete;
    ScrubbedGitEnvironment& operator=(const ScrubbedGitEnvironment&) = delete;

private:
    std::vector<SavedEnvironmentValue> saved_;
};

struct CommandResult {
    std::string output;
    int close_status = -1;
    int exit_status = -1;
    bool started = false;
    bool exited = false;

    [[nodiscard]] bool succeeded() const {
        return started && exited && exit_status == 0;
    }
};

inline CommandResult run_command(const std::string& command) {
    CommandResult result;
#if defined(_WIN32)
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (pipe == nullptr) return result;
    result.started = true;
    char buffer[512]{};
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) result.output += buffer;
#if defined(_WIN32)
    result.close_status = _pclose(pipe);
    result.exit_status = result.close_status;
    result.exited = result.close_status >= 0;
#else
    result.close_status = pclose(pipe);
    if (result.close_status >= 0 && WIFEXITED(result.close_status)) {
        result.exit_status = WEXITSTATUS(result.close_status);
        result.exited = true;
    } else if (result.close_status >= 0 && WIFSIGNALED(result.close_status)) {
        result.exit_status = 128 + WTERMSIG(result.close_status);
    }
#endif
    result.output = trim(std::move(result.output));
    return result;
}

inline CommandResult git_command(const std::string& root, const std::string& arguments) {
    ScrubbedGitEnvironment scrubbed;
    const std::string command = shell_quote(GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE) +
        " -C " + shell_quote(root) + " " + arguments + " 2>/dev/null";
    return run_command(command);
}

inline CommandResult git_command_no_root(const std::string& arguments) {
    ScrubbedGitEnvironment scrubbed;
    const std::string command = shell_quote(GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE) +
        " " + arguments + " 2>/dev/null";
    return run_command(command);
}

inline std::string executable_sha256(const std::string& path) {
#if defined(_WIN32)
    static_cast<void>(path);
    return {};
#else
    for (const char* hasher : {"/usr/bin/sha256sum", "/bin/sha256sum"}) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(hasher, error)) continue;
        const CommandResult command = run_command(
            shell_quote(hasher) + " " + shell_quote(path) + " 2>/dev/null");
        if (!command.succeeded()) continue;
        const std::string& output = command.output;
        const size_t separator = output.find_first_of(" \t");
        const std::string digest = output.substr(0, separator);
        if (digest.size() == 64 &&
            std::all_of(digest.begin(), digest.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
            })) {
            return digest;
        }
    }
    return {};
#endif
}

struct GitIdentity {
    std::string executable;
    std::string sha256;
    std::string runtime_sha256;
    std::string compiled_version;
    std::string runtime_version;
    bool executable_verified = false;
    bool environment_scrubbed = true;
};

inline GitIdentity identity() {
    GitIdentity result;
    result.executable = canonical_path(GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE);
    result.sha256 = GAFIME_NATIVE_TRUSTED_GIT_SHA256;
    result.runtime_sha256 = executable_sha256(result.executable);
    result.compiled_version = GAFIME_NATIVE_TRUSTED_GIT_VERSION;
    const CommandResult version = git_command_no_root("--version");
    result.runtime_version = version.output;
    result.executable_verified =
        !result.executable.empty() &&
        std::filesystem::is_regular_file(result.executable) &&
        !result.sha256.empty() && result.runtime_sha256 == result.sha256 &&
        version.succeeded() &&
        !result.runtime_version.empty();
    return result;
}

struct RepositoryIdentity {
    GitIdentity git;
    std::string root;
    std::string commit;
    std::string show_toplevel;
    std::string git_dir;
    std::string common_dir;
    std::string expected_git_dir;
    std::string expected_common_dir;
    bool verified = false;
    std::string detail;
};

inline std::string resolve_git_path(
    const std::filesystem::path& root, const std::string& value) {
    std::filesystem::path path(value);
    if (!path.is_absolute()) path = root / path;
    return canonical_path(path.string());
}

inline std::string expected_git_dir(const std::filesystem::path& root) {
    const std::filesystem::path dotgit = root / ".git";
    std::error_code error;
    if (std::filesystem::is_directory(dotgit, error)) return canonical_path(dotgit.string());
    if (!std::filesystem::is_regular_file(dotgit, error)) return {};
    std::ifstream input(dotgit);
    std::string line;
    if (!input || !std::getline(input, line)) return {};
    line = trim(std::move(line));
    constexpr std::string_view prefix = "gitdir:";
    if (!line.starts_with(prefix)) return {};
    return resolve_git_path(dotgit.parent_path(), trim(line.substr(prefix.size())));
}

inline std::string expected_common_dir(const std::string& expected_dir) {
    if (expected_dir.empty()) return {};
    const std::filesystem::path commondir =
        std::filesystem::path(expected_dir) / "commondir";
    std::error_code error;
    if (!std::filesystem::is_regular_file(commondir, error)) return expected_dir;
    std::ifstream input(commondir);
    std::string line;
    if (!input || !std::getline(input, line)) return {};
    line = trim(std::move(line));
    return resolve_git_path(std::filesystem::path(expected_dir), line);
}

inline RepositoryIdentity inspect(const std::string& input_root) {
    RepositoryIdentity result;
    result.git = identity();
    result.root = canonical_path(input_root);
    const std::filesystem::path root(result.root);
    if (result.root.empty()) {
        result.detail = "empty source root";
        return result;
    }
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) {
        result.detail = "source root is not a directory";
        return result;
    }
    const CommandResult top = git_command(result.root, "rev-parse --show-toplevel");
    const CommandResult git_dir = git_command(result.root, "rev-parse --git-dir");
    const CommandResult common_dir = git_command(result.root, "rev-parse --git-common-dir");
    const CommandResult commit = git_command(result.root, "rev-parse HEAD");
    if (!top.succeeded() || !git_dir.succeeded() || !common_dir.succeeded() ||
        !commit.succeeded()) {
        result.detail = "trusted Git repository inspection command failed";
        return result;
    }
    result.show_toplevel = canonical_path(top.output);
    result.git_dir = resolve_git_path(root, git_dir.output);
    result.common_dir = resolve_git_path(root, common_dir.output);
    result.expected_git_dir = expected_git_dir(root);
    result.expected_common_dir = expected_common_dir(result.expected_git_dir);
    result.commit = commit.output;
    if (!result.git.executable_verified) {
        result.detail = "trusted Git executable SHA-256/version verification failed";
    } else if (result.show_toplevel != result.root) {
        result.detail = "rev-parse --show-toplevel does not equal the requested source root";
    } else if (result.git_dir.empty() || result.git_dir != result.expected_git_dir) {
        result.detail = "rev-parse --git-dir does not match the physical .git target";
    } else if (result.common_dir.empty() || result.common_dir != result.expected_common_dir) {
        result.detail = "rev-parse --git-common-dir does not match the physical common dir";
    } else if (result.commit.empty()) {
        result.detail = "rev-parse HEAD returned no commit";
    } else {
        result.verified = true;
    }
    return result;
}

inline bool is_full_commit(const std::string& value) {
    return value.size() == 40 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
    });
}

}  // namespace gafime_native_trusted_git
