#ifndef GAFIME_GPU_EXECUTION_GATE_HPP
#define GAFIME_GPU_EXECUTION_GATE_HPP

#include <mutex>

namespace gafime_gpu {

// The mutex itself belongs to the launcher's anonymous namespace: one gate per
// loaded payload, not an interposed process-global symbol or a vendor-runtime
// lock. Both ordinary ABI generations enter it once, at their outer exports.
// Internal adapters must not acquire it again. No Python callbacks occur here.
template <typename Mutex>
class BasicPayloadExecutionGuard {
public:
    explicit BasicPayloadExecutionGuard(Mutex& mutex) noexcept
        : lock_(mutex, std::defer_lock) {
        // A host synchronization failure must not throw across the C ABI or
        // proceed into the vendor runtime without owning the gate.
        try {
            lock_.lock();
        } catch (...) {
        }
    }

    BasicPayloadExecutionGuard(const BasicPayloadExecutionGuard&) = delete;
    BasicPayloadExecutionGuard& operator=(const BasicPayloadExecutionGuard&) = delete;

    bool acquired() const noexcept { return lock_.owns_lock(); }

private:
    std::unique_lock<Mutex> lock_;
};

using PayloadExecutionGuard = BasicPayloadExecutionGuard<std::mutex>;

}  // namespace gafime_gpu

#endif
