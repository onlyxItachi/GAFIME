#include "../../../src/common/gpu_execution_gate.hpp"

#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct ThrowingMutex {
    int unlocks = 0;
    void lock() { throw std::runtime_error("injected host lock failure"); }
    void unlock() noexcept { ++unlocks; }
};

struct RestoreBeforeUnlock {
    bool& restored;
    ~RestoreBeforeUnlock() { restored = true; }
};

bool exclusion_and_restoration() {
    std::mutex mutex;
    std::promise<void> attempted;
    std::future<void> attempt = attempted.get_future();
    bool restored = false;
    bool contender_ok = false;
    std::thread contender;
    {
        gafime_gpu::PayloadExecutionGuard guard(mutex);
        if (!guard.acquired()) return false;
        RestoreBeforeUnlock device_guard{restored};
        contender = std::thread([&] {
            // This checks exclusion deterministically while the owner waits.
            const bool improperly_acquired = mutex.try_lock();
            if (improperly_acquired) mutex.unlock();
            attempted.set_value();
            gafime_gpu::PayloadExecutionGuard next(mutex);
            contender_ok = !improperly_acquired && next.acquired() && restored;
        });
        attempt.wait();
    }
    contender.join();
    return contender_ok;
}

bool contention_and_unwind() {
    std::mutex mutex;
    int shared_count = 0;
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&] {
            for (int iteration = 0; iteration < 1000; ++iteration) {
                gafime_gpu::PayloadExecutionGuard guard(mutex);
                if (!guard.acquired()) {
                    failed = true;
                    return;
                }
                ++shared_count;
            }
        });
    }
    for (auto& worker : workers) worker.join();
    try {
        gafime_gpu::PayloadExecutionGuard guard(mutex);
        if (!guard.acquired()) return false;
        throw std::runtime_error("injected operation failure");
    } catch (const std::runtime_error&) {
    }
    gafime_gpu::PayloadExecutionGuard after_unwind(mutex);
    return after_unwind.acquired() && !failed && shared_count == 4000;
}

bool acquisition_failure_is_closed() {
    ThrowingMutex mutex;
    bool destroyed = false;
    // Models the legacy void free path: it cannot report failure and must not
    // destroy a handle unless it owns the gate. The allocation remains live.
    const auto legacy_free = [&] {
        gafime_gpu::BasicPayloadExecutionGuard<ThrowingMutex> guard(mutex);
        if (!guard.acquired()) return;
        destroyed = true;
    };
    legacy_free();
    gafime_gpu::BasicPayloadExecutionGuard<ThrowingMutex> guard(mutex);
    return !guard.acquired() && !destroyed && mutex.unlocks == 0;
}

}  // namespace

int main() {
    return exclusion_and_restoration() && contention_and_unwind() &&
            acquisition_failure_is_closed()
        ? 0 : 1;
}
