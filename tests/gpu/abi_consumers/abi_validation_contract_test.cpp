#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "../../../src/common/gpu_abi_impl.hpp"

namespace {

struct FutureConstBufferView {
    GafimeConstBufferView known;
    uint64_t future_field;
};

struct FutureMutableBufferView {
    GafimeMutableBufferView known;
    uint64_t future_field;
};

GafimeConstBufferView const_view(const void* data, uint64_t count) {
    GafimeConstBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_count = count;
    view.byte_length = count * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

GafimeMutableBufferView mutable_view(void* data, uint64_t count) {
    GafimeMutableBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = data;
    view.element_capacity = count;
    view.byte_length = count * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

int expect(int actual, int expected, const char* label) {
    if (actual == expected) return 0;
    std::fprintf(stderr, "%s: expected %d, got %d\n", label, expected, actual);
    return 1;
}

int expect_short_buffer_headers_fail_before_payload_reads() {
#if defined(__unix__)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;
    const size_t page = static_cast<size_t>(page_size);
    void* pages = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) return 1;
    auto* inaccessible = static_cast<unsigned char*>(pages) + page;
    if (mprotect(inaccessible, page, PROT_NONE) != 0) {
        munmap(pages, page * 2);
        return 1;
    }

    // Only the version/size header is live. A later field read faults on the
    // adjacent protected page instead of being hidden by adjacent allocation.
    const uint32_t header[] = {GAFIME_PRECISION_ABI_VERSION, 8};
    static_assert(sizeof(header) == 8, "ABI header must be eight bytes");
    auto* bytes = inaccessible - sizeof(header);
    std::memcpy(bytes, header, sizeof(header));
    int failed = 0;
    failed |= expect(
        gafime_gpu_abi::validate_const_buffer(
            reinterpret_cast<const GafimeConstBufferView*>(bytes), GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_ABI_MISMATCH,
        "short standalone const view");
    failed |= expect(
        gafime_gpu_abi::validate_mutable_buffer(
            reinterpret_cast<const GafimeMutableBufferView*>(bytes), GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_ABI_MISMATCH,
        "short standalone mutable view");
    munmap(pages, page * 2);
    return failed;
#else
    return 0;
#endif
}

}  // namespace

int main() {
    float value = 1.0f;
    int failed = 0;
    failed |= expect_short_buffer_headers_fail_before_payload_reads();

    GafimeConstBufferView prefix_const = const_view(&value, 1);
    prefix_const.struct_size = gafime_gpu_abi::kConstBufferStablePrefixSize;
    failed |= expect(
        gafime_gpu_abi::validate_const_buffer(&prefix_const, GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_OK,
        "complete standalone const stable prefix");
    prefix_const.struct_size -= 1;
    failed |= expect(
        gafime_gpu_abi::validate_const_buffer(&prefix_const, GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_ABI_MISMATCH,
        "short standalone const stable prefix");

    GafimeMutableBufferView prefix_mutable = mutable_view(&value, 1);
    prefix_mutable.struct_size = gafime_gpu_abi::kMutableBufferStablePrefixSize;
    failed |= expect(
        gafime_gpu_abi::validate_mutable_buffer(&prefix_mutable, GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_OK,
        "complete standalone mutable stable prefix");
    prefix_mutable.struct_size -= 1;
    failed |= expect(
        gafime_gpu_abi::validate_mutable_buffer(&prefix_mutable, GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_ABI_MISMATCH,
        "short standalone mutable stable prefix");

    FutureConstBufferView future_const{};
    future_const.known = const_view(&value, 1);
    future_const.known.struct_size = sizeof(future_const);
    future_const.future_field = UINT64_C(0x1234);
    failed |= expect(
        gafime_gpu_abi::validate_const_buffer(
            &future_const.known, GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_OK,
        "standalone const view future tail");

    FutureMutableBufferView future_mutable{};
    future_mutable.known = mutable_view(&value, 1);
    future_mutable.known.struct_size = sizeof(future_mutable);
    future_mutable.future_field = UINT64_C(0x5678);
    failed |= expect(
        gafime_gpu_abi::validate_mutable_buffer(
            &future_mutable.known, GAFIME_DTYPE_F32, 1),
        GAFIME_STATUS_OK,
        "standalone mutable view future tail");

    uint32_t combo = 0;
    uint32_t rank = 0;
    uint32_t family = 0;
    uint64_t candidate = 0;
    uint32_t row_flags = 0;
    GafimeNumericResultTable result{};
    result.abi_version = GAFIME_PRECISION_ABI_VERSION;
    result.struct_size = sizeof(result);
    result.max_arity = 1;
    result.metric_count = 1;
    result.capacity = 1;
    result.combo_indices = &combo;
    result.ranks = &rank;
    result.families = &family;
    result.candidate_ids = &candidate;
    result.row_flags = &row_flags;
    result.metric_values = future_mutable.known;
    failed |= expect(
        gafime_gpu_abi::validate_numeric_result_table(&result, GAFIME_DTYPE_F32),
        GAFIME_STATUS_INVALID_ARGUMENT,
        "embedded mutable view future tail");

    GafimeNumericSignificanceTable significance{};
    significance.abi_version = GAFIME_PRECISION_ABI_VERSION;
    significance.struct_size = sizeof(significance);
    significance.metric_count = 1;
    significance.row_count = 1;
    significance.candidate_ids = &candidate;
    significance.observed_metric_values = future_const.known;
    significance.p_values = mutable_view(&value, 1);
    failed |= expect(
        gafime_gpu_abi::validate_numeric_significance_table(
            &significance, GAFIME_DTYPE_F32),
        GAFIME_STATUS_INVALID_ARGUMENT,
        "embedded const view future tail");

    significance.observed_metric_values = const_view(&value, 1);
    significance.p_values = future_mutable.known;
    failed |= expect(
        gafime_gpu_abi::validate_numeric_significance_table(
            &significance, GAFIME_DTYPE_F32),
        GAFIME_STATUS_INVALID_ARGUMENT,
        "embedded mutable significance view future tail");

    return failed;
}
