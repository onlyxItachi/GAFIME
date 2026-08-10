#include <cstdint>
#include <cstdio>
#include <cstring>

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

}  // namespace

int main() {
    float value = 1.0f;
    int failed = 0;

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
