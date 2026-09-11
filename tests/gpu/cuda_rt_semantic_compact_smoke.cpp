// Direct local-RT compact-evidence exercise.  This is intentionally a
// correctness, ownership, and resource-admission fixture rather than a timing
// benchmark.  Its host oracle evaluates the frozen fp32 predicates directly
// and compares exact integer sufficient statistics; it never reads a dense
// materialized region column.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "../../src/common/gpu_abi_impl.hpp"
#include "../../src/cuda/rt_abi.hpp"

namespace {

constexpr uint32_t kAllStatistics =
    GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY |
    GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_PAIRED |
    GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_LABELED;
constexpr uint32_t kAllFinalizers =
    GAFIME_SEMANTIC_RT_REGION_FINALIZE_OCCUPANCY |
    GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_AGREEMENT |
    GAFIME_SEMANTIC_RT_REGION_FINALIZE_PAIRED_IOU |
    GAFIME_SEMANTIC_RT_REGION_FINALIZE_LABELED_GINI_GAIN;
constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kMiB = 1024u * 1024u;
// Three live route queries remain below the shared 1 GiB experiment envelope:
// the two mask-based SM queries require about 262 MiB each for the largest
// paired case; a proof-gated RT grid query is materially smaller.
constexpr uint64_t kBenchmarkPersistentBudget = 300u * kMiB;
constexpr uint64_t kBenchmarkTemporaryBudget = 64u * kMiB;
constexpr uint32_t kBenchmarkWarmups = 3u;
constexpr uint32_t kBenchmarkSamples = 9u;

struct BenchmarkRoute {
    const char* name;
    uint32_t flags;
};

constexpr std::array<BenchmarkRoute, 3> kBenchmarkRoutes = {{
    {"require_rt", GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT},
    {"sm_exhaustive", GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE},
    {"query_binned_cuda", GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM},
}};

int fail(const char* message, int status = GAFIME_STATUS_OK) {
    if (status == GAFIME_STATUS_OK) {
        std::fprintf(stderr, "CUDA RT semantic-compact smoke: %s\n", message);
    } else {
        std::fprintf(
            stderr,
            "CUDA RT semantic-compact smoke: %s (status %d)\n",
            message,
            status
        );
    }
    return 1;
}

bool require(bool condition, const char* message) {
    if (condition) return true;
    static_cast<void>(fail(message));
    return false;
}

uint64_t f32_bits(float value) {
    return static_cast<uint64_t>(std::bit_cast<uint32_t>(value));
}

float f32_from_bits(uint64_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

// The short ABI prefix is deliberately backed by sentinel bytes and aligned
// for the complete table type.  Native must inspect only the declared prefix
// before rejecting it; this is a host-only regression, not a CUDA allocation.
struct alignas(GafimeSemanticRtRegionStatsTable) CompactStatsShortPrefix {
    uint32_t abi_version;
    uint32_t struct_size;
    std::array<unsigned char, sizeof(GafimeSemanticRtRegionStatsTable) + 32u> sentinel;
};

static_assert(offsetof(CompactStatsShortPrefix, sentinel) == 2u * sizeof(uint32_t));
static_assert(alignof(CompactStatsShortPrefix) >= alignof(GafimeSemanticRtRegionStatsTable));

template <typename T>
const T* unreadable_misaligned_abi_pointer() {
    // The test never dereferences this address.  A native implementation must
    // reject it on alignment before it indexes/copies the caller array.
    return reinterpret_cast<const T*>(static_cast<uintptr_t>(1u));
}

int expect_misaligned_compact_create_array_rejection(
    const char* context,
    const GafimeSemanticFrozenRegionTerm* terms,
    const uint32_t* paired_term_slots,
    const uint32_t* region_offsets,
    const uint32_t* partition_offsets,
    bool has_paired_bank
) {
    uint64_t primary_bank_marker = 0u;
    uint64_t paired_bank_marker = 0u;
    GafimeSemanticRtRegionQueryDesc desc{};
    desc.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.primary_bank = &primary_bank_marker;
    desc.paired_bank = has_paired_bank ? static_cast<void*>(&paired_bank_marker) : nullptr;
    desc.terms = terms;
    desc.paired_term_slots = paired_term_slots;
    desc.region_offsets = region_offsets;
    desc.partition_offsets = partition_offsets;
    desc.term_count = 1u;
    desc.region_count = 1u;
    desc.partition_count = 1u;
    desc.flags = GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE;
    desc.max_persistent_bytes = kUnlimited;

    GafimeGpuSemanticRegionQuery query = reinterpret_cast<void*>(static_cast<uintptr_t>(1u));
    uint64_t persistent = std::numeric_limits<uint64_t>::max();
    const int status = gafime_gpu_semantic_region_query_create_rt_v1(
        &desc, &query, &persistent);
    if (status != GAFIME_STATUS_INVALID_ARGUMENT || query != nullptr) {
        return fail(context, status);
    }
    return 0;
}

int exercise_host_only_compact_abi_rejections() {
    GafimeSemanticFrozenRegionTerm term{
        0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(0.0f)};
    const std::array<uint32_t, 2> region_offsets = {0u, 1u};
    const std::array<uint32_t, 2> partition_offsets = {0u, 1u};

    if (expect_misaligned_compact_create_array_rejection(
            "misaligned compact terms were not rejected before access",
            unreadable_misaligned_abi_pointer<GafimeSemanticFrozenRegionTerm>(),
            nullptr,
            region_offsets.data(),
            partition_offsets.data(),
            false) != 0) {
        return 1;
    }
    if (expect_misaligned_compact_create_array_rejection(
            "misaligned compact paired-term slots were not rejected before access",
            &term,
            unreadable_misaligned_abi_pointer<uint32_t>(),
            region_offsets.data(),
            partition_offsets.data(),
            true) != 0) {
        return 1;
    }
    if (expect_misaligned_compact_create_array_rejection(
            "misaligned compact region offsets were not rejected before access",
            &term,
            nullptr,
            unreadable_misaligned_abi_pointer<uint32_t>(),
            partition_offsets.data(),
            false) != 0) {
        return 1;
    }
    if (expect_misaligned_compact_create_array_rejection(
            "misaligned compact partition offsets were not rejected before access",
            &term,
            nullptr,
            region_offsets.data(),
            unreadable_misaligned_abi_pointer<uint32_t>(),
            false) != 0) {
        return 1;
    }

    CompactStatsShortPrefix short_stats{};
    short_stats.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    short_stats.struct_size = 2u * sizeof(uint32_t);
    short_stats.sentinel.fill(0xa5u);
    const CompactStatsShortPrefix before = short_stats;
    uint64_t temporary_peak = std::numeric_limits<uint64_t>::max();
    const int status = gafime_gpu_semantic_region_query_execute_rt_v1(
        nullptr,
        nullptr,
        reinterpret_cast<GafimeSemanticRtRegionStatsTable*>(&short_stats),
        &temporary_peak
    );
    if (status != GAFIME_STATUS_ABI_MISMATCH ||
        std::memcmp(&short_stats, &before, sizeof(short_stats)) != 0) {
        return fail("short compact stats ABI prefix was mutated before rejection", status);
    }
    return 0;
}

bool caller_device_is(int expected, const char* context) {
    int actual = -1;
    if (cudaGetDevice(&actual) != cudaSuccess || actual != expected) {
        std::fprintf(
            stderr,
            "CUDA RT semantic-compact smoke: caller device changed after %s (expected %d, got %d)\n",
            context,
            expected,
            actual
        );
        return false;
    }
    return true;
}

GafimeConstBufferView const_f32_view(const std::vector<float>& values) {
    GafimeConstBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = values.data();
    view.element_count = values.size();
    view.byte_length = values.size() * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

GafimeMutableBufferView mutable_f32_view(std::vector<float>& values) {
    GafimeMutableBufferView view{};
    view.abi_version = GAFIME_PRECISION_ABI_VERSION;
    view.struct_size = sizeof(view);
    view.dtype = GAFIME_DTYPE_F32;
    view.flags = GAFIME_BUFFER_FLAG_HOST | GAFIME_BUFFER_FLAG_CONTIGUOUS;
    view.data = values.data();
    view.element_capacity = values.size();
    view.byte_length = values.size() * sizeof(float);
    view.byte_stride = sizeof(float);
    return view;
}

struct Bank {
    GafimeGpuSemanticBank raw = nullptr;

    Bank() = default;
    Bank(const Bank&) = delete;
    Bank& operator=(const Bank&) = delete;

    ~Bank() {
        if (raw != nullptr) {
            static_cast<void>(gafime_gpu_semantic_bank_free_v1(raw));
        }
    }

    int close() {
        if (raw == nullptr) return GAFIME_STATUS_OK;
        const int status = gafime_gpu_semantic_bank_free_v1(raw);
        if (status == GAFIME_STATUS_OK) raw = nullptr;
        return status;
    }
};

bool allocate_uploaded_f32_bank(
    const std::vector<float>& column_major,
    uint64_t rows,
    uint32_t source_slots,
    uint32_t slot_capacity,
    Bank* bank_out
) {
    if (bank_out == nullptr || rows == 0u || source_slots == 0u || source_slots > slot_capacity ||
        column_major.size() != rows * static_cast<uint64_t>(source_slots)) {
        return false;
    }
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    GafimeSemanticBankDesc desc{};
    desc.abi_version = GAFIME_SEMANTIC_PRIMITIVES_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.route = route;
    desc.layout = GAFIME_MATRIX_COLUMN_MAJOR;
    desc.rows = rows;
    desc.source_slots = source_slots;
    // Extra capacity is an uninitialized physical slot, not a precomputed
    // dense root.  The coverage assertion below makes that distinction
    // observable: the fresh slot cannot be downloaded before evidence.
    desc.slot_capacity = slot_capacity;
    desc.bytes = rows * static_cast<uint64_t>(slot_capacity) * sizeof(float);
    GafimeGpuSemanticBank raw = nullptr;
    int status = gafime_gpu_semantic_bank_alloc_v1(0u, &desc, &raw);
    if (status != GAFIME_STATUS_OK || raw == nullptr) {
        return false;
    }
    const GafimeConstBufferView source = const_f32_view(column_major);
    status = gafime_gpu_semantic_bank_upload_v1(raw, &route, &source);
    if (status != GAFIME_STATUS_OK) {
        static_cast<void>(gafime_gpu_semantic_bank_free_v1(raw));
        return false;
    }
    bank_out->raw = raw;
    return true;
}

bool f32_slot_is_uninitialized(GafimeGpuSemanticBank bank, uint64_t rows, uint32_t slot) {
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    std::vector<float> values(rows, -123.0f);
    GafimeMutableBufferView view = mutable_f32_view(values);
    return gafime_gpu_semantic_bank_download_v1(bank, {&slot, 1u}, &route, &view) ==
        GAFIME_STATUS_INVALID_ARGUMENT;
}

bool download_f32_slot(
    GafimeGpuSemanticBank bank,
    uint64_t rows,
    uint32_t slot,
    std::vector<float>* values_out
) {
    if (values_out == nullptr) return false;
    values_out->assign(rows, -123.0f);
    const GafimeNumericRoute route = gafime_gpu_abi::numeric_route(GAFIME_PRECISION_FP32);
    GafimeMutableBufferView view = mutable_f32_view(*values_out);
    return gafime_gpu_semantic_bank_download_v1(bank, {&slot, 1u}, &route, &view) ==
        GAFIME_STATUS_OK;
}

struct Query {
    GafimeGpuSemanticRegionQuery raw = nullptr;

    Query() = default;
    Query(const Query&) = delete;
    Query& operator=(const Query&) = delete;

    ~Query() {
        if (raw != nullptr) {
            static_cast<void>(gafime_gpu_semantic_region_query_free_rt_v1(raw));
        }
    }

    int close() {
        if (raw == nullptr) return GAFIME_STATUS_OK;
        const int status = gafime_gpu_semantic_region_query_free_rt_v1(raw);
        if (status == GAFIME_STATUS_OK) raw = nullptr;
        return status;
    }
};

struct RegionSpec {
    std::vector<GafimeSemanticFrozenRegionTerm> terms;
    std::vector<uint32_t> paired_term_slots;
    std::vector<uint32_t> region_offsets;
    std::vector<uint32_t> partition_offsets;
};

RegionSpec tiny_region_spec() {
    RegionSpec spec{};
    spec.terms = {
        {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(-0.0f)},
        {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(-1.0f)},
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(2.0f)},
        {1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(1.0f)},
        {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(-2.0f)},
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(1.0f)},
    };
    // The paired bank deliberately has [junk, x, y] rather than [x, y].
    spec.paired_term_slots = {1u, 1u, 1u, 2u, 1u, 1u};
    spec.region_offsets = {0u, 1u, 4u, 6u};
    spec.partition_offsets = {0u, 3u};
    return spec;
}

GafimeSemanticRtRegionQueryDesc query_desc(
    GafimeGpuSemanticBank primary,
    GafimeGpuSemanticBank paired,
    const RegionSpec& spec,
    uint32_t flags,
    uint64_t max_persistent_bytes
) {
    GafimeSemanticRtRegionQueryDesc desc{};
    desc.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.primary_bank = primary;
    desc.paired_bank = paired;
    desc.terms = spec.terms.data();
    desc.paired_term_slots = paired == nullptr ? nullptr : spec.paired_term_slots.data();
    desc.region_offsets = spec.region_offsets.data();
    desc.partition_offsets = spec.partition_offsets.data();
    desc.term_count = spec.terms.size();
    desc.region_count = static_cast<uint32_t>(spec.region_offsets.size() - 1u);
    desc.partition_count = static_cast<uint32_t>(spec.partition_offsets.size() - 1u);
    desc.flags = flags;
    desc.max_persistent_bytes = max_persistent_bytes;
    return desc;
}

struct Labels {
    std::vector<uint64_t> rows;
    std::vector<uint8_t> values;
};

GafimeSemanticRtBinaryLabelContext raw_labels(const Labels& labels) {
    GafimeSemanticRtBinaryLabelContext raw{};
    raw.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    raw.struct_size = sizeof(raw);
    raw.row_indices = labels.rows.data();
    raw.values = labels.values.data();
    raw.count = labels.rows.size();
    return raw;
}

struct Execution {
    std::vector<GafimeSemanticRtRegionExactStats> records;
    GafimeSemanticRtRegionStatsTable table{};
    uint64_t temporary_peak = 0u;
};

int execute_query(
    GafimeGpuSemanticRegionQuery query,
    uint32_t region_count,
    uint32_t statistic_mask,
    uint32_t finalizer_mask,
    const Labels& labels,
    uint64_t max_temporary_bytes,
    Execution* execution
) {
    if (execution == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    execution->records.assign(region_count, {});
    // A failed all-channel execute must not leave a plausible partial record.
    std::memset(
        execution->records.data(),
        0xa5,
        execution->records.size() * sizeof(GafimeSemanticRtRegionExactStats)
    );
    execution->table = {};
    execution->table.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    execution->table.struct_size = sizeof(execution->table);
    execution->table.capacity = region_count;
    execution->table.records = execution->records.data();
    GafimeSemanticRtRegionExecuteDesc desc{};
    desc.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    desc.struct_size = sizeof(desc);
    desc.statistic_mask = statistic_mask;
    desc.finalizer_mask = finalizer_mask;
    desc.labels = raw_labels(labels);
    desc.max_temporary_bytes = max_temporary_bytes;
    execution->temporary_peak = 0u;
    return gafime_gpu_semantic_region_query_execute_rt_v1(
        query,
        &desc,
        &execution->table,
        &execution->temporary_peak
    );
}

// The benchmark path prepares host descriptors and output storage outside the
// timed interval.  Its samples therefore measure the real synchronous native
// execute call, not vector allocation or result formatting in this fixture.
struct PreparedExecution {
    std::vector<GafimeSemanticRtRegionExactStats> records;
    GafimeSemanticRtRegionStatsTable table{};
    GafimeSemanticRtRegionExecuteDesc desc{};
    uint64_t temporary_peak = 0u;
};

bool prepare_execution(
    uint32_t region_count,
    uint32_t statistic_mask,
    uint32_t finalizer_mask,
    const Labels& labels,
    uint64_t max_temporary_bytes,
    PreparedExecution* execution
) {
    if (execution == nullptr) return false;
    execution->records.assign(region_count, {});
    execution->table = {};
    execution->table.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    execution->table.struct_size = sizeof(execution->table);
    execution->table.capacity = region_count;
    execution->table.records = execution->records.data();
    execution->desc = {};
    execution->desc.abi_version = GAFIME_CUDA_RT_SEMANTIC_REGION_QUERY_ABI_VERSION;
    execution->desc.struct_size = sizeof(execution->desc);
    execution->desc.statistic_mask = statistic_mask;
    execution->desc.finalizer_mask = finalizer_mask;
    execution->desc.labels = raw_labels(labels);
    execution->desc.max_temporary_bytes = max_temporary_bytes;
    execution->temporary_peak = 0u;
    return true;
}

void reset_prepared_execution(PreparedExecution* execution) {
    if (execution == nullptr) return;
    std::memset(
        execution->records.data(),
        0,
        execution->records.size() * sizeof(GafimeSemanticRtRegionExactStats)
    );
    execution->table.requested_statistic_mask = 0u;
    execution->table.finalized_mask = 0u;
    execution->table.count = 0u;
    execution->temporary_peak = 0u;
}

int execute_prepared(GafimeGpuSemanticRegionQuery query, PreparedExecution* execution) {
    if (execution == nullptr) return GAFIME_STATUS_INVALID_ARGUMENT;
    reset_prepared_execution(execution);
    return gafime_gpu_semantic_region_query_execute_rt_v1(
        query,
        &execution->desc,
        &execution->table,
        &execution->temporary_peak
    );
}

bool holds(float value, const GafimeSemanticFrozenRegionTerm& term) {
    const float threshold = f32_from_bits(term.threshold_bits);
    if (term.relation == GAFIME_SEMANTIC_REGION_LESS_EQUAL) return value <= threshold;
    if (term.relation == GAFIME_SEMANTIC_REGION_GREATER_THAN) return value > threshold;
    return false;
}

struct OracleRecord {
    uint64_t row_count = 0u;
    uint64_t label_support = 0u;
    uint64_t occupancy_inside = 0u;
    uint64_t paired_n00 = 0u;
    uint64_t paired_n01 = 0u;
    uint64_t paired_n10 = 0u;
    uint64_t paired_n11 = 0u;
    uint64_t label_outside_0 = 0u;
    uint64_t label_outside_1 = 0u;
    uint64_t label_inside_0 = 0u;
    uint64_t label_inside_1 = 0u;
};

using Columns = std::vector<std::vector<float>>;

bool region_holds(
    const Columns& columns,
    const RegionSpec& spec,
    uint32_t begin,
    uint32_t end,
    uint64_t row,
    bool paired
) {
    for (uint32_t index = begin; index < end; ++index) {
        const auto& term = spec.terms[index];
        const uint32_t slot = paired ? spec.paired_term_slots[index] : term.input_slot;
        if (slot >= columns.size() || row >= columns[slot].size() || !holds(columns[slot][row], term)) {
            return false;
        }
    }
    return true;
}

std::vector<OracleRecord> exact_oracle(
    const Columns& primary,
    const Columns& paired,
    const RegionSpec& spec,
    const Labels& labels
) {
    const uint64_t rows = primary.empty() ? 0u : primary.front().size();
    std::vector<int> label_by_row(rows, -1);
    for (size_t index = 0; index < labels.rows.size(); ++index) {
        if (labels.rows[index] < rows && labels.values[index] <= 1u) {
            label_by_row[labels.rows[index]] = labels.values[index];
        }
    }
    std::vector<OracleRecord> output(spec.region_offsets.size() - 1u);
    for (size_t region = 0; region < output.size(); ++region) {
        OracleRecord record{};
        record.row_count = rows;
        record.label_support = labels.rows.size();
        const uint32_t begin = spec.region_offsets[region];
        const uint32_t end = spec.region_offsets[region + 1u];
        for (uint64_t row = 0; row < rows; ++row) {
            const bool primary_member = region_holds(primary, spec, begin, end, row, false);
            const bool paired_member = region_holds(paired, spec, begin, end, row, true);
            if (primary_member) ++record.occupancy_inside;
            if (!primary_member && !paired_member) ++record.paired_n00;
            if (!primary_member && paired_member) ++record.paired_n01;
            if (primary_member && !paired_member) ++record.paired_n10;
            if (primary_member && paired_member) ++record.paired_n11;
            const int label = label_by_row[row];
            if (label == 0) {
                if (primary_member) ++record.label_inside_0;
                else ++record.label_outside_0;
            } else if (label == 1) {
                if (primary_member) ++record.label_inside_1;
                else ++record.label_outside_1;
            }
        }
        output[region] = record;
    }
    return output;
}

bool same_exact(const GafimeSemanticRtRegionExactStats& actual, const OracleRecord& expected) {
    return actual.row_count == expected.row_count &&
        actual.label_support == expected.label_support &&
        actual.occupancy_inside == expected.occupancy_inside &&
        actual.paired_n00 == expected.paired_n00 &&
        actual.paired_n01 == expected.paired_n01 &&
        actual.paired_n10 == expected.paired_n10 &&
        actual.paired_n11 == expected.paired_n11 &&
        actual.label_outside_0 == expected.label_outside_0 &&
        actual.label_outside_1 == expected.label_outside_1 &&
        actual.label_inside_0 == expected.label_inside_0 &&
        actual.label_inside_1 == expected.label_inside_1;
}

bool same_native_record(
    const GafimeSemanticRtRegionExactStats& left,
    const GafimeSemanticRtRegionExactStats& right
) {
    return left.row_count == right.row_count &&
        left.label_support == right.label_support &&
        left.occupancy_inside == right.occupancy_inside &&
        left.paired_n00 == right.paired_n00 &&
        left.paired_n01 == right.paired_n01 &&
        left.paired_n10 == right.paired_n10 &&
        left.paired_n11 == right.paired_n11 &&
        left.label_outside_0 == right.label_outside_0 &&
        left.label_outside_1 == right.label_outside_1 &&
        left.label_inside_0 == right.label_inside_0 &&
        left.label_inside_1 == right.label_inside_1 &&
        std::bit_cast<uint32_t>(left.occupancy) == std::bit_cast<uint32_t>(right.occupancy) &&
        std::bit_cast<uint32_t>(left.paired_agreement) ==
            std::bit_cast<uint32_t>(right.paired_agreement) &&
        std::bit_cast<uint32_t>(left.paired_iou) == std::bit_cast<uint32_t>(right.paired_iou) &&
        std::bit_cast<uint32_t>(left.labeled_gini_gain) ==
            std::bit_cast<uint32_t>(right.labeled_gini_gain) &&
        left.occupancy_state == right.occupancy_state &&
        left.paired_agreement_state == right.paired_agreement_state &&
        left.paired_iou_state == right.paired_iou_state &&
        left.labeled_gini_gain_state == right.labeled_gini_gain_state;
}

bool same_completed_native_records(
    const PreparedExecution& left,
    const PreparedExecution& right,
    uint32_t region_count
) {
    if (left.table.requested_statistic_mask != kAllStatistics ||
        right.table.requested_statistic_mask != kAllStatistics ||
        left.table.finalized_mask != kAllFinalizers ||
        right.table.finalized_mask != kAllFinalizers ||
        left.table.count != region_count || right.table.count != region_count ||
        left.records.size() != region_count || right.records.size() != region_count) {
        return false;
    }
    for (uint32_t region = 0u; region < region_count; ++region) {
        if (!same_native_record(left.records[region], right.records[region])) return false;
    }
    return true;
}

bool same_native_record_vectors(
    const std::vector<GafimeSemanticRtRegionExactStats>& left,
    const std::vector<GafimeSemanticRtRegionExactStats>& right
) {
    if (left.size() != right.size()) return false;
    for (size_t region = 0u; region < left.size(); ++region) {
        if (!same_native_record(left[region], right[region])) return false;
    }
    return true;
}

bool same_oracle(const OracleRecord& left, const OracleRecord& right) {
    return left.row_count == right.row_count &&
        left.label_support == right.label_support &&
        left.occupancy_inside == right.occupancy_inside &&
        left.paired_n00 == right.paired_n00 &&
        left.paired_n01 == right.paired_n01 &&
        left.paired_n10 == right.paired_n10 &&
        left.paired_n11 == right.paired_n11 &&
        left.label_outside_0 == right.label_outside_0 &&
        left.label_outside_1 == right.label_outside_1 &&
        left.label_inside_0 == right.label_inside_0 &&
        left.label_inside_1 == right.label_inside_1;
}

float gini_f32(uint64_t zero_count, uint64_t one_count) {
    const float total = static_cast<float>(zero_count + one_count);
    const float p0 = static_cast<float>(zero_count) / total;
    const float p1 = static_cast<float>(one_count) / total;
    return 1.0f - p0 * p0 - p1 * p1;
}

bool approximately_equal(float left, float right) {
    return std::fabs(left - right) <= 2.0e-6f;
}

bool check_finalizers(const GafimeSemanticRtRegionExactStats& record) {
    if (record.occupancy_state != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        record.paired_agreement_state != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        record.paired_iou_state != GAFIME_SEMANTIC_SCALAR_MEASURED ||
        record.labeled_gini_gain_state != GAFIME_SEMANTIC_SCALAR_MEASURED) {
        return false;
    }
    const uint64_t paired_union = record.paired_n01 + record.paired_n10 + record.paired_n11;
    const uint64_t support = record.label_support;
    const uint64_t outside = record.label_outside_0 + record.label_outside_1;
    const uint64_t inside = record.label_inside_0 + record.label_inside_1;
    const uint64_t global_zero = record.label_outside_0 + record.label_inside_0;
    const uint64_t global_one = record.label_outside_1 + record.label_inside_1;
    const float expected_occupancy =
        static_cast<float>(record.occupancy_inside) / static_cast<float>(record.row_count);
    const float expected_agreement = static_cast<float>(record.paired_n00 + record.paired_n11) /
        static_cast<float>(record.row_count);
    const float expected_iou = static_cast<float>(record.paired_n11) /
        static_cast<float>(paired_union);
    const float expected_gain =
        (gini_f32(global_zero, global_one) -
         (static_cast<float>(outside) / static_cast<float>(support)) *
             gini_f32(record.label_outside_0, record.label_outside_1)) -
        (static_cast<float>(inside) / static_cast<float>(support)) *
            gini_f32(record.label_inside_0, record.label_inside_1);
    return approximately_equal(record.occupancy, expected_occupancy) &&
        approximately_equal(record.paired_agreement, expected_agreement) &&
        approximately_equal(record.paired_iou, expected_iou) &&
        approximately_equal(record.labeled_gini_gain, expected_gain);
}

bool prepared_finalizers_are_valid(const PreparedExecution& execution, uint32_t region_count) {
    if (execution.table.count != region_count || execution.records.size() != region_count) {
        return false;
    }
    for (const auto& record : execution.records) {
        if (!check_finalizers(record)) return false;
    }
    return true;
}

bool records_match(
    const Execution& execution,
    const std::vector<OracleRecord>& expected,
    bool require_finalizers
) {
    if (execution.table.count != expected.size() || execution.records.size() != expected.size()) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!same_exact(execution.records[index], expected[index])) return false;
        if (require_finalizers && !check_finalizers(execution.records[index])) return false;
    }
    return true;
}

std::vector<float> column_major(const Columns& columns) {
    std::vector<float> output;
    for (const auto& column : columns) output.insert(output.end(), column.begin(), column.end());
    return output;
}

bool all_records_unchanged(const Execution& execution) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(execution.records.data());
    const size_t count = execution.records.size() * sizeof(GafimeSemanticRtRegionExactStats);
    return std::all_of(bytes, bytes + count, [](unsigned char value) { return value == 0xa5u; });
}

int exercise_tiny_rt_smoke(int caller_device) {
    const Columns primary = {
        {-1.0f, -0.0f, 0.0f, 0.5f, 1.0f, 2.0f, -2.0f, 3.0f},
        {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 2.0f},
    };
    const Columns paired = {
        {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f},
        {-1.0f, 1.0f, 0.0f, 0.5f, -1.0f, 2.0f, 2.0f, 3.0f},
        {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 2.0f},
    };
    Labels labels{{0u, 1u, 3u, 4u, 5u, 7u}, {0u, 0u, 1u, 1u, 1u, 1u}};
    RegionSpec spec = tiny_region_spec();
    const auto expected = exact_oracle(primary, paired, spec, labels);
    const std::array<OracleRecord, 3> manual = {{
        {8u, 6u, 4u, 2u, 2u, 1u, 3u, 2u, 0u, 0u, 4u},
        {8u, 6u, 5u, 2u, 1u, 1u, 4u, 1u, 1u, 1u, 3u},
        {8u, 6u, 5u, 3u, 0u, 0u, 5u, 0u, 2u, 2u, 2u},
    }};
    if (!require(expected.size() == manual.size(), "tiny host oracle region count changed")) return 1;
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!same_oracle(manual[index], expected[index])) {
            // Do not permit a self-consistent generated oracle to hide a
            // threshold/order regression in this hand-audited fixture.
            return fail("tiny host oracle no longer matches the hand-audited count table");
        }
    }

    Bank primary_bank;
    Bank paired_bank;
    if (!allocate_uploaded_f32_bank(column_major(primary), 8u, 2u, 3u, &primary_bank) ||
        !allocate_uploaded_f32_bank(column_major(paired), 8u, 3u, 3u, &paired_bank)) {
        return fail("compact semantic-bank setup failed");
    }
    if (!require(primary_bank.raw != nullptr && paired_bank.raw != nullptr,
                 "compact test bank setup returned null")) return 1;

    // Establish actual OptiX availability before treating an unsupported
    // RequireRT call as a hardware-conditional CTest skip.
    GafimeSemanticRtRegionQueryDesc desc = query_desc(
        primary_bank.raw,
        paired_bank.raw,
        spec,
        GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT,
        kUnlimited
    );
    Query availability;
    uint64_t persistent = 0u;
    int status = gafime_gpu_semantic_region_query_create_rt_v1(
        &desc, &availability.raw, &persistent);
    if (status == GAFIME_STATUS_UNSUPPORTED_BACKEND) return 77;
    if (status != GAFIME_STATUS_OK || availability.raw == nullptr || persistent == 0u) {
        return fail("RequireRT compact query creation failed", status);
    }
    if (!caller_device_is(caller_device, "initial RequireRT query creation")) return 1;
    status = availability.close();
    if (status != GAFIME_STATUS_OK || !caller_device_is(caller_device, "initial query free")) {
        return fail("initial compact query free failed", status);
    }

    // The overlapping tiny shape cannot take direct first-hit, so an exact SM
    // query establishes the former generic descriptor/point/mask allocation
    // floor.  RequireRT must report a larger complete plan once its owned
    // GAS/IAS/SBT/parameter/build allocations are included.  An intermediate
    // budget specifically prevents an old generic-only estimate from admitting
    // a query that cannot own all of its RT resources.  A failing preflight may
    // report either this known full bound or a truthful conservative staged
    // minimum before it reaches full OptiX sizing; both must exceed the budget.
    const GafimeSemanticRtRegionQueryDesc generic_desc = query_desc(
        primary_bank.raw,
        paired_bank.raw,
        spec,
        GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE,
        kUnlimited
    );
    Query generic_only;
    uint64_t generic_only_persistent = 0u;
    status = gafime_gpu_semantic_region_query_create_rt_v1(
        &generic_desc, &generic_only.raw, &generic_only_persistent);
    if (status != GAFIME_STATUS_OK || generic_only.raw == nullptr ||
        generic_only_persistent == 0u) {
        return fail("generic-only compact persistent baseline creation failed", status);
    }
    status = generic_only.close();
    if (status != GAFIME_STATUS_OK ||
        !caller_device_is(caller_device, "generic-only persistent baseline free")) {
        return fail("generic-only compact persistent baseline free failed", status);
    }
    if (persistent <= generic_only_persistent ||
        persistent - generic_only_persistent <= 1u) {
        return fail("RequireRT persistent plan omitted its owned RT allocation footprint");
    }
    const uint64_t intermediate_persistent_budget = generic_only_persistent +
        (persistent - generic_only_persistent) / 2u;
    if (intermediate_persistent_budget <= generic_only_persistent ||
        intermediate_persistent_budget >= persistent) {
        return fail("invalid compact intermediate persistent-budget fixture");
    }
    desc.max_persistent_bytes = intermediate_persistent_budget;
    Query missing_rt_allocation_budget;
    uint64_t reported_required_persistent = 0u;
    status = gafime_gpu_semantic_region_query_create_rt_v1(
        &desc, &missing_rt_allocation_budget.raw, &reported_required_persistent);
    if (status != GAFIME_STATUS_OUT_OF_MEMORY || missing_rt_allocation_budget.raw != nullptr ||
        reported_required_persistent <= intermediate_persistent_budget ||
        reported_required_persistent > persistent) {
        return fail("RequireRT intermediate persistent preflight was not fail-closed", status);
    }
    if (!caller_device_is(caller_device, "full RT persistent-budget rejection")) return 1;

    // Persistent preflight must fail without creating an owned query.
    desc.max_persistent_bytes = 0u;
    Query too_small;
    uint64_t required_persistent = 0u;
    status = gafime_gpu_semantic_region_query_create_rt_v1(
        &desc, &too_small.raw, &required_persistent);
    if (status != GAFIME_STATUS_OUT_OF_MEMORY || too_small.raw != nullptr || required_persistent == 0u) {
        return fail("compact persistent-budget preflight was not exact", status);
    }
    if (!caller_device_is(caller_device, "persistent-budget rejection")) return 1;

    desc.max_persistent_bytes = kUnlimited;
    Query rt_query;
    persistent = 0u;
    status = gafime_gpu_semantic_region_query_create_rt_v1(&desc, &rt_query.raw, &persistent);
    if (status != GAFIME_STATUS_OK || rt_query.raw == nullptr || persistent == 0u) {
        return fail("strict RT compact query creation failed", status);
    }
    if (!caller_device_is(caller_device, "strict RT query creation")) return 1;

    // A spare physical output slot is not a dense root.  Until an all-channel
    // compact execute succeeds, this query has no retained membership from
    // which coverage could be materialized, and the fresh slot remains
    // unreadable.
    constexpr uint32_t kCoverageSlot = 2u;
    uint64_t coverage_peak = std::numeric_limits<uint64_t>::max();
    status = gafime_gpu_semantic_region_query_materialize_coverage_rt_v1(
        rt_query.raw, kCoverageSlot, kUnlimited, &coverage_peak);
    if (status != GAFIME_STATUS_INVALID_ARGUMENT ||
        !f32_slot_is_uninitialized(primary_bank.raw, 8u, kCoverageSlot)) {
        return fail("coverage was accepted before compact evidence completed", status);
    }
    if (!caller_device_is(caller_device, "pre-evidence coverage rejection")) return 1;

    // Create copied these descriptor arrays.  Mutating them after creation
    // must neither change the query nor require caller descriptor lifetime.
    spec.terms[0].input_slot = 99u;
    spec.terms[0].threshold_bits = f32_bits(-99.0f);
    spec.paired_term_slots[0] = 99u;
    spec.region_offsets[1] = 0u;
    spec.partition_offsets[1] = 0u;

    Execution execution;
    status = execute_query(
        rt_query.raw,
        3u,
        kAllStatistics,
        kAllFinalizers,
        labels,
        0u,
        &execution
    );
    if (status != GAFIME_STATUS_OUT_OF_MEMORY || execution.temporary_peak == 0u ||
        execution.table.count != 0u || !all_records_unchanged(execution)) {
        return fail("compact temporary-budget preflight was not atomic", status);
    }
    if (!caller_device_is(caller_device, "temporary-budget rejection")) return 1;

    status = execute_query(
        rt_query.raw,
        3u,
        kAllStatistics,
        kAllFinalizers,
        labels,
        kUnlimited,
        &execution
    );
    if (status != GAFIME_STATUS_OK || !records_match(execution, expected, true)) {
        return fail("strict RT compact counts/finalizers differ from the exact oracle", status);
    }
    if (!caller_device_is(caller_device, "strict RT query execute")) return 1;

    // A present but empty binary-label context is distinct from absent labels:
    // it contributes zero contingency counts and has insufficient support for
    // Gini, while the requested occupancy and paired evidence still commits.
    // This is the native counterpart of Rust's Some(empty) label set.
    const Labels empty_labels{};
    const auto empty_label_expected = exact_oracle(
        primary, paired, tiny_region_spec(), empty_labels);
    status = execute_query(
        rt_query.raw,
        3u,
        kAllStatistics,
        kAllFinalizers,
        empty_labels,
        kUnlimited,
        &execution
    );
    if (status != GAFIME_STATUS_OK ||
        execution.table.requested_statistic_mask != kAllStatistics ||
        execution.table.finalized_mask != kAllFinalizers ||
        !records_match(execution, empty_label_expected, false)) {
        return fail("empty label context did not commit exact compact evidence", status);
    }
    for (const auto& record : execution.records) {
        if (record.label_support != 0u || record.label_outside_0 != 0u ||
            record.label_outside_1 != 0u || record.label_inside_0 != 0u ||
            record.label_inside_1 != 0u ||
            record.labeled_gini_gain_state !=
                GAFIME_SEMANTIC_SCALAR_INSUFFICIENT_SUPPORT) {
            return fail("empty label context did not report insufficient Gini support");
        }
    }
    if (!caller_device_is(caller_device, "empty-label compact query execute")) return 1;

    // Coverage is an explicit selected-materialization operation after
    // evidence, not a pre-evidence dense-column adapter.  It counts submitted
    // primary regions and needs no execute-local staging in this bounded path.
    coverage_peak = std::numeric_limits<uint64_t>::max();
    status = gafime_gpu_semantic_region_query_materialize_coverage_rt_v1(
        rt_query.raw, kCoverageSlot, 0u, &coverage_peak);
    const std::vector<float> expected_coverage = {1.0f, 2.0f, 2.0f, 3.0f,
                                                   3.0f, 2.0f, 0.0f, 1.0f};
    std::vector<float> actual_coverage;
    if (status != GAFIME_STATUS_OK || coverage_peak != 0u ||
        !download_f32_slot(primary_bank.raw, 8u, kCoverageSlot, &actual_coverage) ||
        actual_coverage != expected_coverage) {
        return fail("post-evidence compact coverage differs from the exact rule count", status);
    }
    // The output bank tracks freshness.  A second selected materialization
    // cannot silently overwrite the first committed result.
    status = gafime_gpu_semantic_region_query_materialize_coverage_rt_v1(
        rt_query.raw, kCoverageSlot, kUnlimited, &coverage_peak);
    if (status != GAFIME_STATUS_INVALID_ARGUMENT ||
        !caller_device_is(caller_device, "post-evidence coverage materialization")) {
        return fail("coverage overwrote a committed slot", status);
    }

    // Labels are copied at execute, not cached by host pointer or query.  A
    // changed binary subset must change exact bins while descriptors stay fixed.
    Labels changed_labels{{0u, 1u, 3u, 4u, 5u, 7u}, {1u, 1u, 0u, 0u, 0u, 0u}};
    const RegionSpec original_spec = tiny_region_spec();
    const auto changed_expected = exact_oracle(primary, paired, original_spec, changed_labels);
    status = execute_query(
        rt_query.raw,
        3u,
        kAllStatistics,
        kAllFinalizers,
        changed_labels,
        kUnlimited,
        &execution
    );
    if (status != GAFIME_STATUS_OK || !records_match(execution, changed_expected, true)) {
        return fail("compact query cached stale labels", status);
    }

    // An invalid label context must fail the complete execute rather than emit
    // occupancy/paired rows and silently omit only the label channel.
    Labels duplicate_labels{{0u, 0u}, {0u, 1u}};
    status = execute_query(
        rt_query.raw,
        3u,
        kAllStatistics,
        kAllFinalizers,
        duplicate_labels,
        kUnlimited,
        &execution
    );
    if (status != GAFIME_STATUS_INVALID_ARGUMENT || execution.table.count != 0u ||
        !all_records_unchanged(execution)) {
        return fail("invalid labels produced a partial compact result", status);
    }

    status = rt_query.close();
    if (status != GAFIME_STATUS_OK || !caller_device_is(caller_device, "strict RT query free")) {
        return fail("strict RT compact query free failed", status);
    }

    // A compact query has no implicit dispatch policy.  Exactly one route is
    // selected at creation, so an empty flag set cannot defer RT failure into
    // a later SM fallback after resource admission.
    desc = query_desc(primary_bank.raw, paired_bank.raw, original_spec, 0u, kUnlimited);
    Query missing_route;
    status = gafime_gpu_semantic_region_query_create_rt_v1(
        &desc, &missing_route.raw, &persistent);
    if (status != GAFIME_STATUS_INVALID_ARGUMENT || missing_route.raw != nullptr) {
        return fail("compact query accepted an implicit route selector", status);
    }
    if (!caller_device_is(caller_device, "implicit-route rejection")) return 1;

    // Both SM comparators are exact compact routes rather than dense
    // membership adapters.  They must agree with strict RT and the host oracle
    // before the optional three-lane benchmark is permitted to time them.
    for (const uint32_t sm_flags : {
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE,
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM,
         }) {
        RegionSpec sm_spec = tiny_region_spec();
        desc = query_desc(
            primary_bank.raw,
            paired_bank.raw,
            sm_spec,
            sm_flags,
            kUnlimited
        );
        Query sm_query;
        status = gafime_gpu_semantic_region_query_create_rt_v1(&desc, &sm_query.raw, &persistent);
        if (status != GAFIME_STATUS_OK || sm_query.raw == nullptr) {
            return fail("forced-SM compact baseline creation failed", status);
        }
        status = execute_query(
            sm_query.raw, 3u, kAllStatistics, kAllFinalizers, labels, kUnlimited, &execution);
        if (status != GAFIME_STATUS_OK || !records_match(execution, expected, true)) {
            return fail("forced-SM compact baseline differs from exact oracle", status);
        }
        status = sm_query.close();
        if (status != GAFIME_STATUS_OK || !caller_device_is(caller_device, "forced-SM query free")) {
            return fail("forced-SM compact baseline free failed", status);
        }
    }

    // Dispatch selectors are mutually exclusive descriptor modes, never an
    // environment-selected geometry fallback.  The exhaustive SM lane is a
    // diagnostic comparator; FORCE_SM remains the structure-aware bin path.
    for (const uint32_t contradictory_flags : {
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT |
                 GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM,
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT |
                 GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE,
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM |
                 GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM_EXHAUSTIVE,
         }) {
        desc = query_desc(
            primary_bank.raw,
            paired_bank.raw,
            original_spec,
            contradictory_flags,
            kUnlimited
        );
        Query contradictory;
        status = gafime_gpu_semantic_region_query_create_rt_v1(
            &desc, &contradictory.raw, &persistent);
        if (status != GAFIME_STATUS_INVALID_ARGUMENT || contradictory.raw != nullptr) {
            return fail("contradictory compact dispatch flags were accepted", status);
        }
    }

    const int paired_free = paired_bank.close();
    const int primary_free = primary_bank.close();
    if (paired_free != GAFIME_STATUS_OK || primary_free != GAFIME_STATUS_OK) {
        return fail("compact semantic-bank free failed", paired_free != GAFIME_STATUS_OK ? paired_free : primary_free);
    }
    return 0;
}

// A final geometry fixture has no caller-visible geometry selector.  It proves
// that a shape eligible for a native internal lowering still returns the same
// exact evidence through RequireRT, exhaustive SM, and query-binned SM.
int exercise_exact_three_lane_case(
    const char* context,
    const Columns& primary,
    const Columns& paired,
    const RegionSpec& spec,
    const Labels& labels,
    int caller_device
) {
    if (context == nullptr || primary.empty() || paired.empty() ||
        primary.front().empty() || paired.front().size() != primary.front().size() ||
        spec.region_offsets.size() < 2u) {
        return fail("invalid final compact geometry fixture");
    }
    const uint64_t rows = primary.front().size();
    for (const auto& column : primary) {
        if (column.size() != rows) return fail("primary geometry fixture columns were ragged");
    }
    for (const auto& column : paired) {
        if (column.size() != rows) return fail("paired geometry fixture columns were ragged");
    }
    const auto expected = exact_oracle(primary, paired, spec, labels);
    const uint32_t region_count = static_cast<uint32_t>(expected.size());
    if (region_count == 0u || primary.size() > std::numeric_limits<uint32_t>::max() ||
        paired.size() > std::numeric_limits<uint32_t>::max()) {
        return fail("final compact geometry fixture shape overflow");
    }

    Bank primary_bank;
    Bank paired_bank;
    if (!allocate_uploaded_f32_bank(
            column_major(primary),
            rows,
            static_cast<uint32_t>(primary.size()),
            static_cast<uint32_t>(primary.size() + kBenchmarkRoutes.size()),
            &primary_bank) ||
        !allocate_uploaded_f32_bank(
            column_major(paired),
            rows,
            static_cast<uint32_t>(paired.size()),
            static_cast<uint32_t>(paired.size()),
            &paired_bank)) {
        return fail("final compact geometry semantic-bank setup failed");
    }

    std::vector<GafimeSemanticRtRegionExactStats> baseline_records;
    uint32_t weighted_slot = static_cast<uint32_t>(primary.size());
    for (const BenchmarkRoute route : kBenchmarkRoutes) {
        const GafimeSemanticRtRegionQueryDesc desc = query_desc(
            primary_bank.raw, paired_bank.raw, spec, route.flags, kUnlimited);
        Query query;
        uint64_t persistent = 0u;
        int status = gafime_gpu_semantic_region_query_create_rt_v1(
            &desc, &query.raw, &persistent);
        if (status != GAFIME_STATUS_OK || query.raw == nullptr || persistent == 0u) {
            return fail(context, status);
        }
        Execution execution;
        status = execute_query(
            query.raw,
            region_count,
            kAllStatistics,
            kAllFinalizers,
            labels,
            kUnlimited,
            &execution
        );
        if (status != GAFIME_STATUS_OK ||
            execution.table.requested_statistic_mask != kAllStatistics ||
            execution.table.finalized_mask != kAllFinalizers ||
            !records_match(execution, expected, true)) {
            return fail(context, status);
        }
        if (baseline_records.empty()) {
            baseline_records = execution.records;
        } else if (!same_native_record_vectors(baseline_records, execution.records)) {
            return fail("final compact geometry lanes disagreed in exact records/finalizers");
        }
        std::vector<float> weights(region_count);
        for (uint32_t r = 0; r < region_count; ++r) {
            weights[r] = (r % 2u == 0u ? 1.0f : -1.0f) * static_cast<float>(r + 1u);
        }
        uint64_t weighted_peak = 0u;
        status = gafime_gpu_semantic_region_query_materialize_weighted_sum_rt_v1(
            query.raw, weighted_slot, weights.data(), region_count, kUnlimited, &weighted_peak);
        std::vector<float> actual;
        if (status != GAFIME_STATUS_OK ||
            !download_f32_slot(primary_bank.raw, rows, weighted_slot, &actual)) {
            return fail("geometry weighted materialization failed", status);
        }
        for (uint64_t row = 0u; row < rows; ++row) {
            float expected_weighted = 0.0f;
            for (uint32_t r = 0u; r < region_count; ++r) {
                bool inside = true;
                for (uint32_t t = spec.region_offsets[r]; t < spec.region_offsets[r + 1u]; ++t) {
                    inside = inside && holds(primary[spec.terms[t].input_slot][row], spec.terms[t]);
                }
                if (inside) expected_weighted += weights[r];
            }
            if (f32_bits(actual[row]) != f32_bits(expected_weighted)) {
                return fail("geometry weighted ordinal/canonical sum disagrees with oracle");
            }
        }
        ++weighted_slot;
        status = query.close();
        if (status != GAFIME_STATUS_OK || !caller_device_is(caller_device, context)) {
            return fail("final compact geometry query free failed", status);
        }
    }
    const int paired_free = paired_bank.close();
    const int primary_free = primary_bank.close();
    if (paired_free != GAFIME_STATUS_OK || primary_free != GAFIME_STATUS_OK) {
        return fail(
            "final compact geometry semantic-bank free failed",
            paired_free != GAFIME_STATUS_OK ? paired_free : primary_free
        );
    }
    return 0;
}

int exercise_weighted_regions(int caller_device) {
    const Columns primary = {
        {-1.0f, -0.0f, 0.0f, 0.5f, 1.0f, 2.0f, -2.0f, 3.0f},
        {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 2.0f},
    };
    const RegionSpec spec = tiny_region_spec();
    for (const uint32_t flags : {
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT,
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM}) {
        Bank bank;
        if (!allocate_uploaded_f32_bank(column_major(primary), 8u, 2u, 5u, &bank)) {
            return fail("weighted region bank allocation failed");
        }
        const auto desc = query_desc(bank.raw, nullptr, spec, flags, kUnlimited);
        Query query;
        uint64_t persistent = 0u;
        int status = gafime_gpu_semantic_region_query_create_rt_v1(
            &desc, &query.raw, &persistent);
        if (status != GAFIME_STATUS_OK) return fail("weighted region query failed", status);
        const std::vector<float> weights{2.0f, -3.0f, 5.0f};
        uint64_t peak = 0u;
        auto materialize = [&](uint32_t slot, const float* values, uint64_t count, uint64_t budget) {
            return gafime_gpu_semantic_region_query_materialize_weighted_sum_rt_v1(
                query.raw, slot, values, count, budget, &peak);
        };
        if (materialize(2u, weights.data(), 3u, kUnlimited) == GAFIME_STATUS_OK ||
            !f32_slot_is_uninitialized(bank.raw, 8u, 2u)) {
            return fail("weighted region accepted materialization before execute");
        }
        Execution execution;
        status = execute_query(query.raw, 3u, GAFIME_SEMANTIC_RT_REGION_STAT_BINARY_OCCUPANCY,
                               0u, {}, kUnlimited, &execution);
        if (status != GAFIME_STATUS_OK) return fail("weighted region execute failed", status);
        const std::vector<float> invalid{2.0f, std::numeric_limits<float>::infinity(), 5.0f};
        if (materialize(2u, weights.data(), 1u, kUnlimited) != GAFIME_STATUS_INVALID_ARGUMENT ||
            peak != 0u ||
            materialize(2u, weights.data(), 65u, kUnlimited) != GAFIME_STATUS_INVALID_ARGUMENT ||
            peak != 0u) {
            return fail("weighted count envelope did not reject before reading weights");
        }
        if (materialize(2u, nullptr, 3u, kUnlimited) == GAFIME_STATUS_OK ||
            materialize(2u, weights.data(), 2u, kUnlimited) == GAFIME_STATUS_OK ||
            materialize(2u, invalid.data(), 3u, kUnlimited) == GAFIME_STATUS_OK ||
            materialize(2u, weights.data(), 3u, 1u) == GAFIME_STATUS_OK ||
            !f32_slot_is_uninitialized(bank.raw, 8u, 2u)) {
            return fail("weighted region validation/budget failure committed output");
        }
        status = materialize(2u, weights.data(), 3u, kUnlimited);
        if (status != GAFIME_STATUS_OK || peak != 4u * sizeof(float)) {
            return fail("weighted region materialization/budget accounting failed", status);
        }
        std::vector<float> result;
        if (!download_f32_slot(bank.raw, 8u, 2u, &result)) return fail("weighted download failed");
        const std::vector<float> expected{5.0f, 2.0f, 2.0f, 4.0f, 4.0f, -1.0f, 0.0f, 2.0f};
        for (size_t row = 0; row < expected.size(); ++row) {
            if (f32_bits(result[row]) != f32_bits(expected[row])) {
                return fail("weighted region sum differs bit-for-bit from hand oracle");
            }
        }
        if (materialize(2u, weights.data(), 3u, kUnlimited) == GAFIME_STATUS_OK) {
            return fail("weighted region overwrote a committed slot");
        }
        const std::vector<float> overflow(3u, std::numeric_limits<float>::max());
        if (materialize(3u, overflow.data(), 3u, kUnlimited) == GAFIME_STATUS_OK ||
            !f32_slot_is_uninitialized(bank.raw, 8u, 3u) ||
            materialize(3u, weights.data(), 3u, kUnlimited) != GAFIME_STATUS_OK) {
            return fail("weighted overflow did not fail closed/retry cleanly");
        }
        const float tiny = std::numeric_limits<float>::denorm_min();
        const std::vector<float> subnormal{tiny, tiny, tiny};
        if (materialize(4u, subnormal.data(), 3u, kUnlimited) != GAFIME_STATUS_OK ||
            !download_f32_slot(bank.raw, 8u, 4u, &result) ||
            f32_bits(result[3]) != f32_bits(tiny + tiny + tiny)) {
            return fail("weighted finite subnormal arithmetic was narrowed/flushed");
        }
        int current_device = -1;
        if (cudaGetDevice(&current_device) != cudaSuccess || current_device != caller_device) {
            return fail("weighted materialization changed caller device");
        }
    }
    return 0;
}

int exercise_triangle_eligible_partition_case(int caller_device) {
    // Four finite, safely wide 2D cells.  They satisfy the native triangle
    // eligibility preconditions, but this fixture observes only exact output;
    // it never assumes which internal geometry implementation was selected.
    const Columns primary = {
        {0.25f, 1.25f, 0.25f, 1.25f, 0.75f, 1.75f, 0.75f, 1.75f},
        {0.25f, 0.25f, 1.25f, 1.25f, 0.75f, 0.75f, 1.75f, 1.75f},
    };
    const Columns paired = {
        {29.0f, 29.0f, 29.0f, 29.0f, 29.0f, 29.0f, 29.0f, 29.0f},
        {0.25f, 0.25f, 1.25f, 1.25f, 1.75f, 0.75f, 1.75f, 0.75f},
        {0.25f, 0.25f, 1.25f, 1.25f, 0.75f, 0.75f, 1.75f, 1.75f},
    };
    const Labels labels{
        {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u},
        {0u, 1u, 0u, 1u, 1u, 0u, 1u, 0u},
    };
    RegionSpec spec{};
    for (const float lo_y : {0.0f, 1.0f}) {
        for (const float lo_x : {0.0f, 1.0f}) {
            spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
            spec.terms.push_back({0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(lo_x)});
            spec.terms.push_back({0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(lo_x + 1.0f)});
            spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(lo_y)});
            spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(lo_y + 1.0f)});
            spec.paired_term_slots.insert(
                spec.paired_term_slots.end(), {1u, 1u, 2u, 2u});
        }
    }
    spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
    spec.partition_offsets = {0u, 4u};
    return exercise_exact_three_lane_case(
        "triangle-eligible compact partition differs from exact evidence",
        primary,
        paired,
        spec,
        labels,
        caller_device
    );
}

int exercise_multi_axis_group_case(int caller_device) {
    // Partition 0 has the (x,y) axes and partition 1 has the (y,z) axes.
    // Separate partition ranges prevent the planner from merging them into a
    // synthetic 3D group; paired slots deliberately use a different layout.
    const Columns primary = {
        {0.25f, 0.75f, 1.25f, 1.75f, 0.25f, 1.25f, 0.25f, 1.25f, 0.75f, 1.75f, 0.25f, 1.25f},
        {0.25f, 0.75f, 0.25f, 0.75f, 1.25f, 1.75f, 0.25f, 0.25f, 1.25f, 1.75f, 0.75f, 0.75f},
        {0.25f, 0.75f, 0.25f, 0.75f, 0.25f, 0.75f, 1.25f, 1.25f, 0.25f, 0.75f, 0.25f, 0.75f},
    };
    const Columns paired = {
        {31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f, 31.0f},
        {0.75f, 0.25f, 0.75f, 0.25f, 0.75f, 0.25f, 0.25f, 0.25f, 0.75f, 0.25f, 0.75f, 0.25f},
        {0.25f, 1.75f, 1.25f, 0.75f, 1.25f, 0.25f, 0.75f, 1.75f, 0.25f, 0.75f, 1.25f, 0.25f},
        {0.25f, 0.75f, 0.75f, 0.25f, 1.75f, 1.25f, 0.75f, 0.75f, 1.75f, 1.25f, 0.25f, 0.75f},
    };
    const Labels labels{
        {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u},
        {0u, 1u, 0u, 1u, 1u, 0u, 1u, 0u, 0u, 1u, 1u, 0u},
    };
    RegionSpec spec{};
    // (x,y): [0,1] and (1,2] x-cells, both with y in (0,1].
    for (const float lo_x : {0.0f, 1.0f}) {
        spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
        spec.terms.push_back({0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(lo_x)});
        spec.terms.push_back({0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(lo_x + 1.0f)});
        spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(0.0f)});
        spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(1.0f)});
        spec.paired_term_slots.insert(
            spec.paired_term_slots.end(), {2u, 2u, 3u, 3u});
    }
    // (y,z): [0,1] and (1,2] y-cells, both with z in (0,1].
    for (const float lo_y : {0.0f, 1.0f}) {
        spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
        spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(lo_y)});
        spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(lo_y + 1.0f)});
        spec.terms.push_back({2u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(0.0f)});
        spec.terms.push_back({2u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(1.0f)});
        spec.paired_term_slots.insert(
            spec.paired_term_slots.end(), {3u, 3u, 1u, 1u});
    }
    spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
    spec.partition_offsets = {0u, 2u, 4u};
    return exercise_exact_three_lane_case(
        "multi-axis-group compact query differs from exact evidence",
        primary,
        paired,
        spec,
        labels,
        caller_device
    );
}

int exercise_generated_binned_case(uint64_t rows, bool permute_paired, int caller_device) {
    Columns primary(2);
    Columns paired(3);
    primary[0].reserve(rows);
    primary[1].reserve(rows);
    paired[0].assign(rows, 19.0f);
    paired[1].reserve(rows);
    paired[2].reserve(rows);
    Labels labels;
    for (uint64_t row = 0; row < rows; ++row) {
        const float x = static_cast<float>((row * 17u + 3u) % 59u) / 8.0f - 3.5f;
        const float paired_x = static_cast<float>((row * 29u + (permute_paired ? 11u : 3u)) % 61u) /
                8.0f -
            3.5f;
        const float y = static_cast<float>((row * 7u + 1u) % 23u) / 4.0f - 2.0f;
        primary[0].push_back(x);
        primary[1].push_back(y);
        paired[1].push_back(paired_x);
        paired[2].push_back(y);
        if (row % 3u != 2u) {
            labels.rows.push_back(row);
            labels.values.push_back(static_cast<uint8_t>((row * 5u + 1u) & 1u));
        }
    }
    RegionSpec spec{};
    const std::array<float, 5> thresholds = {-2.5f, -1.0f, 0.0f, 1.25f, 2.5f};
    for (size_t index = 0; index < thresholds.size(); ++index) {
        spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
        spec.terms.push_back({0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(thresholds[index] - 0.75f)});
        spec.terms.push_back({0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(thresholds[index] + 0.75f)});
        if ((index & 1u) != 0u) {
            spec.terms.push_back({1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(1.0f)});
        }
    }
    spec.region_offsets.push_back(static_cast<uint32_t>(spec.terms.size()));
    spec.paired_term_slots.reserve(spec.terms.size());
    for (const auto& term : spec.terms) {
        spec.paired_term_slots.push_back(term.input_slot == 0u ? 1u : 2u);
    }
    // Two contiguous logical groups ensure the binned comparator preserves
    // submitted ordinals even if it internally groups spatial work.
    spec.partition_offsets = {0u, 2u, static_cast<uint32_t>(thresholds.size())};
    const auto expected = exact_oracle(primary, paired, spec, labels);

    Bank primary_bank;
    Bank paired_bank;
    if (!allocate_uploaded_f32_bank(column_major(primary), rows, 2u, 2u, &primary_bank) ||
        !allocate_uploaded_f32_bank(column_major(paired), rows, 3u, 3u, &paired_bank)) {
        return fail("generated compact semantic-bank setup failed");
    }
    const GafimeSemanticRtRegionQueryDesc desc = query_desc(
        primary_bank.raw,
        paired_bank.raw,
        spec,
        GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM,
        kUnlimited
    );
    Query query;
    uint64_t persistent = 0u;
    int status = gafime_gpu_semantic_region_query_create_rt_v1(&desc, &query.raw, &persistent);
    if (status != GAFIME_STATUS_OK || query.raw == nullptr || persistent == 0u) {
        return fail("generated binned compact query creation failed", status);
    }
    Execution execution;
    status = execute_query(
        query.raw,
        static_cast<uint32_t>(expected.size()),
        kAllStatistics,
        kAllFinalizers,
        labels,
        kUnlimited,
        &execution
    );
    if (status != GAFIME_STATUS_OK || !records_match(execution, expected, true)) {
        return fail("generated binned compact counts differ from exact oracle", status);
    }
    status = query.close();
    if (status != GAFIME_STATUS_OK || !caller_device_is(caller_device, "generated query free")) {
        return fail("generated compact query free failed", status);
    }
    const int paired_free = paired_bank.close();
    const int primary_free = primary_bank.close();
    if (paired_free != GAFIME_STATUS_OK || primary_free != GAFIME_STATUS_OK) {
        return fail("generated compact semantic-bank free failed", paired_free != GAFIME_STATUS_OK ? paired_free : primary_free);
    }
    return 0;
}

int exercise_nonoverlap_partition_case(int caller_device) {
    const Columns primary = {
        {-3.0f, -2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f, 3.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    };
    const Columns paired = {
        {23.0f, 23.0f, 23.0f, 23.0f, 23.0f, 23.0f, 23.0f, 23.0f},
        {-3.0f, -2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f, 3.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    };
    Labels labels{{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}, {0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u}};
    RegionSpec spec{};
    spec.terms = {
        {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(0.0f)},
        {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(0.0f)},
    };
    spec.paired_term_slots = {1u, 1u};
    spec.region_offsets = {0u, 1u, 2u};
    spec.partition_offsets = {0u, 2u};
    const auto expected = exact_oracle(primary, paired, spec, labels);

    Bank primary_bank;
    Bank paired_bank;
    if (!allocate_uploaded_f32_bank(column_major(primary), 8u, 2u, 2u, &primary_bank) ||
        !allocate_uploaded_f32_bank(column_major(paired), 8u, 3u, 3u, &paired_bank)) {
        return fail("non-overlap compact semantic-bank setup failed");
    }

    for (const uint32_t flags : {
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_REQUIRE_RT,
             GAFIME_SEMANTIC_RT_REGION_QUERY_FLAG_FORCE_SM,
         }) {
        const GafimeSemanticRtRegionQueryDesc desc = query_desc(
            primary_bank.raw, paired_bank.raw, spec, flags, kUnlimited);
        Query query;
        uint64_t persistent = 0u;
        int status = gafime_gpu_semantic_region_query_create_rt_v1(&desc, &query.raw, &persistent);
        if (status != GAFIME_STATUS_OK || query.raw == nullptr) {
            return fail("non-overlap compact query creation failed", status);
        }
        Execution execution;
        status = execute_query(
            query.raw,
            2u,
            kAllStatistics,
            kAllFinalizers,
            labels,
            kUnlimited,
            &execution
        );
        if (status != GAFIME_STATUS_OK || !records_match(execution, expected, true)) {
            return fail("non-overlap RT/SM compact query differs from exact oracle", status);
        }
        status = query.close();
        if (status != GAFIME_STATUS_OK || !caller_device_is(caller_device, "non-overlap query free")) {
            return fail("non-overlap compact query free failed", status);
        }
    }
    const int paired_free = paired_bank.close();
    const int primary_free = primary_bank.close();
    if (paired_free != GAFIME_STATUS_OK || primary_free != GAFIME_STATUS_OK) {
        return fail("non-overlap compact semantic-bank free failed", paired_free != GAFIME_STATUS_OK ? paired_free : primary_free);
    }
    return 0;
}

// This optional diagnostic has a deliberately narrow shape: every region is
// one finite 2D cell, every cell receives 64 rows, the paired view changes
// memberships without changing physical predicate layout, and labels remain a
// proper binary subset.  It is not part of CTest and never emits a speedup.
struct BenchmarkFixture {
    Columns primary;
    Columns paired;
    Labels labels;
    RegionSpec spec;
    uint64_t rows = 0u;
    uint32_t regions = 0u;
    uint32_t grid_side = 0u;
};

bool make_benchmark_fixture(uint64_t rows, uint32_t grid_side, BenchmarkFixture* fixture) {
    if (fixture == nullptr || grid_side == 0u ||
        grid_side > std::numeric_limits<uint32_t>::max() / grid_side) {
        return false;
    }
    const uint32_t regions = grid_side * grid_side;
    if (rows == 0u || rows % regions != 0u || rows / regions != 64u) return false;

    BenchmarkFixture output{};
    output.rows = rows;
    output.regions = regions;
    output.grid_side = grid_side;
    output.primary.resize(2u);
    output.paired.resize(3u);
    for (auto& column : output.primary) column.reserve(rows);
    for (auto& column : output.paired) column.reserve(rows);
    output.labels.rows.reserve(rows - rows / 3u);
    output.labels.values.reserve(rows - rows / 3u);

    constexpr uint32_t kSamplesPerAxis = 8u;
    for (uint32_t region = 0u; region < regions; ++region) {
        const uint32_t primary_x_cell = region % grid_side;
        const uint32_t primary_y_cell = region / grid_side;
        for (uint32_t sample = 0u; sample < 64u; ++sample) {
            const uint32_t sample_x = sample % kSamplesPerAxis;
            const uint32_t sample_y = sample / kSamplesPerAxis;
            const float local_x = (static_cast<float>(sample_x) + 0.5f) /
                static_cast<float>(kSamplesPerAxis);
            const float local_y = (static_cast<float>(sample_y) + 0.5f) /
                static_cast<float>(kSamplesPerAxis);
            const float primary_x = (static_cast<float>(primary_x_cell) + local_x) /
                static_cast<float>(grid_side);
            const float primary_y = (static_cast<float>(primary_y_cell) + local_y) /
                static_cast<float>(grid_side);
            // Move subsets of the paired rows into adjacent cells.  This
            // preserves a finite 2D grid while exercising all paired bins.
            const uint32_t paired_x_cell =
                (primary_x_cell + ((sample & 1u) == 0u ? 0u : 1u)) % grid_side;
            const uint32_t paired_y_cell =
                (primary_y_cell + ((sample & 2u) == 0u ? 0u : 1u)) % grid_side;
            const float paired_x = (static_cast<float>(paired_x_cell) + local_x) /
                static_cast<float>(grid_side);
            const float paired_y = (static_cast<float>(paired_y_cell) + local_y) /
                static_cast<float>(grid_side);
            const uint64_t row = static_cast<uint64_t>(region) * 64u + sample;
            output.primary[0].push_back(primary_x);
            output.primary[1].push_back(primary_y);
            output.paired[0].push_back(17.0f);
            output.paired[1].push_back(paired_x);
            output.paired[2].push_back(paired_y);
            if (row % 3u != 0u) {
                output.labels.rows.push_back(row);
                output.labels.values.push_back(static_cast<uint8_t>(
                    (sample + primary_x_cell + primary_y_cell) & 1u));
            }
        }
    }

    output.spec.region_offsets.reserve(static_cast<size_t>(regions) + 1u);
    output.spec.paired_term_slots.reserve(static_cast<size_t>(regions) * 4u);
    output.spec.terms.reserve(static_cast<size_t>(regions) * 4u);
    for (uint32_t cell_y = 0u; cell_y < grid_side; ++cell_y) {
        for (uint32_t cell_x = 0u; cell_x < grid_side; ++cell_x) {
            output.spec.region_offsets.push_back(static_cast<uint32_t>(output.spec.terms.size()));
            const float lo_x = static_cast<float>(cell_x) / static_cast<float>(grid_side);
            const float hi_x = static_cast<float>(cell_x + 1u) / static_cast<float>(grid_side);
            const float lo_y = static_cast<float>(cell_y) / static_cast<float>(grid_side);
            const float hi_y = static_cast<float>(cell_y + 1u) / static_cast<float>(grid_side);
            output.spec.terms.push_back(
                {0u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(lo_x)});
            output.spec.terms.push_back(
                {0u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(hi_x)});
            output.spec.terms.push_back(
                {1u, GAFIME_SEMANTIC_REGION_GREATER_THAN, f32_bits(lo_y)});
            output.spec.terms.push_back(
                {1u, GAFIME_SEMANTIC_REGION_LESS_EQUAL, f32_bits(hi_y)});
            output.spec.paired_term_slots.insert(
                output.spec.paired_term_slots.end(), {1u, 1u, 2u, 2u});
        }
    }
    output.spec.region_offsets.push_back(static_cast<uint32_t>(output.spec.terms.size()));
    // One shared axis-compatible group avoids conflating the first compact
    // binned diagnostic with multi-group packing overhead.
    output.spec.partition_offsets = {0u, regions};
    *fixture = std::move(output);
    return true;
}

uint64_t elapsed_nanoseconds(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end
) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

uint64_t median_nanoseconds(const std::vector<uint64_t>& samples) {
    if (samples.empty()) return 0u;
    std::vector<uint64_t> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2u];
}

void print_raw_samples(const std::vector<uint64_t>& samples) {
    std::fputc('[', stdout);
    for (size_t index = 0u; index < samples.size(); ++index) {
        if (index != 0u) std::fputc(',', stdout);
        std::fprintf(stdout, "%llu", static_cast<unsigned long long>(samples[index]));
    }
    std::fputc(']', stdout);
}

int create_compact_query(
    const GafimeSemanticRtRegionQueryDesc& desc,
    Query* query,
    uint64_t* persistent_bytes
) {
    if (query == nullptr || persistent_bytes == nullptr || query->raw != nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    *persistent_bytes = 0u;
    return gafime_gpu_semantic_region_query_create_rt_v1(&desc, &query->raw, persistent_bytes);
}

bool prepared_execution_completed(
    const PreparedExecution& execution,
    uint32_t region_count,
    uint64_t temporary_budget
) {
    return execution.temporary_peak <= temporary_budget &&
        execution.table.requested_statistic_mask == kAllStatistics &&
        execution.table.finalized_mask == kAllFinalizers &&
        execution.table.count == region_count &&
        prepared_finalizers_are_valid(execution, region_count);
}

int benchmark_cold_create(
    const std::array<GafimeSemanticRtRegionQueryDesc, kBenchmarkRoutes.size()>& descs,
    std::array<std::vector<uint64_t>, kBenchmarkRoutes.size()>* samples_out,
    int caller_device
) {
    if (samples_out == nullptr) return fail("benchmark cold-create sample output was null");
    for (auto& samples : *samples_out) samples.clear();
    for (uint32_t warmup = 0u; warmup < kBenchmarkWarmups; ++warmup) {
        for (size_t offset = 0u; offset < kBenchmarkRoutes.size(); ++offset) {
            const size_t route = (warmup + offset) % kBenchmarkRoutes.size();
            Query query;
            uint64_t persistent = 0u;
            const int status = create_compact_query(descs[route], &query, &persistent);
            if (status != GAFIME_STATUS_OK || query.raw == nullptr || persistent == 0u ||
                persistent > kBenchmarkPersistentBudget || query.close() != GAFIME_STATUS_OK) {
                return fail("benchmark cold-create warmup failed", status);
            }
        }
    }
    if (!caller_device_is(caller_device, "benchmark cold-create warmups")) return 1;
    for (uint32_t sample = 0u; sample < kBenchmarkSamples; ++sample) {
        for (size_t offset = 0u; offset < kBenchmarkRoutes.size(); ++offset) {
            const size_t route = (sample + offset) % kBenchmarkRoutes.size();
            Query query;
            uint64_t persistent = 0u;
            const auto begin = std::chrono::steady_clock::now();
            const int status = create_compact_query(descs[route], &query, &persistent);
            const auto end = std::chrono::steady_clock::now();
            if (status != GAFIME_STATUS_OK || query.raw == nullptr || persistent == 0u ||
                persistent > kBenchmarkPersistentBudget) {
                return fail("benchmark cold-create sample failed", status);
            }
            (*samples_out)[route].push_back(elapsed_nanoseconds(begin, end));
            const int close_status = query.close();
            if (close_status != GAFIME_STATUS_OK ||
                !caller_device_is(caller_device, "benchmark cold-query free")) {
                return fail("benchmark cold-create query free failed", close_status);
            }
        }
    }
    return 0;
}

int benchmark_execute_case(uint64_t rows, uint32_t grid_side, int caller_device) {
    BenchmarkFixture fixture{};
    if (!make_benchmark_fixture(rows, grid_side, &fixture)) {
        return fail("benchmark fixture construction failed");
    }
    Bank primary_bank;
    Bank paired_bank;
    if (!allocate_uploaded_f32_bank(
            column_major(fixture.primary), rows, 2u, 2u, &primary_bank) ||
        !allocate_uploaded_f32_bank(
            column_major(fixture.paired), rows, 3u, 3u, &paired_bank)) {
        return fail("benchmark semantic-bank setup failed");
    }

    std::array<GafimeSemanticRtRegionQueryDesc, kBenchmarkRoutes.size()> descs{};
    for (size_t route = 0u; route < kBenchmarkRoutes.size(); ++route) {
        descs[route] = query_desc(
            primary_bank.raw,
            paired_bank.raw,
            fixture.spec,
            kBenchmarkRoutes[route].flags,
            kBenchmarkPersistentBudget
        );
    }
    std::array<std::vector<uint64_t>, kBenchmarkRoutes.size()> cold_samples{};
    if (const int cold = benchmark_cold_create(descs, &cold_samples, caller_device); cold != 0) {
        return cold;
    }

    std::array<Query, kBenchmarkRoutes.size()> queries{};
    std::array<uint64_t, kBenchmarkRoutes.size()> persistent_bytes{};
    std::array<PreparedExecution, kBenchmarkRoutes.size()> executions{};
    for (size_t route = 0u; route < kBenchmarkRoutes.size(); ++route) {
        const int status = create_compact_query(descs[route], &queries[route], &persistent_bytes[route]);
        if (status != GAFIME_STATUS_OK || queries[route].raw == nullptr ||
            persistent_bytes[route] == 0u || persistent_bytes[route] > kBenchmarkPersistentBudget ||
            !prepare_execution(
                fixture.regions,
                kAllStatistics,
                kAllFinalizers,
                fixture.labels,
                kBenchmarkTemporaryBudget,
                &executions[route])) {
            return fail("benchmark resident query setup failed", status);
        }
        const int execute_status = execute_prepared(queries[route].raw, &executions[route]);
        if (execute_status != GAFIME_STATUS_OK ||
            !prepared_execution_completed(
                executions[route], fixture.regions, kBenchmarkTemporaryBudget)) {
            return fail("benchmark pre-timing execute/finalizer validation failed", execute_status);
        }
    }
    for (size_t route = 1u; route < kBenchmarkRoutes.size(); ++route) {
        if (!same_completed_native_records(executions[0u], executions[route], fixture.regions)) {
            return fail("benchmark routes differ in exact compact evidence before timing");
        }
    }

    for (uint32_t warmup = 0u; warmup < kBenchmarkWarmups; ++warmup) {
        for (size_t offset = 0u; offset < kBenchmarkRoutes.size(); ++offset) {
            const size_t route = (warmup + offset) % kBenchmarkRoutes.size();
            const int status = execute_prepared(queries[route].raw, &executions[route]);
            if (status != GAFIME_STATUS_OK ||
                !prepared_execution_completed(
                    executions[route], fixture.regions, kBenchmarkTemporaryBudget)) {
                return fail("benchmark warm execute failed", status);
            }
        }
    }

    std::array<std::vector<uint64_t>, kBenchmarkRoutes.size()> execute_samples{};
    for (uint32_t sample = 0u; sample < kBenchmarkSamples; ++sample) {
        for (size_t offset = 0u; offset < kBenchmarkRoutes.size(); ++offset) {
            const size_t route = (sample + offset) % kBenchmarkRoutes.size();
            reset_prepared_execution(&executions[route]);
            const auto begin = std::chrono::steady_clock::now();
            const int status = gafime_gpu_semantic_region_query_execute_rt_v1(
                queries[route].raw,
                &executions[route].desc,
                &executions[route].table,
                &executions[route].temporary_peak
            );
            const auto end = std::chrono::steady_clock::now();
            if (status != GAFIME_STATUS_OK ||
                !prepared_execution_completed(
                    executions[route], fixture.regions, kBenchmarkTemporaryBudget)) {
                return fail("benchmark timed execute failed", status);
            }
            execute_samples[route].push_back(elapsed_nanoseconds(begin, end));
        }
    }
    if (!caller_device_is(caller_device, "benchmark timed executes")) return 1;

    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    for (size_t route = 0u; route < kBenchmarkRoutes.size(); ++route) {
        const int status = queries[route].close();
        if (status != GAFIME_STATUS_OK ||
            !caller_device_is(caller_device, "benchmark resident query free")) {
            return fail("benchmark resident query free failed", status);
        }
    }
    const int paired_free = paired_bank.close();
    const int primary_free = primary_bank.close();
    if (paired_free != GAFIME_STATUS_OK || primary_free != GAFIME_STATUS_OK) {
        return fail("benchmark semantic-bank free failed", paired_free != GAFIME_STATUS_OK ? paired_free : primary_free);
    }

    std::printf(
        "RT_COMPACT_EVIDENCE cell_rows=%llu cell_regions=%u dims=2 paired=1 partial_binary_labels=%llu "
        "statistics=occupancy,paired,labeled finalizers=occupancy,agreement,iou,gini "
        "persistent_budget=%llu temporary_budget=%llu warmups=%u samples_per_lane=%u "
        "live_query_free_bytes=%llu live_query_total_bytes=%llu\\n",
        static_cast<unsigned long long>(rows),
        fixture.regions,
        static_cast<unsigned long long>(fixture.labels.rows.size()),
        static_cast<unsigned long long>(kBenchmarkPersistentBudget),
        static_cast<unsigned long long>(kBenchmarkTemporaryBudget),
        kBenchmarkWarmups,
        kBenchmarkSamples,
        static_cast<unsigned long long>(memory_status == cudaSuccess ? free_bytes : 0u),
        static_cast<unsigned long long>(memory_status == cudaSuccess ? total_bytes : 0u)
    );
    for (size_t route = 0u; route < kBenchmarkRoutes.size(); ++route) {
        std::printf(
            "RT_COMPACT_EVIDENCE lane=%s phase=cold_create resident_inputs=1 persistent_bytes=%llu "
            "median_ns=%llu raw_ns=",
            kBenchmarkRoutes[route].name,
            static_cast<unsigned long long>(persistent_bytes[route]),
            static_cast<unsigned long long>(median_nanoseconds(cold_samples[route]))
        );
        print_raw_samples(cold_samples[route]);
        std::printf("\\n");
        std::printf(
            "RT_COMPACT_EVIDENCE lane=%s phase=warm_execute temporary_peak_bytes=%llu "
            "median_ns=%llu raw_ns=",
            kBenchmarkRoutes[route].name,
            static_cast<unsigned long long>(executions[route].temporary_peak),
            static_cast<unsigned long long>(median_nanoseconds(execute_samples[route]))
        );
        print_raw_samples(execute_samples[route]);
        std::printf("\\n");
    }
    std::printf("RT_COMPACT_EVIDENCE verification=all_native_records_and_finalizer_bits_equal_before_timing\\n");
    return 0;
}

int run_benchmark(int caller_device) {
    cudaDeviceProp properties{};
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaGetDeviceProperties(&properties, caller_device) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        return fail("benchmark CUDA device provenance query failed");
    }
    std::printf(
        "RT_COMPACT_EVIDENCE_BEGIN device=%d name=%s compute_capability=%d.%d driver=%d runtime=%d "
        "method=resident-input_cold-create_and_warm-synchronous-execute no_speedup_claim=1\\n",
        caller_device,
        properties.name,
        properties.major,
        properties.minor,
        driver_version,
        runtime_version
    );
    if (benchmark_execute_case(65536u, 32u, caller_device) != 0 ||
        benchmark_execute_case(262144u, 64u, caller_device) != 0) {
        return 1;
    }
    std::printf("RT_COMPACT_EVIDENCE_END status=accepted_exactness_precondition_only\\n");
    return 0;
}

int run_correctness(int* caller_device_out) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
    int caller_device = 0;
    if (cudaGetDevice(&caller_device) != cudaSuccess) return 77;

    const int tiny = exercise_tiny_rt_smoke(caller_device);
    if (tiny != 0) return tiny;
    std::puts("RT_COMPACT_CASE PASS case=overlap_exact_oracle");
    std::puts("RT_COMPACT_CASE PASS case=empty_labels_insufficient_support");
    if (exercise_weighted_regions(caller_device) != 0) return 1;
    std::puts("RT_COMPACT_CASE PASS case=weighted_regions_exact_and_fail_closed");
    // There is intentionally no caller first-hit selector. This 1D partition
    // is a semantic parity case, not a request for the internal 2D first-hit
    // proof; exact records must match both RequireRT and FORCE_SM regardless
    // of the native geometry choice.
    if (exercise_nonoverlap_partition_case(caller_device) != 0) return 1;
    if (exercise_triangle_eligible_partition_case(caller_device) != 0) return 1;
    std::puts("RT_COMPACT_CASE PASS case=triangle_eligible_rt_parity");
    if (exercise_multi_axis_group_case(caller_device) != 0) return 1;
    std::puts("RT_COMPACT_CASE PASS case=multigroup_exact_parity");
    // These two different non-power-of-two row counts are deliberately not a
    // timing campaign. They prevent an 8-row-only mask/bin implementation from
    // masquerading as a general compact-count path and exercise a fresh
    // paired-bank permutation/query identity.
    if (exercise_generated_binned_case(37u, false, caller_device) != 0 ||
        exercise_generated_binned_case(257u, true, caller_device) != 0) {
        return 1;
    }
    if (caller_device_out != nullptr) *caller_device_out = caller_device;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string_view(argv[1]) == "--benchmark";
    const bool abi_only = argc == 2 && std::string_view(argv[1]) == "--abi-only";
    if (argc > 2 || (argc == 2 && !benchmark && !abi_only)) {
        std::fprintf(
            stderr,
            "usage: gafime_cuda_rt_semantic_compact_smoke [--abi-only|--benchmark]\\n"
        );
        return 2;
    }
    const int abi = exercise_host_only_compact_abi_rejections();
    if (abi != 0) return abi;
    std::puts("RT_COMPACT_CASE PASS case=abi_safety");
    if (abi_only) return 0;
    int caller_device = -1;
    const int correctness = run_correctness(&caller_device);
    if (correctness != 0 || !benchmark) return correctness;
    return run_benchmark(caller_device);
}
