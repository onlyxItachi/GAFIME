// Host-only checks of the actual Metal result writer. No device/runtime calls.
#include "../../src/metal/launcher.mm"

#include <array>
#include <cstdlib>

// These checks must execute in Release/RelWithDebInfo builds too.
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d: %s failed\\n", __FILE__, __LINE__, #condition); \
    std::exit(1); \
} } while (false)

namespace {
struct Fixture {
    std::array<uint32_t, 3> descriptors{2, 0, 77};
    uint32_t metric = GAFIME_METRIC_PEARSON;
    GafimeArityChunk chunk{};
    GafimeLaunchProtocol protocol{};
    GafimeResultTable result{};
    std::array<uint32_t, 4> combos{};
    std::array<float, 4> scores{99, 99, 99, 99};
    std::array<uint32_t, 2> ranks{}, families{}, flags{};
    std::array<uint64_t, 2> ids{};

    explicit Fixture(uint64_t rows = 1) {
        chunk.family = GAFIME_FAMILY_CONTINUOUS;
        chunk.arity = 1;
        chunk.combo_count = rows;
        chunk.descriptor_count = rows;
        protocol.abi_version = GAFIME_ABI_VERSION;
        protocol.backend_kind = GAFIME_BACKEND_METAL;
        protocol.max_arity = 1;
        protocol.n_samples = 4;
        protocol.n_features = 3;
        protocol.family_count = 1;
        protocol.combo_indices = {descriptors.data(), rows};
        protocol.metric_ids = {&metric, 1};
        protocol.chunks = &chunk;
        protocol.chunk_count = 1;
        result.abi_version = GAFIME_ABI_VERSION;
        result.max_arity = 1;
        result.metric_count = 1;
        result.capacity = rows;
        result.combo_indices = combos.data();
        result.metric_values = scores.data();
        result.ranks = ranks.data();
        result.families = families.data();
        result.candidate_ids = ids.data();
        result.row_flags = flags.data();
    }

    void write(const std::vector<uint32_t>& rows, bool compact = false) {
        CHECK(validate_protocol(&protocol, 4, 3) == GAFIME_STATUS_OK);
        CHECK(validate_result_table(&protocol, &result) == GAFIME_STATUS_OK);
        CHECK(write_result_rows(&protocol, &result, {0.25f, 0.5f}, rows, compact)
            == GAFIME_STATUS_OK);
    }
};
}  // namespace

int main() {
    {
        Fixture f;
        // The declared metric span is one float; spare physical storage keeps
        // this negative control bounded even against the original writer.
        f.result.combo_indices = &f.result.metric_count;
        f.write({0});
        CHECK(f.result.metric_count == 2);
        CHECK(f.scores[0] == 0.25f && f.scores[1] == 99);
        CHECK(f.result.row_count == 1);
    }
    {
        Fixture f(2);
        f.result.combo_indices = f.descriptors.data() + 1;
        f.write({0, 1});
        CHECK(f.descriptors[1] == 2 && f.descriptors[2] == 0);
    }
    {
        Fixture f(2);
        f.result.max_arity = 2;
        f.result.metric_count = 2;
        f.write({1, 0});
        CHECK((f.combos == std::array<uint32_t, 4>{0, UINT32_MAX, 2, UINT32_MAX}));
        CHECK((f.scores == std::array<float, 4>{0.5f, 0, 0.25f, 0}));
        CHECK(f.ids[0] == 1 && f.ids[1] == 0);
    }
    {
        Fixture f(2);
        f.write({1}, true);
        CHECK(f.scores[0] == 0.25f && f.scores[1] == 99);
        CHECK(f.combos[0] == 0 && f.result.row_count == 1);
    }
    {
        Fixture f;
        f.result.flags = UINT32_MAX;
        f.write({});
        CHECK(f.scores[0] == 99 && f.result.row_count == 0 && f.result.flags == 0);
    }
    {
        Fixture f;
        // Valid unary descriptor sharing storage with result.flags must be
        // preserved before flag initialization clears the header field.
        f.result.flags = 2;
        f.protocol.combo_indices = {&f.result.flags, 1};
        f.write({0});
        CHECK(f.combos[0] == 2);
        CHECK(f.scores[0] == 0.25f && f.scores[1] == 99);
        CHECK(f.result.flags == 0 && f.result.row_count == 1);
    }
    std::puts("Metal host result-metadata controls passed (no GPU execution)");
}
