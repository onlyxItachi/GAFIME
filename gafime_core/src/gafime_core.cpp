#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gafime_native_data.h"
#include "gafime_real.h"
#include "gafime_simd.h"
#include "dlpack.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

enum class MetricId : int {
    Pearson = 0,
    Spearman = 1,
    MutualInfo = 2,
    R2 = 3,
};

struct RankScratch {
    std::vector<std::size_t> indices;
};

constexpr std::array<int, 7> kAdaptiveMiBinLevels{2, 4, 8, 16, 32, 64, 96};
constexpr int kSmallMiBins = 32;

struct MiScratch {
    std::array<real_t, static_cast<std::size_t>(kSmallMiBins) * static_cast<std::size_t>(kSmallMiBins)>
        joint_small{};
    std::array<real_t, static_cast<std::size_t>(kSmallMiBins)> p_x_small{};
    std::array<real_t, static_cast<std::size_t>(kSmallMiBins)> p_y_small{};
    std::vector<real_t> joint;
    std::vector<real_t> p_x;
    std::vector<real_t> p_y;

    void ensure(int bins) {
        if (bins > kSmallMiBins) {
            std::size_t size = static_cast<std::size_t>(bins);
            joint.resize(size * size);
            p_x.resize(size);
            p_y.resize(size);
        }
    }

    real_t *joint_ptr(int bins) {
        return (bins <= kSmallMiBins) ? joint_small.data() : joint.data();
    }

    real_t *p_x_ptr(int bins) {
        return (bins <= kSmallMiBins) ? p_x_small.data() : p_x.data();
    }

    real_t *p_y_ptr(int bins) {
        return (bins <= kSmallMiBins) ? p_y_small.data() : p_y.data();
    }
};

struct NativeReportRecord {
    std::vector<std::int64_t> combo;
    std::vector<std::uint32_t> feature_name_ids;
    std::vector<std::uint32_t> metric_name_ids;
    std::vector<real_t> metric_values;
    std::vector<std::uint32_t> secondary_metric_name_ids;
    std::vector<real_t> secondary_metric_values;
    std::uint32_t family_id = 0;
    std::uint32_t expression_id = 0;
    std::uint32_t candidate_id_id = 0;
    bool has_params = false;
    py::object params;
};

struct NativeReportTable {
    std::vector<NativeReportRecord> records;
    std::vector<std::string> string_pool;
    std::unordered_map<std::string, std::uint32_t> string_ids;

    NativeReportTable() {
        intern("");
    }

    [[nodiscard]] std::size_t size() const {
        return records.size();
    }

    [[nodiscard]] std::size_t interned_string_count() const {
        return string_pool.size();
    }

    void append(
        std::vector<std::int64_t> combo,
        std::vector<std::string> feature_names,
        std::vector<std::string> metric_names,
        std::vector<real_t> metric_values,
        std::vector<std::string> secondary_metric_names,
        std::vector<real_t> secondary_metric_values,
        std::string family,
        std::string expression,
        py::object params,
        std::string candidate_id) {
        if (metric_names.size() != metric_values.size()) {
            throw std::invalid_argument("metric_names and metric_values length mismatch");
        }
        if (secondary_metric_names.size() != secondary_metric_values.size()) {
            throw std::invalid_argument("secondary metric names and values length mismatch");
        }
        NativeReportRecord record;
        record.combo = std::move(combo);
        record.feature_name_ids = intern_all(feature_names);
        record.metric_name_ids = intern_all(metric_names);
        record.metric_values = std::move(metric_values);
        record.secondary_metric_name_ids = intern_all(secondary_metric_names);
        record.secondary_metric_values = std::move(secondary_metric_values);
        record.family_id = intern(family);
        record.expression_id = intern(expression);
        record.candidate_id_id = intern(candidate_id);
        record.has_params = !params_empty(params);
        if (record.has_params) {
            record.params = std::move(params);
        }
        records.push_back(std::move(record));
    }

    const NativeReportRecord &at(std::size_t index) const {
        if (index >= records.size()) {
            throw py::index_error();
        }
        return records[index];
    }

    std::vector<std::string> feature_names(std::size_t index) const {
        return resolve_all(at(index).feature_name_ids);
    }

    std::vector<std::string> metric_names(std::size_t index) const {
        return resolve_all(at(index).metric_name_ids);
    }

    std::vector<std::string> secondary_metric_names(std::size_t index) const {
        return resolve_all(at(index).secondary_metric_name_ids);
    }

    std::string family(std::size_t index) const {
        return resolve(at(index).family_id);
    }

    std::string expression(std::size_t index) const {
        return resolve(at(index).expression_id);
    }

    py::object params_at(std::size_t index) const {
        const NativeReportRecord &record = at(index);
        if (!record.has_params) {
            return py::dict();
        }
        return record.params;
    }

    std::string candidate_id(std::size_t index) const {
        return resolve(at(index).candidate_id_id);
    }

private:
    std::uint32_t intern(const std::string &value) {
        auto it = string_ids.find(value);
        if (it != string_ids.end()) {
            return it->second;
        }
        if (string_pool.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::overflow_error("NativeReportTable string pool exceeds uint32 id capacity");
        }
        std::uint32_t id = static_cast<std::uint32_t>(string_pool.size());
        string_pool.push_back(value);
        string_ids.emplace(string_pool.back(), id);
        return id;
    }

    std::vector<std::uint32_t> intern_all(const std::vector<std::string> &values) {
        std::vector<std::uint32_t> ids;
        ids.reserve(values.size());
        for (const std::string &value : values) {
            ids.push_back(intern(value));
        }
        return ids;
    }

    const std::string &resolve(std::uint32_t id) const {
        if (id >= string_pool.size()) {
            throw py::index_error();
        }
        return string_pool[id];
    }

    std::vector<std::string> resolve_all(const std::vector<std::uint32_t> &ids) const {
        std::vector<std::string> values;
        values.reserve(ids.size());
        for (std::uint32_t id : ids) {
            values.push_back(resolve(id));
        }
        return values;
    }

    static bool params_empty(const py::object &params) {
        if (!params || params.is_none()) {
            return true;
        }
        try {
            return py::len(params) == 0;
        } catch (const py::error_already_set &) {
            return false;
        }
    }
};

std::string to_lower_ascii(std::string value) {
    for (char &ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return value;
}

int parse_metric_id(py::handle obj) {
    if (py::isinstance<py::int_>(obj)) {
        int id = obj.cast<int>();
        if (id < 0 || id > 3) {
            throw std::invalid_argument("metric id must be in [0, 3]");
        }
        return id;
    }
    if (py::isinstance<py::str>(obj)) {
        std::string name = to_lower_ascii(obj.cast<std::string>());
        if (name == "pearson") {
            return static_cast<int>(MetricId::Pearson);
        }
        if (name == "spearman") {
            return static_cast<int>(MetricId::Spearman);
        }
        if (name == "mutual_info") {
            return static_cast<int>(MetricId::MutualInfo);
        }
        if (name == "r2") {
            return static_cast<int>(MetricId::R2);
        }
        throw std::invalid_argument("unknown metric name: " + name);
    }
    throw std::invalid_argument("metric id must be an int or str");
}

std::vector<int> parse_metric_ids(const py::object &obj) {
    if (obj.is_none()) {
        return {static_cast<int>(MetricId::Pearson),
                static_cast<int>(MetricId::Spearman),
                static_cast<int>(MetricId::MutualInfo),
                static_cast<int>(MetricId::R2)};
    }
    py::sequence seq = obj.cast<py::sequence>();
    std::vector<int> ids;
    ids.reserve(seq.size());
    for (py::handle item : seq) {
        ids.push_back(parse_metric_id(item));
    }
    return ids;
}

Matrix parse_matrix(const py::sequence &rows) {
    Matrix matrix;
    matrix.n_samples = static_cast<std::size_t>(rows.size());
    if (matrix.n_samples == 0) {
        throw std::invalid_argument("X must contain at least one sample");
    }

    py::sequence first = rows[0].cast<py::sequence>();
    matrix.n_features = static_cast<std::size_t>(first.size());
    if (matrix.n_features == 0) {
        throw std::invalid_argument("X must contain at least one feature");
    }
    matrix.data.reserve(matrix.n_samples * matrix.n_features);
    for (py::handle row_obj : rows) {
        py::sequence row = row_obj.cast<py::sequence>();
        if (static_cast<std::size_t>(row.size()) != matrix.n_features) {
            throw std::invalid_argument("X rows must all have the same feature count");
        }
        for (py::handle value : row) {
            real_t out = value.cast<real_t>();
            if (!std::isfinite(out)) {
                throw std::invalid_argument("X must be finite");
            }
            matrix.data.push_back(out);
        }
    }
    return matrix;
}

Matrix parse_matrix_buffer(py::buffer values, std::size_t n_samples, std::size_t n_features) {
    if (n_samples == 0) {
        throw std::invalid_argument("X must contain at least one sample");
    }
    if (n_features == 0) {
        throw std::invalid_argument("X must contain at least one feature");
    }
    py::buffer_info info = values.request();
    std::size_t expected = n_samples * n_features;
    if (info.ndim != 1 || static_cast<std::size_t>(info.size) != expected) {
        throw std::invalid_argument("X buffer must be flat row-major with n_samples * n_features values");
    }
    Matrix matrix;
    matrix.n_samples = n_samples;
    matrix.n_features = n_features;
    matrix.data.resize(expected);
    if (info.itemsize == sizeof(float)) {
        const auto *ptr = static_cast<const float *>(info.ptr);
        for (std::size_t i = 0; i < expected; ++i) {
            real_t value = static_cast<real_t>(ptr[i]);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("X must be finite");
            }
            matrix.data[i] = value;
        }
        return matrix;
    }
    if (info.itemsize == sizeof(double)) {
        const auto *ptr = static_cast<const double *>(info.ptr);
        for (std::size_t i = 0; i < expected; ++i) {
            real_t value = static_cast<real_t>(ptr[i]);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("X must be finite");
            }
            matrix.data[i] = value;
        }
        return matrix;
    }
    throw std::invalid_argument("X buffer must contain float32 or float64 values");
}

std::vector<real_t> parse_vector(const py::sequence &values, const char *name) {
    std::vector<real_t> out;
    out.reserve(static_cast<std::size_t>(values.size()));
    for (py::handle value : values) {
        real_t v = value.cast<real_t>();
        if (!std::isfinite(v)) {
            throw std::invalid_argument(std::string(name) + " must be finite");
        }
        out.push_back(v);
    }
    return out;
}

std::vector<real_t> parse_vector_buffer(py::buffer values, std::size_t expected, const char *name) {
    py::buffer_info info = values.request();
    if (info.ndim != 1 || static_cast<std::size_t>(info.size) != expected) {
        throw std::invalid_argument(std::string(name) + " buffer must be flat with the expected length");
    }
    std::vector<real_t> out(expected);
    if (info.itemsize == sizeof(float)) {
        const auto *ptr = static_cast<const float *>(info.ptr);
        for (std::size_t i = 0; i < expected; ++i) {
            real_t value = static_cast<real_t>(ptr[i]);
            if (!std::isfinite(value)) {
                throw std::invalid_argument(std::string(name) + " must be finite");
            }
            out[i] = value;
        }
        return out;
    }
    if (info.itemsize == sizeof(double)) {
        const auto *ptr = static_cast<const double *>(info.ptr);
        for (std::size_t i = 0; i < expected; ++i) {
            real_t value = static_cast<real_t>(ptr[i]);
            if (!std::isfinite(value)) {
                throw std::invalid_argument(std::string(name) + " must be finite");
            }
            out[i] = value;
        }
        return out;
    }
    throw std::invalid_argument(std::string(name) + " buffer must contain float32 or float64 values");
}

Vector parse_native_vector(const py::sequence &values, const char *name) {
    return Vector{parse_vector(values, name)};
}

Vector parse_native_vector_buffer(py::buffer values, std::size_t expected, const char *name) {
    return Vector{parse_vector_buffer(values, expected, name)};
}

std::vector<std::size_t> parse_row_indices(const py::sequence &indices, std::size_t n_samples) {
    std::vector<std::size_t> out;
    out.reserve(static_cast<std::size_t>(indices.size()));
    for (py::handle idx_obj : indices) {
        auto idx = static_cast<std::int64_t>(idx_obj.cast<long long>());
        if (idx < 0 || static_cast<std::size_t>(idx) >= n_samples) {
            throw std::invalid_argument("row index out of bounds");
        }
        out.push_back(static_cast<std::size_t>(idx));
    }
    return out;
}

std::vector<real_t> matrix_row(const Matrix &X, std::size_t row) {
    if (row >= X.n_samples) {
        throw std::invalid_argument("row index out of bounds");
    }
    std::size_t start = row * X.n_features;
    return std::vector<real_t>(X.data.begin() + static_cast<std::ptrdiff_t>(start),
                               X.data.begin() + static_cast<std::ptrdiff_t>(start + X.n_features));
}

std::vector<std::vector<real_t>> matrix_rows(const Matrix &X) {
    std::vector<std::vector<real_t>> rows;
    rows.reserve(X.n_samples);
    for (std::size_t row = 0; row < X.n_samples; ++row) {
        rows.push_back(matrix_row(X, row));
    }
    return rows;
}

std::vector<real_t> matrix_column(const Matrix &X, std::size_t col) {
    if (col >= X.n_features) {
        throw std::invalid_argument("column index out of bounds");
    }
    std::vector<real_t> out(X.n_samples);
    for (std::size_t row = 0; row < X.n_samples; ++row) {
        out[row] = X.value(row, col);
    }
    return out;
}

Matrix matrix_select_rows(const Matrix &X, const py::sequence &indices) {
    std::vector<std::size_t> rows = parse_row_indices(indices, X.n_samples);
    Matrix out;
    out.n_samples = rows.size();
    out.n_features = X.n_features;
    out.data.reserve(out.n_samples * out.n_features);
    for (std::size_t row : rows) {
        std::size_t start = row * X.n_features;
        out.data.insert(out.data.end(),
                        X.data.begin() + static_cast<std::ptrdiff_t>(start),
                        X.data.begin() + static_cast<std::ptrdiff_t>(start + X.n_features));
    }
    return out;
}

Vector vector_select(const Vector &values, const py::sequence &indices) {
    std::vector<std::size_t> rows = parse_row_indices(indices, values.data.size());
    Vector out;
    out.data.reserve(rows.size());
    for (std::size_t row : rows) {
        out.data.push_back(values.data[row]);
    }
    return out;
}

std::vector<std::vector<std::int64_t>> parse_combos(
    const py::sequence &combos,
    std::size_t n_features) {
    std::vector<std::vector<std::int64_t>> out;
    out.reserve(static_cast<std::size_t>(combos.size()));
    for (py::handle combo_obj : combos) {
        py::sequence combo = combo_obj.cast<py::sequence>();
        if (combo.size() < 1) {
            throw std::invalid_argument("combination entries must be non-empty");
        }
        std::vector<std::int64_t> values;
        values.reserve(static_cast<std::size_t>(combo.size()));
        for (py::handle idx_obj : combo) {
            auto idx = static_cast<std::int64_t>(idx_obj.cast<long long>());
            if (idx < 0 || static_cast<std::size_t>(idx) >= n_features) {
                throw std::invalid_argument("combo index out of feature bounds");
            }
            values.push_back(idx);
        }
        out.push_back(std::move(values));
    }
    return out;
}

std::vector<real_t> compute_means(const Matrix &X) {
    std::vector<real_t> means(X.n_features, real_t{0});
    if (X.n_samples == 0 || X.n_features == 0) {
        return means;
    }
    for (std::size_t row = 0; row < X.n_samples; ++row) {
        for (std::size_t col = 0; col < X.n_features; ++col) {
            means[col] += X.value(row, col);
        }
    }
    real_t inv = real_t{1} / static_cast<real_t>(X.n_samples);
    for (real_t &value : means) {
        value *= inv;
    }
    return means;
}

real_t compute_column_std(const Matrix &X, std::size_t feature) {
    if (feature >= X.n_features || X.n_samples == 0) {
        return real_t{0};
    }
    real_t sum = real_t{0};
    real_t sum_sq = real_t{0};
    for (std::size_t row = 0; row < X.n_samples; ++row) {
        real_t value = X.value(row, feature);
        sum += value;
        sum_sq += value * value;
    }
    real_t mean = sum / static_cast<real_t>(X.n_samples);
    real_t var = (sum_sq / static_cast<real_t>(X.n_samples)) - mean * mean;
    return static_cast<real_t>(std::sqrt(static_cast<double>(std::max(var, real_t{0}))));
}

std::vector<real_t> centered_column(const Matrix &X, std::size_t feature, real_t mean_value) {
    if (feature >= X.n_features) {
        throw std::invalid_argument("column index out of bounds");
    }
    std::vector<real_t> out(X.n_samples);
    for (std::size_t row = 0; row < X.n_samples; ++row) {
        out[row] = X.value(row, feature) - mean_value;
    }
    return out;
}

Vector centered_column_buffer(const Matrix &X, std::size_t feature, real_t mean_value) {
    return Vector{centered_column(X, feature, mean_value)};
}

Vector feature_major_buffer(const Matrix &X) {
    Vector out;
    out.data.reserve(X.data.size());
    for (std::size_t col = 0; col < X.n_features; ++col) {
        for (std::size_t row = 0; row < X.n_samples; ++row) {
            out.data.push_back(X.value(row, col));
        }
    }
    return out;
}

void rankdata(std::span<const real_t> data, RankScratch &scratch, std::vector<real_t> &out) {
    std::size_t n = data.size();
    out.resize(n);
    scratch.indices.resize(n);
    std::iota(scratch.indices.begin(), scratch.indices.end(), 0);

    std::stable_sort(scratch.indices.begin(), scratch.indices.end(),
                     [data](std::size_t a, std::size_t b) { return data[a] < data[b]; });

    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i + 1;
        while (j < n && data[scratch.indices[j]] == data[scratch.indices[i]]) {
            ++j;
        }
        real_t avg_rank = real_t{0.5} * static_cast<real_t>(i + j - 1);
        for (std::size_t k = i; k < j; ++k) {
            out[scratch.indices[k]] = avg_rank;
        }
        i = j;
    }
}

real_t pearson_from_sums(
    real_t sum_x,
    real_t sum_x2,
    real_t dot_xy,
    real_t var_y,
    std::size_t n) {
    if (n == 0) {
        return real_t{0};
    }
    real_t mean_x = sum_x / static_cast<real_t>(n);
    real_t var_x = sum_x2 - (sum_x * mean_x);
    if (var_x <= real_t{0} || var_y <= real_t{0}) {
        return real_t{0};
    }
    real_t denom = static_cast<real_t>(std::sqrt(static_cast<double>(var_x * var_y)));
    if (denom == real_t{0}) {
        return real_t{0};
    }
    return dot_xy / denom;
}

int choose_dense_mi_bins(std::size_t n, int max_bins) {
    int capped = std::max(2, std::min(max_bins, 96));
    int best = 2;
    for (int level : kAdaptiveMiBinLevels) {
        if (level > capped) {
            break;
        }
        long long required = static_cast<long long>(level) * level * 8LL;
        if (static_cast<long long>(n) >= required) {
            best = level;
        }
    }
    return best;
}

bool build_bins(std::span<const real_t> values, int max_bins, std::vector<int> &out_bins) {
    std::size_t n = values.size();
    int bins = choose_dense_mi_bins(n, max_bins);
    if (bins < 2 || n == 0) {
        return false;
    }
    real_t vmin = values[0];
    real_t vmax = values[0];
    for (std::size_t i = 1; i < n; ++i) {
        vmin = std::min(vmin, values[i]);
        vmax = std::max(vmax, values[i]);
    }
    if (vmin == vmax) {
        return false;
    }

    std::vector<real_t> unique(values.begin(), values.end());
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.size() <= static_cast<std::size_t>(bins)) {
        out_bins.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto it = std::lower_bound(unique.begin(), unique.end(), values[i]);
            out_bins[i] = static_cast<int>(std::distance(unique.begin(), it));
        }
        return true;
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [values](std::size_t a, std::size_t b) { return values[a] < values[b]; });
    out_bins.resize(n);
    for (std::size_t pos = 0; pos < n; ++pos) {
        int bin = static_cast<int>((pos * static_cast<std::size_t>(bins)) / n);
        out_bins[order[pos]] = std::min(bin, bins - 1);
    }
    return true;
}

void rankdata_double(std::span<const double> data, RankScratch &scratch, std::vector<real_t> &out) {
    std::size_t n = data.size();
    out.resize(n);
    scratch.indices.resize(n);
    std::iota(scratch.indices.begin(), scratch.indices.end(), 0);

    std::stable_sort(scratch.indices.begin(), scratch.indices.end(),
                     [data](std::size_t a, std::size_t b) { return data[a] < data[b]; });

    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i + 1;
        while (j < n && data[scratch.indices[j]] == data[scratch.indices[i]]) {
            ++j;
        }
        real_t avg_rank = real_t{0.5} * static_cast<real_t>(i + j - 1);
        for (std::size_t k = i; k < j; ++k) {
            out[scratch.indices[k]] = avg_rank;
        }
        i = j;
    }
}

bool build_bins_double(std::span<const double> values, int max_bins, std::vector<int> &out_bins) {
    std::size_t n = values.size();
    int bins = choose_dense_mi_bins(n, max_bins);
    if (bins < 2 || n == 0) {
        return false;
    }
    double vmin = values[0];
    double vmax = values[0];
    for (std::size_t i = 1; i < n; ++i) {
        vmin = std::min(vmin, values[i]);
        vmax = std::max(vmax, values[i]);
    }
    if (vmin == vmax) {
        return false;
    }

    std::vector<double> unique(values.begin(), values.end());
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.size() <= static_cast<std::size_t>(bins)) {
        out_bins.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto it = std::lower_bound(unique.begin(), unique.end(), values[i]);
            out_bins[i] = static_cast<int>(std::distance(unique.begin(), it));
        }
        return true;
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [values](std::size_t a, std::size_t b) { return values[a] < values[b]; });
    out_bins.resize(n);
    for (std::size_t pos = 0; pos < n; ++pos) {
        int bin = static_cast<int>((pos * static_cast<std::size_t>(bins)) / n);
        out_bins[order[pos]] = std::min(bin, bins - 1);
    }
    return true;
}

real_t mutual_info_from_vector(
    std::span<const real_t> x,
    int max_bins,
    std::span<const int> y_bins,
    MiScratch &scratch) {
    std::size_t n = x.size();
    int bins = choose_dense_mi_bins(n, max_bins);
    if (bins < 2 || n == 0 || y_bins.size() != n) {
        return real_t{0};
    }

    std::vector<int> x_bins;
    if (!build_bins(x, bins, x_bins)) {
        return real_t{0};
    }

    scratch.ensure(bins);
    real_t *joint = scratch.joint_ptr(bins);
    real_t *p_x = scratch.p_x_ptr(bins);
    real_t *p_y = scratch.p_y_ptr(bins);

    std::size_t bins_size = static_cast<std::size_t>(bins);
    std::fill_n(joint, bins_size * bins_size, real_t{0});
    std::fill_n(p_x, bins_size, real_t{0});
    std::fill_n(p_y, bins_size, real_t{0});

    for (std::size_t i = 0; i < n; ++i) {
        int x_bin = x_bins[i];
        int y_bin = y_bins[i];
        std::size_t x_idx = static_cast<std::size_t>(x_bin);
        std::size_t y_idx = static_cast<std::size_t>(y_bin);
        joint[x_idx * bins_size + y_idx] += real_t{1};
        p_x[x_idx] += real_t{1};
        p_y[y_idx] += real_t{1};
    }

    real_t inv_total = real_t{1} / static_cast<real_t>(n);
    real_t mi = real_t{0};
    std::size_t nonzero_x = 0;
    std::size_t nonzero_y = 0;
    for (std::size_t bx = 0; bx < bins_size; ++bx) {
        if (p_x[bx] > real_t{0}) {
            ++nonzero_x;
        }
    }
    for (std::size_t by = 0; by < bins_size; ++by) {
        if (p_y[by] > real_t{0}) {
            ++nonzero_y;
        }
    }
    if (nonzero_x < 2 || nonzero_y < 2) {
        return real_t{0};
    }
    for (std::size_t bx = 0; bx < bins_size; ++bx) {
        for (std::size_t by = 0; by < bins_size; ++by) {
            real_t count = joint[bx * bins_size + by];
            if (count <= real_t{0}) {
                continue;
            }
            real_t p = count * inv_total;
            real_t expected = (p_x[bx] * inv_total) * (p_y[by] * inv_total);
            if (expected > real_t{0}) {
                mi += static_cast<real_t>(static_cast<double>(p) * std::log(static_cast<double>(p / expected)));
            }
        }
    }
    real_t bias = static_cast<real_t>((nonzero_x - 1) * (nonzero_y - 1)) /
                  (real_t{2} * static_cast<real_t>(n));
    return std::max(mi - bias, real_t{0});
}

std::vector<real_t> build_interaction(
    const Matrix &X,
    const std::vector<std::int64_t> &combo,
    const std::vector<real_t> &means) {
    std::vector<real_t> out(X.n_samples);
    if (combo.size() == 1) {
        std::size_t feature = static_cast<std::size_t>(combo[0]);
        for (std::size_t row = 0; row < X.n_samples; ++row) {
            out[row] = X.value(row, feature);
        }
        return out;
    }
    for (std::size_t row = 0; row < X.n_samples; ++row) {
        real_t value = real_t{1};
        for (std::int64_t raw_feature : combo) {
            std::size_t feature = static_cast<std::size_t>(raw_feature);
            value *= X.value(row, feature) - means[feature];
        }
        out[row] = value;
    }
    return out;
}

}  // namespace

py::tuple pack_combos(const py::sequence &combos) {
    std::vector<std::int64_t> indices;
    std::vector<std::int64_t> offsets;
    offsets.push_back(0);
    for (py::handle combo_obj : combos) {
        py::sequence combo = combo_obj.cast<py::sequence>();
        for (py::handle idx_obj : combo) {
            indices.push_back(static_cast<std::int64_t>(idx_obj.cast<long long>()));
        }
        offsets.push_back(static_cast<std::int64_t>(indices.size()));
    }
    return py::make_tuple(indices, offsets);
}

std::vector<std::vector<real_t>> interaction_matrix(
    const py::sequence &X_rows,
    const py::sequence &combos_in) {
    Matrix X = parse_matrix(X_rows);
    auto combos = parse_combos(combos_in, X.n_features);
    std::vector<real_t> means = compute_means(X);
    std::vector<std::vector<real_t>> out;
    out.reserve(combos.size());
    {
        py::gil_scoped_release release;
        for (const auto &combo : combos) {
            out.push_back(build_interaction(X, combo, means));
        }
    }
    return out;
}

std::vector<std::vector<real_t>> score_combos_native(
    const Matrix &X,
    const std::vector<real_t> &y,
    const py::sequence &combos_in,
    const py::object &metric_ids,
    int mi_bins) {
    if (y.size() != X.n_samples) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    auto combos = parse_combos(combos_in, X.n_features);
    std::vector<int> metrics = parse_metric_ids(metric_ids);
    std::size_t n_metrics = metrics.size();
    std::vector<std::vector<real_t>> output(combos.size(), std::vector<real_t>(n_metrics, real_t{0}));

    bool need_pearson = false;
    bool need_spearman = false;
    bool need_mi = false;
    for (int id : metrics) {
        if (id == static_cast<int>(MetricId::Pearson) || id == static_cast<int>(MetricId::R2)) {
            need_pearson = true;
        } else if (id == static_cast<int>(MetricId::Spearman)) {
            need_spearman = true;
        } else if (id == static_cast<int>(MetricId::MutualInfo)) {
            need_mi = true;
        }
    }

    std::vector<real_t> means = compute_means(X);

    real_t sum_y = real_t{0};
    real_t sum_y2 = real_t{0};
    std::vector<real_t> y_centered;
    real_t var_y = real_t{0};
    if (need_pearson) {
        y_centered.resize(X.n_samples);
        for (real_t value : y) {
            sum_y += value;
            sum_y2 += value * value;
        }
        real_t mean_y = sum_y / static_cast<real_t>(X.n_samples);
        for (std::size_t i = 0; i < X.n_samples; ++i) {
            y_centered[i] = y[i] - mean_y;
        }
        var_y = sum_y2 - (sum_y * mean_y);
    }

    std::vector<real_t> y_rank;
    std::vector<real_t> y_rank_centered;
    RankScratch rank_scratch;
    real_t var_y_rank = real_t{0};
    if (need_spearman) {
        rankdata(std::span<const real_t>(y.data(), y.size()), rank_scratch, y_rank);
        y_rank_centered.resize(X.n_samples);
        real_t sum_r = real_t{0};
        real_t sum_r2 = real_t{0};
        for (real_t value : y_rank) {
            sum_r += value;
            sum_r2 += value * value;
        }
        real_t mean_r = sum_r / static_cast<real_t>(X.n_samples);
        for (std::size_t i = 0; i < X.n_samples; ++i) {
            y_rank_centered[i] = y_rank[i] - mean_r;
        }
        var_y_rank = sum_r2 - (sum_r * mean_r);
    }

    std::vector<int> y_bins;
    bool mi_ready = false;
    if (need_mi) {
        mi_ready = build_bins(std::span<const real_t>(y.data(), y.size()), mi_bins, y_bins);
    }
    std::span<const int> y_bins_span;
    if (mi_ready) {
        y_bins_span = std::span<const int>(y_bins.data(), y_bins.size());
    }

    {
        py::gil_scoped_release release;
#ifdef GAFIME_CORE_OPENMP
#pragma omp parallel
#endif
        {
            std::vector<real_t> x_rank;
            RankScratch thread_rank_scratch;
            MiScratch mi_scratch;
            if (need_mi && mi_bins >= 2) {
                mi_scratch.ensure(mi_bins);
            }

#ifdef GAFIME_CORE_OPENMP
#pragma omp for schedule(static)
#endif
            for (long long combo_idx = 0; combo_idx < static_cast<long long>(combos.size()); ++combo_idx) {
                const auto &combo = combos[static_cast<std::size_t>(combo_idx)];
                std::vector<real_t> interaction = build_interaction(X, combo, means);
                std::span<const real_t> interaction_span(interaction.data(), interaction.size());

                real_t sum_x = real_t{0};
                real_t sum_x2 = real_t{0};
                real_t dot_xy = real_t{0};
                if (need_pearson) {
                    GafimeAccumStats stats = gafime_accumulate_vector_stats_dispatch(
                        interaction.data(),
                        y_centered.data(),
                        X.n_samples);
                    sum_x = stats.sum_x;
                    sum_x2 = stats.sum_x2;
                    dot_xy = stats.dot_xy;
                }

                for (std::size_t m = 0; m < n_metrics; ++m) {
                    int id = metrics[m];
                    if (id == static_cast<int>(MetricId::Pearson)) {
                        output[static_cast<std::size_t>(combo_idx)][m] =
                            pearson_from_sums(sum_x, sum_x2, dot_xy, var_y, X.n_samples);
                    } else if (id == static_cast<int>(MetricId::R2)) {
                        real_t corr = pearson_from_sums(sum_x, sum_x2, dot_xy, var_y, X.n_samples);
                        output[static_cast<std::size_t>(combo_idx)][m] = corr * corr;
                    } else if (id == static_cast<int>(MetricId::Spearman)) {
                        rankdata(interaction_span, thread_rank_scratch, x_rank);
                        real_t sum_r = real_t{0};
                        real_t sum_r2 = real_t{0};
                        real_t dot_r = real_t{0};
                        for (std::size_t i = 0; i < X.n_samples; ++i) {
                            real_t val = x_rank[i];
                            sum_r += val;
                            sum_r2 += val * val;
                            dot_r += val * y_rank_centered[i];
                        }
                        output[static_cast<std::size_t>(combo_idx)][m] =
                            pearson_from_sums(sum_r, sum_r2, dot_r, var_y_rank, X.n_samples);
                    } else if (id == static_cast<int>(MetricId::MutualInfo)) {
                        real_t mi = real_t{0};
                        if (mi_ready && mi_bins >= 2) {
                            mi = mutual_info_from_vector(interaction_span, mi_bins, y_bins_span, mi_scratch);
                        }
                        output[static_cast<std::size_t>(combo_idx)][m] = mi;
                    } else {
                        throw std::invalid_argument("unknown metric id");
                    }
                }
            }
        }
    }
    return output;
}

std::vector<std::vector<real_t>> score_combos_buffer(
    const Matrix &X,
    const Vector &y_values,
    const py::sequence &combos_in,
    const py::object &metric_ids,
    int mi_bins) {
    return score_combos_native(X, y_values.data, combos_in, metric_ids, mi_bins);
}

// Native ridge baseline for split-aware discrete ranking. Faithful port of the
// Python `_continuous_baseline_prediction`: build standardized interaction
// columns for the (already top-k selected) combos, form X'X + ridge, solve, and
// return per-row predictions. The Python triple loop was ~72% of discrete wall
// at n=16384; this moves the O(n*k^2) accumulation + O(k^3) solve into C++.
// XtX/solve use double to match Python float semantics; columns reuse
// build_interaction so the math matches the pure-Python path.
std::vector<real_t> ridge_baseline_prediction(
    const Matrix &X,
    const Vector &y_values,
    const py::sequence &combos_in,
    double alpha) {
    const std::vector<real_t> &y = y_values.data;
    std::size_t n = X.n_samples;
    std::vector<real_t> out(n, real_t{0});
    if (n == 0) {
        return out;
    }
    double y_mean = 0.0;
    for (real_t value : y) {
        y_mean += static_cast<double>(value);
    }
    y_mean /= static_cast<double>(n);
    for (real_t &value : out) {
        value = static_cast<real_t>(y_mean);
    }
    if (y.size() != n) {
        return out;
    }

    auto combos = parse_combos(combos_in, X.n_features);
    if (combos.empty()) {
        return out;
    }
    std::vector<real_t> means = compute_means(X);

    std::vector<std::vector<real_t>> cols;
    cols.reserve(combos.size());
    for (const auto &combo : combos) {
        std::vector<real_t> col = build_interaction(X, combo, means);
        if (col.size() != n) {
            continue;
        }
        double col_mean = 0.0;
        for (real_t value : col) {
            col_mean += static_cast<double>(value);
        }
        col_mean /= static_cast<double>(n);
        double var = 0.0;
        for (real_t value : col) {
            double d = static_cast<double>(value) - col_mean;
            var += d * d;
        }
        var /= static_cast<double>(n);
        double sd = std::sqrt(std::max(var, 0.0));
        if (sd <= 1e-12) {
            continue;
        }
        for (real_t &value : col) {
            value = static_cast<real_t>((static_cast<double>(value) - col_mean) / sd);
        }
        cols.push_back(std::move(col));
    }
    std::size_t k = cols.size();
    if (k == 0) {
        return out;
    }

    std::vector<double> xtx(k * k, 0.0);
    std::vector<double> xty(k, 0.0);
    {
        py::gil_scoped_release release;
        for (std::size_t row = 0; row < n; ++row) {
            double yc = static_cast<double>(y[row]) - y_mean;
            for (std::size_t i = 0; i < k; ++i) {
                double xi = static_cast<double>(cols[i][row]);
                xty[i] += xi * yc;
                double *xtx_i = &xtx[i * k];
                for (std::size_t j = 0; j < k; ++j) {
                    xtx_i[j] += xi * static_cast<double>(cols[j][row]);
                }
            }
        }
    }
    for (std::size_t i = 0; i < k; ++i) {
        xtx[i * k + i] += alpha;
    }

    // Gauss-Jordan with partial pivoting (mirrors _solve_linear_system).
    std::vector<double> aug(k * (k + 1));
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            aug[i * (k + 1) + j] = xtx[i * k + j];
        }
        aug[i * (k + 1) + k] = xty[i];
    }
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t pivot = col;
        double best = std::fabs(aug[col * (k + 1) + col]);
        for (std::size_t r = col + 1; r < k; ++r) {
            double v = std::fabs(aug[r * (k + 1) + col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best <= 1e-12) {
            return out;  // singular -> mean fallback (matches Python None case)
        }
        if (pivot != col) {
            for (std::size_t j = 0; j <= k; ++j) {
                std::swap(aug[col * (k + 1) + j], aug[pivot * (k + 1) + j]);
            }
        }
        double pv = aug[col * (k + 1) + col];
        for (std::size_t j = col; j <= k; ++j) {
            aug[col * (k + 1) + j] /= pv;
        }
        for (std::size_t r = 0; r < k; ++r) {
            if (r == col) {
                continue;
            }
            double factor = aug[r * (k + 1) + col];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = col; j <= k; ++j) {
                aug[r * (k + 1) + j] -= factor * aug[col * (k + 1) + j];
            }
        }
    }
    std::vector<double> coef(k);
    for (std::size_t i = 0; i < k; ++i) {
        coef[i] = aug[i * (k + 1) + k];
    }

    for (std::size_t row = 0; row < n; ++row) {
        double acc = y_mean;
        for (std::size_t i = 0; i < k; ++i) {
            acc += static_cast<double>(cols[i][row]) * coef[i];
        }
        out[row] = static_cast<real_t>(acc);
    }
    return out;
}

// ============================================================================
// v0.5 residency v2: continuous report-metric cache (split-aware permutation).
// Caches the X-invariant work once (interaction vectors for pearson/r2, ranks
// for spearman, bins for MI), so each permuted y only re-does y-side work +
// O(n) reductions. Formulas mirror score_combos_native exactly for parity.
// Budget-gated: build returns None when the estimate exceeds max_bytes, and the
// session falls back to v1 resident scoring.
// ============================================================================
struct ContinuousMetricCache {
    std::size_t n = 0;
    std::vector<std::vector<std::int64_t>> combos;
    std::vector<int> metrics;            // metric ids, output order
    int mi_bins = 96;
    int mi_bin_count = 0;
    bool need_pearson = false;
    bool need_spearman = false;
    bool need_mi = false;
    std::vector<std::vector<real_t>> interactions;  // [k][n] if need_pearson
    std::vector<std::vector<real_t>> x_ranks;       // [k][n] if need_spearman
    std::vector<std::vector<int>> x_bins;           // [k][n] if need_mi
    std::vector<char> mi_ready;                     // [k] if need_mi
    std::size_t bytes = 0;
};

real_t mi_from_cached_bins(
    const std::vector<int> &x_bins,
    const std::vector<int> &y_bins,
    int bins,
    std::size_t n) {
    if (bins < 2 || n == 0 || x_bins.size() != n || y_bins.size() != n) {
        return real_t{0};
    }
    std::size_t bs = static_cast<std::size_t>(bins);
    std::vector<real_t> joint(bs * bs, real_t{0});
    std::vector<real_t> p_x(bs, real_t{0});
    std::vector<real_t> p_y(bs, real_t{0});
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t xb = static_cast<std::size_t>(x_bins[i]);
        std::size_t yb = static_cast<std::size_t>(y_bins[i]);
        joint[xb * bs + yb] += real_t{1};
        p_x[xb] += real_t{1};
        p_y[yb] += real_t{1};
    }
    real_t inv_total = real_t{1} / static_cast<real_t>(n);
    std::size_t nonzero_x = 0, nonzero_y = 0;
    for (std::size_t b = 0; b < bs; ++b) {
        if (p_x[b] > real_t{0}) ++nonzero_x;
        if (p_y[b] > real_t{0}) ++nonzero_y;
    }
    if (nonzero_x < 2 || nonzero_y < 2) {
        return real_t{0};
    }
    real_t mi = real_t{0};
    for (std::size_t bx = 0; bx < bs; ++bx) {
        for (std::size_t by = 0; by < bs; ++by) {
            real_t count = joint[bx * bs + by];
            if (count <= real_t{0}) continue;
            real_t p = count * inv_total;
            real_t expected = (p_x[bx] * inv_total) * (p_y[by] * inv_total);
            if (expected > real_t{0}) {
                mi += static_cast<real_t>(static_cast<double>(p) * std::log(static_cast<double>(p / expected)));
            }
        }
    }
    real_t bias = static_cast<real_t>((nonzero_x - 1) * (nonzero_y - 1)) /
                  (real_t{2} * static_cast<real_t>(n));
    return std::max(mi - bias, real_t{0});
}

py::object build_continuous_metric_cache(
    const Matrix &X,
    const py::sequence &combos_in,
    const py::object &metric_ids,
    int mi_bins,
    std::size_t max_bytes) {
    auto combos = parse_combos(combos_in, X.n_features);
    auto metrics = parse_metric_ids(metric_ids);
    bool need_pearson = false, need_spearman = false, need_mi = false;
    for (int id : metrics) {
        if (id == static_cast<int>(MetricId::Pearson) || id == static_cast<int>(MetricId::R2)) {
            need_pearson = true;
        } else if (id == static_cast<int>(MetricId::Spearman)) {
            need_spearman = true;
        } else if (id == static_cast<int>(MetricId::MutualInfo)) {
            need_mi = true;
        }
    }
    std::size_t n = X.n_samples;
    std::size_t k = combos.size();
    std::size_t per_row =
        (need_pearson ? sizeof(real_t) : 0) +
        (need_spearman ? sizeof(real_t) : 0) +
        (need_mi ? sizeof(int) : 0);
    std::size_t est = n * k * per_row;
    if (max_bytes > 0 && est > max_bytes) {
        return py::none();  // budget exceeded -> session falls back to v1
    }

    ContinuousMetricCache cache;
    cache.n = n;
    cache.combos = combos;
    cache.metrics = metrics;
    cache.mi_bins = mi_bins;
    cache.need_pearson = need_pearson;
    cache.need_spearman = need_spearman;
    cache.need_mi = need_mi;
    cache.mi_bin_count = need_mi ? choose_dense_mi_bins(n, mi_bins) : 0;
    cache.bytes = est;
    std::vector<real_t> means = compute_means(X);
    if (need_pearson) cache.interactions.resize(k);
    if (need_spearman) cache.x_ranks.resize(k);
    if (need_mi) { cache.x_bins.resize(k); cache.mi_ready.assign(k, 0); }

    {
        py::gil_scoped_release release;
#ifdef GAFIME_CORE_OPENMP
#pragma omp parallel
#endif
        {
            RankScratch rank_scratch;  // per-thread
#ifdef GAFIME_CORE_OPENMP
#pragma omp for schedule(static)
#endif
            for (long long ii = 0; ii < static_cast<long long>(k); ++ii) {
                std::size_t i = static_cast<std::size_t>(ii);
                std::vector<real_t> interaction = build_interaction(X, combos[i], means);
                if (need_spearman) {
                    std::vector<real_t> ranks;
                    rankdata(std::span<const real_t>(interaction.data(), n), rank_scratch, ranks);
                    cache.x_ranks[i] = std::move(ranks);
                }
                if (need_mi) {
                    std::vector<int> bins;
                    cache.mi_ready[i] = build_bins(std::span<const real_t>(interaction.data(), n), mi_bins, bins) ? 1 : 0;
                    cache.x_bins[i] = std::move(bins);
                }
                if (need_pearson) {
                    cache.interactions[i] = std::move(interaction);
                }
            }
        }
    }
    return py::cast(std::move(cache));
}

std::vector<std::vector<real_t>> score_continuous_metric_cache(
    ContinuousMetricCache &cache,
    const Vector &y_values) {
    const std::vector<real_t> &y = y_values.data;
    std::size_t n = cache.n;
    if (y.size() != n) {
        throw std::invalid_argument("y length must match the cached sample count");
    }
    std::size_t k = cache.combos.size();
    std::size_t n_metrics = cache.metrics.size();
    std::vector<std::vector<real_t>> output(k, std::vector<real_t>(n_metrics, real_t{0}));

    std::vector<real_t> y_centered;
    real_t var_y = real_t{0};
    if (cache.need_pearson) {
        real_t sum_y = real_t{0}, sum_y2 = real_t{0};
        for (real_t value : y) { sum_y += value; sum_y2 += value * value; }
        real_t mean_y = sum_y / static_cast<real_t>(n);
        y_centered.resize(n);
        for (std::size_t i = 0; i < n; ++i) y_centered[i] = y[i] - mean_y;
        var_y = sum_y2 - (sum_y * mean_y);
    }
    std::vector<real_t> y_rank_centered;
    real_t var_y_rank = real_t{0};
    if (cache.need_spearman) {
        RankScratch rs;
        std::vector<real_t> y_rank;
        rankdata(std::span<const real_t>(y.data(), n), rs, y_rank);
        real_t sum_r = real_t{0}, sum_r2 = real_t{0};
        for (real_t value : y_rank) { sum_r += value; sum_r2 += value * value; }
        real_t mean_r = sum_r / static_cast<real_t>(n);
        y_rank_centered.resize(n);
        for (std::size_t i = 0; i < n; ++i) y_rank_centered[i] = y_rank[i] - mean_r;
        var_y_rank = sum_r2 - (sum_r * mean_r);
    }
    std::vector<int> y_bins;
    bool y_mi_ready = false;
    if (cache.need_mi) {
        y_mi_ready = build_bins(std::span<const real_t>(y.data(), n), cache.mi_bins, y_bins);
    }

    {
        py::gil_scoped_release release;
#ifdef GAFIME_CORE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long ii = 0; ii < static_cast<long long>(k); ++ii) {
            std::size_t i = static_cast<std::size_t>(ii);
            real_t sum_x = real_t{0}, sum_x2 = real_t{0}, dot_xy = real_t{0};
            if (cache.need_pearson) {
                GafimeAccumStats stats = gafime_accumulate_vector_stats_dispatch(
                    cache.interactions[i].data(), y_centered.data(), n);
                sum_x = stats.sum_x; sum_x2 = stats.sum_x2; dot_xy = stats.dot_xy;
            }
            for (std::size_t m = 0; m < n_metrics; ++m) {
                int id = cache.metrics[m];
                if (id == static_cast<int>(MetricId::Pearson)) {
                    output[i][m] = pearson_from_sums(sum_x, sum_x2, dot_xy, var_y, n);
                } else if (id == static_cast<int>(MetricId::R2)) {
                    real_t corr = pearson_from_sums(sum_x, sum_x2, dot_xy, var_y, n);
                    output[i][m] = corr * corr;
                } else if (id == static_cast<int>(MetricId::Spearman)) {
                    const std::vector<real_t> &xr = cache.x_ranks[i];
                    real_t sum_r = real_t{0}, sum_r2 = real_t{0}, dot_r = real_t{0};
                    for (std::size_t j = 0; j < n; ++j) {
                        real_t v = xr[j];
                        sum_r += v; sum_r2 += v * v; dot_r += v * y_rank_centered[j];
                    }
                    output[i][m] = pearson_from_sums(sum_r, sum_r2, dot_r, var_y_rank, n);
                } else if (id == static_cast<int>(MetricId::MutualInfo)) {
                    real_t mi = real_t{0};
                    if (cache.need_mi && y_mi_ready && cache.mi_ready[i]) {
                        mi = mi_from_cached_bins(cache.x_bins[i], y_bins, cache.mi_bin_count, n);
                    }
                    output[i][m] = mi;
                }
            }
        }
    }
    return output;
}

// Helpers for the host time-series transform that mirrors
// evaluate_time_series_candidate (Python) for report-metric cache parity.
double fsum_range(const std::vector<real_t> &col, std::size_t start, std::size_t end) {
    std::vector<double> partials;
    partials.reserve(end >= start ? end - start + 1 : 0);
    for (std::size_t j = start; j <= end; ++j) {
        double x = static_cast<double>(col[j]);
        std::size_t i = 0;
        for (std::size_t p = 0; p < partials.size(); ++p) {
            double y = partials[p];
            if (std::fabs(x) < std::fabs(y)) {
                std::swap(x, y);
            }
            double hi = x + y;
            double yr = hi - x;
            double lo = y - yr;
            if (lo != 0.0) {
                partials[i++] = lo;
            }
            x = hi;
        }
        partials.resize(i);
        partials.push_back(x);
    }
    double total = 0.0;
    for (double value : partials) {
        total += value;
    }
    return total;
}

double fsum_squared_deviations(
    const std::vector<real_t> &col,
    std::size_t start,
    std::size_t end,
    double mean) {
    std::vector<double> partials;
    partials.reserve(end >= start ? end - start + 1 : 0);
    for (std::size_t j = start; j <= end; ++j) {
        double delta = static_cast<double>(col[j]) - mean;
        double x = delta * delta;
        std::size_t i = 0;
        for (std::size_t p = 0; p < partials.size(); ++p) {
            double y = partials[p];
            if (std::fabs(x) < std::fabs(y)) {
                std::swap(x, y);
            }
            double hi = x + y;
            double yr = hi - x;
            double lo = y - yr;
            if (lo != 0.0) {
                partials[i++] = lo;
            }
            x = hi;
        }
        partials.resize(i);
        partials.push_back(x);
    }
    double total = 0.0;
    for (double value : partials) {
        total += value;
    }
    return total;
}

// Kind codes match TIME_SERIES_KIND_CODES:
// 1 lag, 2 delta, 3 velocity, 4 accel, 5 rolling_mean, 6 rolling_std, 7 rolling_sum.
std::vector<double> build_ts_vector_double(const std::vector<real_t> &col, int kind, int lag, int window) {
    std::size_t n = col.size();
    std::vector<double> out(n);
    lag = std::max(1, lag);
    window = std::max(1, window);
    for (std::size_t idx = 0; idx < n; ++idx) {
        std::size_t lag_idx = (idx >= static_cast<std::size_t>(lag)) ? idx - static_cast<std::size_t>(lag) : 0;
        double value = static_cast<double>(col[idx]);
        double lag_value = static_cast<double>(col[lag_idx]);
        switch (kind) {
            case 1:
                out[idx] = lag_value;
                break;
            case 2:
                out[idx] = value - lag_value;
                break;
            case 3:
                out[idx] = (value - lag_value) / static_cast<double>(lag);
                break;
            case 4: {
                std::size_t lag2 = (idx >= static_cast<std::size_t>(2 * lag)) ? idx - static_cast<std::size_t>(2 * lag) : 0;
                out[idx] = (value - 2.0 * lag_value + static_cast<double>(col[lag2])) /
                           static_cast<double>(lag * lag);
                break;
            }
            case 5:
            case 6:
            case 7: {
                std::size_t start = (idx + 1 >= static_cast<std::size_t>(window)) ? idx + 1 - static_cast<std::size_t>(window) : 0;
                std::size_t count = idx - start + 1;
                double total = fsum_range(col, start, idx);
                if (kind == 7) {
                    out[idx] = total;
                } else {
                    double mean = total / static_cast<double>(count);
                    if (kind == 5) {
                        out[idx] = mean;
                    } else {
                        double variance = fsum_squared_deviations(col, start, idx, mean) /
                                          static_cast<double>(count);
                        out[idx] = std::sqrt(std::max(variance, 0.0));
                    }
                }
                break;
            }
            default:
                out[idx] = value;
        }
    }
    return out;
}

// TS-family twin of build_continuous_metric_cache: engineers each candidate's
// time-series vector once and caches the same X-invariant state, so the
// permutation null reuses score_continuous_metric_cache unchanged.
py::object build_time_series_metric_cache(
    const Matrix &X,
    const py::sequence &kinds_in,
    const py::sequence &feats_in,
    const py::sequence &lags_in,
    const py::sequence &windows_in,
    const py::object &metric_ids,
    int mi_bins,
    std::size_t max_bytes) {
    std::size_t k = static_cast<std::size_t>(kinds_in.size());
    if (static_cast<std::size_t>(feats_in.size()) != k ||
        static_cast<std::size_t>(lags_in.size()) != k ||
        static_cast<std::size_t>(windows_in.size()) != k) {
        throw std::invalid_argument("time-series descriptor arrays must have equal length");
    }
    std::vector<int> kinds(k), feats(k), lags(k), windows(k);
    for (std::size_t i = 0; i < k; ++i) {
        kinds[i] = kinds_in[i].cast<int>();
        feats[i] = feats_in[i].cast<int>();
        lags[i] = lags_in[i].cast<int>();
        windows[i] = windows_in[i].cast<int>();
        if (feats[i] < 0 || static_cast<std::size_t>(feats[i]) >= X.n_features) {
            throw std::invalid_argument("time-series feature index out of bounds");
        }
    }
    auto metrics = parse_metric_ids(metric_ids);
    bool need_pearson = false, need_spearman = false, need_mi = false;
    for (int id : metrics) {
        if (id == static_cast<int>(MetricId::Pearson) || id == static_cast<int>(MetricId::R2)) need_pearson = true;
        else if (id == static_cast<int>(MetricId::Spearman)) need_spearman = true;
        else if (id == static_cast<int>(MetricId::MutualInfo)) need_mi = true;
    }
    std::size_t n = X.n_samples;
    std::size_t per_row =
        (need_pearson ? sizeof(real_t) : 0) +
        (need_spearman ? sizeof(real_t) : 0) +
        (need_mi ? sizeof(int) : 0);
    std::size_t est = n * k * per_row;
    if (max_bytes > 0 && est > max_bytes) {
        return py::none();
    }

    ContinuousMetricCache cache;
    cache.n = n;
    cache.combos.resize(k);
    for (std::size_t i = 0; i < k; ++i) cache.combos[i] = {static_cast<std::int64_t>(feats[i])};
    cache.metrics = metrics;
    cache.mi_bins = mi_bins;
    cache.need_pearson = need_pearson;
    cache.need_spearman = need_spearman;
    cache.need_mi = need_mi;
    cache.mi_bin_count = need_mi ? choose_dense_mi_bins(n, mi_bins) : 0;
    cache.bytes = est;
    if (need_pearson) cache.interactions.resize(k);
    if (need_spearman) cache.x_ranks.resize(k);
    if (need_mi) { cache.x_bins.resize(k); cache.mi_ready.assign(k, 0); }

    {
        py::gil_scoped_release release;
#ifdef GAFIME_CORE_OPENMP
#pragma omp parallel
#endif
        {
            RankScratch rank_scratch;
#ifdef GAFIME_CORE_OPENMP
#pragma omp for schedule(static)
#endif
            for (long long ii = 0; ii < static_cast<long long>(k); ++ii) {
                std::size_t i = static_cast<std::size_t>(ii);
                std::vector<real_t> col = matrix_column(X, static_cast<std::size_t>(feats[i]));
                std::vector<double> vec = build_ts_vector_double(col, kinds[i], lags[i], windows[i]);
                if (need_spearman) {
                    std::vector<real_t> ranks;
                    rankdata_double(std::span<const double>(vec.data(), n), rank_scratch, ranks);
                    cache.x_ranks[i] = std::move(ranks);
                }
                if (need_mi) {
                    std::vector<int> bins;
                    cache.mi_ready[i] = build_bins_double(std::span<const double>(vec.data(), n), mi_bins, bins) ? 1 : 0;
                    cache.x_bins[i] = std::move(bins);
                }
                if (need_pearson) {
                    std::vector<real_t> narrowed(n);
                    for (std::size_t row = 0; row < n; ++row) {
                        narrowed[row] = static_cast<real_t>(vec[row]);
                    }
                    cache.interactions[i] = std::move(narrowed);
                }
            }
        }
    }
    return py::cast(std::move(cache));
}

std::string precision_name() {
#ifdef GAFIME_USE_DOUBLE_PRECISION
    return "float64";
#else
    return "float32";
#endif
}

// ---------------------------------------------------------------------------
// decision_path family (native C++ Core) — GBDT-style split-finding math over
// the whole dataset (NO subsampling — that was sklearn's crutch). Exact mode
// uses a prefix-sum sweep over sorted feature values; bounded mode samples split
// boundaries but still evaluates every row in the active node.
// Histogram/SIMD acceleration remains a later optimization. This implementation
// already supports depth-k recursion, residual boosting rounds, and an optional
// split-boundary cap that preserves whole-data evaluation while limiting the
// number of threshold candidates considered per feature.
// ---------------------------------------------------------------------------
struct DecisionPathRecord {
    std::vector<int> features;      // path features (depth 1 in staging 1)
    std::vector<double> thresholds;
    std::vector<int> signs;         // convention: -1 => x <= thr (left child), +1 => x > thr (right child)
    double gain = 0.0;              // variance-reduction fraction in (0, 1]
    double support = 0.0;           // fraction of rows on the selected (>thr) side
    int round_id = 0;               // boosting round (0 in staging 1)
    int candidate_id = -1;          // stable rank id within the returned list
};

// Best one-level variance-reduction split on feature f over all n rows (exact:
// sorts (x,y) and sweeps every value boundary via prefix sums).
// One condition along a root->leaf path. sign: -1 => x <= thr, +1 => x > thr.
struct GafimePathCond { int feature; double threshold; int sign; };

static std::vector<std::size_t> gafime_select_split_boundaries(
    const std::vector<std::size_t> &valid_bounds,
    int max_bins_per_feature) {
    if (max_bins_per_feature <= 0 ||
        valid_bounds.size() <= static_cast<std::size_t>(max_bins_per_feature)) {
        return valid_bounds;
    }
    const std::size_t cap = static_cast<std::size_t>(std::max(1, max_bins_per_feature));
    std::vector<std::size_t> selected;
    selected.reserve(cap);
    if (cap == 1) {
        selected.push_back(valid_bounds[valid_bounds.size() / 2]);
        return selected;
    }
    for (std::size_t j = 0; j < cap; ++j) {
        const std::size_t pos = (j * (valid_bounds.size() - 1)) / (cap - 1);
        const std::size_t boundary = valid_bounds[pos];
        if (selected.empty() || selected.back() != boundary) {
            selected.push_back(boundary);
        }
    }
    return selected;
}

// Best variance-reduction split over a SUBSET of rows (whole data WITHIN the node;
// no subsampling). Scans the feature shortlist; exact prefix-sum sweep over sorted
// values; splits only at value boundaries; respects min_leaf on both children.
static bool gafime_best_split_subset(
    const Matrix &X, const std::vector<double> &target,
    const std::vector<std::size_t> &idx, const std::vector<std::size_t> &feats,
    std::size_t min_leaf, int max_bins_per_feature, int &best_feature, double &best_thr) {
    const std::size_t m = idx.size();
    if (m < 2 * min_leaf) return false;
    double tsum = 0.0, tsq = 0.0;
    for (std::size_t i : idx) { double v = target[i]; tsum += v; tsq += v * v; }
    const double node_sse = tsq - tsum * tsum / static_cast<double>(m);
    if (!(node_sse > 1e-12)) return false;
    const std::size_t nf = X.n_features;
    double best_gain = 0.0;
    bool found = false;
    std::vector<std::pair<double, double>> xy(m);
    for (std::size_t fk = 0; fk < feats.size(); ++fk) {
        const std::size_t f = feats[fk];
        for (std::size_t k = 0; k < m; ++k) {
            xy[k].first = static_cast<double>(X.data[idx[k] * nf + f]);
            xy[k].second = target[idx[k]];
        }
        std::sort(xy.begin(), xy.end(),
                  [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
                      return a.first < b.first;
                  });
        if (max_bins_per_feature <= 0) {
            double ls = 0.0, lss = 0.0;
            std::size_t ln = 0;
            for (std::size_t k = 0; k + 1 < m; ++k) {
                ls += xy[k].second; lss += xy[k].second * xy[k].second; ln += 1;
                if (xy[k].first == xy[k + 1].first) continue;
                std::size_t rn = m - ln;
                if (ln < min_leaf || rn < min_leaf) continue;
                double rs = tsum - ls, rss = tsq - lss;
                double lsse = lss - ls * ls / static_cast<double>(ln);
                double rsse = rss - rs * rs / static_cast<double>(rn);
                double gain = (node_sse - lsse - rsse) / node_sse;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = static_cast<int>(f);
                    best_thr = 0.5 * (xy[k].first + xy[k + 1].first);
                    found = true;
                }
            }
        } else {
            std::vector<double> prefix_sum(m + 1, 0.0), prefix_sq(m + 1, 0.0);
            for (std::size_t k = 0; k < m; ++k) {
                prefix_sum[k + 1] = prefix_sum[k] + xy[k].second;
                prefix_sq[k + 1] = prefix_sq[k] + xy[k].second * xy[k].second;
            }
            std::vector<std::size_t> valid_bounds;
            valid_bounds.reserve(m);
            for (std::size_t k = 0; k + 1 < m; ++k) {
                if (xy[k].first == xy[k + 1].first) continue;
                std::size_t ln = k + 1;
                std::size_t rn = m - ln;
                if (ln < min_leaf || rn < min_leaf) continue;
                valid_bounds.push_back(k);
            }
            const std::vector<std::size_t> split_bounds =
                gafime_select_split_boundaries(valid_bounds, max_bins_per_feature);
            for (std::size_t k : split_bounds) {
                const std::size_t ln = k + 1;
                const std::size_t rn = m - ln;
                const double ls = prefix_sum[ln], lss = prefix_sq[ln];
                double rs = tsum - ls, rss = tsq - lss;
                double lsse = lss - ls * ls / static_cast<double>(ln);
                double rsse = rss - rs * rs / static_cast<double>(rn);
                double gain = (node_sse - lsse - rsse) / node_sse;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = static_cast<int>(f);
                    best_thr = 0.5 * (xy[k].first + xy[k + 1].first);
                    found = true;
                }
            }
        }
    }
    return found;
}

// Whole-data variance reduction of a conjunction indicator (the root->leaf path)
// vs the root, on the round residual r over ALL n rows. Also returns support.
static double gafime_conjunction_gain(
    const Matrix &X, const std::vector<double> &r, std::size_t n,
    double root_sum, double root_sq, const std::vector<GafimePathCond> &conds, double &support) {
    const std::size_t nf = X.n_features;
    double in_sum = 0.0, in_sq = 0.0;
    std::size_t in_n = 0;
    for (std::size_t i = 0; i < n; ++i) {
        bool inside = true;
        for (const GafimePathCond &c : conds) {
            double x = static_cast<double>(X.data[i * nf + static_cast<std::size_t>(c.feature)]);
            bool pass = (c.sign > 0) ? (x > c.threshold) : (x <= c.threshold);
            if (!pass) { inside = false; break; }
        }
        if (inside) { double v = r[i]; in_sum += v; in_sq += v * v; in_n += 1; }
    }
    support = static_cast<double>(in_n) / static_cast<double>(n);
    if (in_n == 0 || in_n == n) return 0.0;
    const std::size_t out_n = n - in_n;
    const double root_sse = root_sq - root_sum * root_sum / static_cast<double>(n);
    if (!(root_sse > 1e-12)) return 0.0;
    const double out_sum = root_sum - in_sum;
    const double out_sq = root_sq - in_sq;
    const double in_sse = in_sq - in_sum * in_sum / static_cast<double>(in_n);
    const double out_sse = out_sq - out_sum * out_sum / static_cast<double>(out_n);
    double gain = (root_sse - in_sse - out_sse) / root_sse;
    return gain > 0.0 ? gain : 0.0;
}

// Greedy CART recursion on a row-index subset; extracts root->leaf conjunction paths
// and fills per-row leaf predictions (for residual boosting). Whole-data within node.
static void gafime_grow_tree(
    const Matrix &X, const std::vector<double> &resid,
    const std::vector<std::size_t> &idx, const std::vector<std::size_t> &feats,
    int depth, int max_depth, std::size_t min_leaf, int max_bins_per_feature, int round_id,
    std::vector<GafimePathCond> &prefix, std::vector<double> &pred,
    std::size_t n, double root_sum, double root_sq,
    std::vector<DecisionPathRecord> &out) {
    double s = 0.0;
    for (std::size_t i : idx) s += resid[i];
    const double leaf_mean = idx.empty() ? 0.0 : s / static_cast<double>(idx.size());

    int bf = -1;
    double bthr = 0.0;
    bool split = (depth < max_depth) &&
                 gafime_best_split_subset(X, resid, idx, feats, min_leaf, max_bins_per_feature, bf, bthr);
    if (!split) {
        for (std::size_t i : idx) pred[i] = leaf_mean;
        if (!prefix.empty()) {
            DecisionPathRecord rec;
            double support = 0.0;
            rec.gain = gafime_conjunction_gain(X, resid, n, root_sum, root_sq, prefix, support);
            rec.support = support;
            rec.round_id = round_id;
            for (const GafimePathCond &c : prefix) {
                rec.features.push_back(c.feature);
                rec.thresholds.push_back(c.threshold);
                rec.signs.push_back(c.sign);
            }
            out.push_back(std::move(rec));
        }
        return;
    }
    const std::size_t nf = X.n_features;
    std::vector<std::size_t> left, right;
    left.reserve(idx.size());
    right.reserve(idx.size());
    for (std::size_t i : idx) {
        double x = static_cast<double>(X.data[i * nf + static_cast<std::size_t>(bf)]);
        if (x <= bthr) left.push_back(i); else right.push_back(i);
    }
    prefix.push_back(GafimePathCond{bf, bthr, -1});
    gafime_grow_tree(X, resid, left, feats, depth + 1, max_depth, min_leaf, max_bins_per_feature, round_id,
                     prefix, pred, n, root_sum, root_sq, out);
    prefix.back() = GafimePathCond{bf, bthr, +1};
    gafime_grow_tree(X, resid, right, feats, depth + 1, max_depth, min_leaf, max_bins_per_feature, round_id,
                     prefix, pred, n, root_sum, root_sq, out);
    prefix.pop_back();
}

static std::string gafime_path_key(const DecisionPathRecord &r) {
    std::string key;
    for (std::size_t i = 0; i < r.features.size(); ++i) {
        key += std::to_string(r.features[i]);
        key += ':';
        long t = static_cast<long>(r.thresholds[i] * 1e6);  // 1e-6 dedup resolution
        key += std::to_string(t);
        key += (r.signs[i] > 0) ? '>' : 'L';
        key += '|';
    }
    return key;
}

std::vector<DecisionPathRecord> find_decision_path_candidates(
    const Matrix &X,
    const Vector &y_values,
    const py::object &feature_ids_in,
    int max_depth,
    int max_paths,
    int max_bins_per_feature,   // split-boundary RESOLUTION, never a row subsample. <=0 (or >= unique cut
                                // count) => exhaustive exact sweep. Whole data always contributes.
    int min_leaf,
    int rounds,                 // residual-boosting rounds
    double learning_rate) {
    std::vector<DecisionPathRecord> out;
    const std::vector<real_t> &y = y_values.data;
    const std::size_t n = X.n_samples;
    if (n == 0 || y.size() != n || X.n_features == 0) {
        return out;
    }

    std::vector<std::size_t> feats;
    if (feature_ids_in.is_none()) {
        feats.resize(X.n_features);
        for (std::size_t f = 0; f < X.n_features; ++f) feats[f] = f;
    } else {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(feature_ids_in);
        for (auto item : seq) {
            long v = item.cast<long>();
            if (v >= 0 && static_cast<std::size_t>(v) < X.n_features) {
                feats.push_back(static_cast<std::size_t>(v));
            }
        }
        if (feats.empty()) {
            feats.resize(X.n_features);
            for (std::size_t f = 0; f < X.n_features; ++f) feats[f] = f;
        }
    }

    const std::size_t leaf = static_cast<std::size_t>(std::max(1, min_leaf));
    const int depth = std::max(1, max_depth);
    const int nrounds = std::max(1, rounds);
    const int split_cap = std::max(0, max_bins_per_feature);
    const double lr = (learning_rate > 0.0) ? learning_rate : 1.0;

    std::vector<double> resid(n);
    for (std::size_t i = 0; i < n; ++i) resid[i] = static_cast<double>(y[i]);
    std::vector<std::size_t> all_idx(n);
    for (std::size_t i = 0; i < n; ++i) all_idx[i] = i;

    std::vector<DecisionPathRecord> all;
    for (int r = 0; r < nrounds; ++r) {
        double root_sum = 0.0, root_sq = 0.0;
        for (double v : resid) { root_sum += v; root_sq += v * v; }
        const double root_sse = root_sq - root_sum * root_sum / static_cast<double>(n);
        if (!(root_sse > 1e-12)) break;  // residual exhausted
        std::vector<GafimePathCond> prefix;
        std::vector<double> pred(n, 0.0);
        gafime_grow_tree(X, resid, all_idx, feats, 0, depth, leaf, split_cap, r,
                         prefix, pred, n, root_sum, root_sq, all);
        for (std::size_t i = 0; i < n; ++i) resid[i] -= lr * pred[i];
    }

    // dedup identical conjunction paths across rounds (keep the max-gain instance)
    std::vector<std::pair<std::string, std::size_t>> keyed;
    keyed.reserve(all.size());
    for (std::size_t i = 0; i < all.size(); ++i) keyed.emplace_back(gafime_path_key(all[i]), i);
    std::sort(keyed.begin(), keyed.end(),
              [&](const std::pair<std::string, std::size_t> &a,
                  const std::pair<std::string, std::size_t> &b) {
                  if (a.first != b.first) return a.first < b.first;
                  return all[a.second].gain > all[b.second].gain;
              });
    std::vector<DecisionPathRecord> kept;
    for (std::size_t j = 0; j < keyed.size(); ++j) {
        if (j == 0 || keyed[j].first != keyed[j - 1].first) {
            kept.push_back(all[keyed[j].second]);
        }
    }
    std::sort(kept.begin(), kept.end(),
              [](const DecisionPathRecord &a, const DecisionPathRecord &b) { return a.gain > b.gain; });
    std::size_t k = (max_paths > 0)
                        ? std::min(kept.size(), static_cast<std::size_t>(max_paths))
                        : kept.size();
    out.assign(kept.begin(), kept.begin() + static_cast<std::ptrdiff_t>(k));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i].candidate_id = static_cast<int>(i);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Framework-grade zero-copy export protocols for the native buffers.
// __array_interface__ (numpy/cupy), __dlpack__/__dlpack_device__ (torch/JAX/
// numpy from_dlpack). CPU (kDLCPU) path; device handles are a follow-up that
// consumes the resident-session device pointer. Lifetime-safe: a borrowed
// tensor keeps the owning buffer alive until the consumer releases it.
// ---------------------------------------------------------------------------
namespace gafime_export {

// numpy __array_interface__ (v3) over a real_t, C-contiguous buffer.
inline py::dict array_interface(void *data, const std::vector<py::ssize_t> &shape) {
    py::dict iface;
    iface["version"] = 3;
    py::tuple shp(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        shp[i] = shape[i];
    }
    iface["shape"] = shp;
    // little-endian float; width from sizeof(real_t): '<f4' (fp32) / '<f8' (fp64)
    iface["typestr"] = std::string("<f") + std::to_string(sizeof(real_t));
    iface["data"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(data), false);  // (ptr, read_only)
    iface["strides"] = py::none();  // C-contiguous
    return iface;
}

// Owns the exported buffer's lifetime while a DLPack tensor borrows its memory.
struct DLPackContext {
    py::object owner;            // strong ref -> native buffer outlives the borrow
    std::vector<int64_t> shape;  // backing storage for DLTensor.shape
    DLManagedTensor tensor;
};

inline void dlpack_deleter(DLManagedTensor *self) {
    if (self == nullptr) {
        return;
    }
    auto *ctx = static_cast<DLPackContext *>(self->manager_ctx);
    if (ctx != nullptr) {
        py::gil_scoped_acquire gil;  // releasing the owner ref needs the GIL
        ctx->owner = py::object();
        delete ctx;
    }
}

// Fires only if the capsule was NOT consumed (consumers rename to "used_dltensor").
inline void dlpack_capsule_destructor(PyObject *capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto *mt = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
        if (mt != nullptr && mt->deleter != nullptr) {
            mt->deleter(mt);
        }
    }
}

inline py::object make_dlpack(py::object owner, void *data, std::vector<int64_t> shape) {
    auto *ctx = new DLPackContext{};
    ctx->owner = std::move(owner);
    ctx->shape = std::move(shape);
    DLManagedTensor &mt = ctx->tensor;
    mt.dl_tensor.data = data;
    mt.dl_tensor.device = DLDevice{kDLCPU, 0};
    mt.dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
    mt.dl_tensor.dtype = DLDataType{static_cast<uint8_t>(kDLFloat),
                                    static_cast<uint8_t>(sizeof(real_t) * 8), 1};
    mt.dl_tensor.shape = ctx->shape.data();
    mt.dl_tensor.strides = nullptr;  // compact row-major
    mt.dl_tensor.byte_offset = 0;
    mt.manager_ctx = ctx;
    mt.deleter = &dlpack_deleter;
    PyObject *capsule = PyCapsule_New(&mt, "dltensor", &dlpack_capsule_destructor);
    if (capsule == nullptr) {
        ctx->owner = py::object();
        delete ctx;
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(capsule);
}

}  // namespace gafime_export

PYBIND11_MODULE(gafime_core, m) {
    m.doc() = "Native C++ kernels for GAFIME interaction and metric scoring.";

    py::class_<Vector>(m, "NativeVectorBuffer", py::buffer_protocol())
        .def(py::init([](const py::sequence &values) {
            return parse_native_vector(values, "values");
        }))
        .def("__len__", [](const Vector &v) { return v.data.size(); })
        .def("__getitem__", [](const Vector &v, std::size_t idx) {
            if (idx >= v.data.size()) {
                throw py::index_error();
            }
            return v.data[idx];
        })
        .def_property_readonly("nbytes", &Vector::nbytes)
        .def("to_list", [](const Vector &v) { return v.data; })
        .def("select", &vector_select)
        .def_property_readonly("__array_interface__", [](Vector &v) {
            return gafime_export::array_interface(v.data.data(),
                {static_cast<py::ssize_t>(v.data.size())});
        })
        .def("__dlpack__", [](py::object self, py::object, py::object, py::object, py::object) {
            Vector &v = self.cast<Vector &>();
            return gafime_export::make_dlpack(self, v.data.data(),
                {static_cast<int64_t>(v.data.size())});
        }, py::arg("stream") = py::none(), py::arg("max_version") = py::none(),
           py::arg("dl_device") = py::none(), py::arg("copy") = py::none())
        .def("__dlpack_device__", [](const Vector &) {
            return py::make_tuple(static_cast<int>(kDLCPU), 0);
        })
        .def_buffer([](Vector &v) {
            return py::buffer_info(
                v.data.data(),
                sizeof(real_t),
                py::format_descriptor<real_t>::format(),
                1,
                {static_cast<py::ssize_t>(v.data.size())},
                {static_cast<py::ssize_t>(sizeof(real_t))});
        });

    py::class_<Matrix>(m, "NativeMatrixBuffer", py::buffer_protocol())
        .def(py::init([](const py::sequence &rows) {
            return parse_matrix(rows);
        }))
        .def_property_readonly("n_samples", [](const Matrix &X) { return X.n_samples; })
        .def_property_readonly("n_features", [](const Matrix &X) { return X.n_features; })
        .def_property_readonly("size", &Matrix::size)
        .def_property_readonly("nbytes", &Matrix::nbytes)
        .def("value", &Matrix::value)
        .def("row", &matrix_row)
        .def("rows", &matrix_rows)
        .def("column", &matrix_column)
        .def("column_buffer", [](const Matrix &X, std::size_t col) { return Vector{matrix_column(X, col)}; })
        .def("to_list", [](const Matrix &X) { return X.data; })
        .def("select_rows", &matrix_select_rows)
        .def("column_means", &compute_means)
        .def("column_std", &compute_column_std)
        .def("centered_column", &centered_column_buffer)
        .def("feature_major", &feature_major_buffer)
        .def_property_readonly("__array_interface__", [](Matrix &X) {
            return gafime_export::array_interface(X.data.data(),
                {static_cast<py::ssize_t>(X.n_samples), static_cast<py::ssize_t>(X.n_features)});
        })
        .def("__dlpack__", [](py::object self, py::object, py::object, py::object, py::object) {
            Matrix &X = self.cast<Matrix &>();
            return gafime_export::make_dlpack(self, X.data.data(),
                {static_cast<int64_t>(X.n_samples), static_cast<int64_t>(X.n_features)});
        }, py::arg("stream") = py::none(), py::arg("max_version") = py::none(),
           py::arg("dl_device") = py::none(), py::arg("copy") = py::none())
        .def("__dlpack_device__", [](const Matrix &) {
            return py::make_tuple(static_cast<int>(kDLCPU), 0);
        })
        .def_buffer([](Matrix &X) {
            return py::buffer_info(
                X.data.data(),
                sizeof(real_t),
                py::format_descriptor<real_t>::format(),
                1,
                {static_cast<py::ssize_t>(X.data.size())},
                {static_cast<py::ssize_t>(sizeof(real_t))});
        });

    py::class_<NativeReportTable>(m, "NativeReportTable")
        .def(py::init<>())
        .def("__len__", &NativeReportTable::size)
        .def("interned_string_count", &NativeReportTable::interned_string_count)
        .def(
            "append",
            &NativeReportTable::append,
            py::arg("combo"),
            py::arg("feature_names"),
            py::arg("metric_names"),
            py::arg("metric_values"),
            py::arg("secondary_metric_names"),
            py::arg("secondary_metric_values"),
            py::arg("family"),
            py::arg("expression"),
            py::arg("params"),
            py::arg("candidate_id"))
        .def("combo", [](const NativeReportTable &table, std::size_t index) {
            return table.at(index).combo;
        })
        .def("feature_names", [](const NativeReportTable &table, std::size_t index) {
            return table.feature_names(index);
        })
        .def("metric_names", [](const NativeReportTable &table, std::size_t index) {
            return table.metric_names(index);
        })
        .def("metric_values", [](const NativeReportTable &table, std::size_t index) {
            return table.at(index).metric_values;
        })
        .def("secondary_metric_names", [](const NativeReportTable &table, std::size_t index) {
            return table.secondary_metric_names(index);
        })
        .def("secondary_metric_values", [](const NativeReportTable &table, std::size_t index) {
            return table.at(index).secondary_metric_values;
        })
        .def("family", [](const NativeReportTable &table, std::size_t index) {
            return table.family(index);
        })
        .def("expression", [](const NativeReportTable &table, std::size_t index) {
            return table.expression(index);
        })
        .def("params", [](const NativeReportTable &table, std::size_t index) {
            return table.params_at(index);
        })
        .def("candidate_id", [](const NativeReportTable &table, std::size_t index) {
            return table.candidate_id(index);
        });

    py::enum_<MetricId>(m, "MetricId")
        .value("Pearson", MetricId::Pearson)
        .value("Spearman", MetricId::Spearman)
        .value("MutualInfo", MetricId::MutualInfo)
        .value("R2", MetricId::R2)
        .export_values();

    m.def("pack_combos", &pack_combos, "Pack combos into flat native index and offset lists.");
    m.def("interaction_matrix", &interaction_matrix, py::arg("X"), py::arg("combos"));
    m.def(
        "score_combos_buffer",
        &score_combos_buffer,
        py::arg("X"),
        py::arg("y"),
        py::arg("combos"),
        py::arg("metric_ids") = py::none(),
        py::arg("mi_bins") = 96,
        "Compute metrics directly from GAFIME native fp32 buffers.");
    m.def(
        "ridge_baseline_prediction",
        &ridge_baseline_prediction,
        py::arg("X"),
        py::arg("y"),
        py::arg("combos"),
        py::arg("alpha") = 1.0,
        "Native ridge baseline predictions for split-aware discrete ranking.");
    py::class_<ContinuousMetricCache>(m, "ContinuousMetricCache")
        .def_property_readonly("bytes", [](const ContinuousMetricCache &c) { return c.bytes; })
        .def_property_readonly("n_samples", [](const ContinuousMetricCache &c) { return c.n; })
        .def("combos", [](const ContinuousMetricCache &c) { return c.combos; });
    m.def(
        "build_continuous_metric_cache",
        &build_continuous_metric_cache,
        py::arg("X"),
        py::arg("combos"),
        py::arg("metric_ids") = py::none(),
        py::arg("mi_bins") = 96,
        py::arg("max_bytes") = static_cast<std::size_t>(0),
        "Build the X-invariant continuous metric cache (residency v2); returns None if over budget.");
    m.def(
        "score_continuous_metric_cache",
        &score_continuous_metric_cache,
        py::arg("cache"),
        py::arg("y"),
        "Score cached continuous combos against y, recomputing only y-side work.");
    m.def(
        "build_time_series_metric_cache",
        &build_time_series_metric_cache,
        py::arg("X"),
        py::arg("kinds"),
        py::arg("feats"),
        py::arg("lags"),
        py::arg("windows"),
        py::arg("metric_ids") = py::none(),
        py::arg("mi_bins") = 96,
        py::arg("max_bytes") = static_cast<std::size_t>(0),
        "Build the X-invariant time-series metric cache (residency v2); scored via "
        "score_continuous_metric_cache. Returns None if over budget.");
    py::class_<DecisionPathRecord>(m, "DecisionPathRecord")
        .def_property_readonly("features", [](const DecisionPathRecord &r) { return r.features; })
        .def_property_readonly("thresholds", [](const DecisionPathRecord &r) { return r.thresholds; })
        .def_property_readonly("signs", [](const DecisionPathRecord &r) { return r.signs; })
        .def_property_readonly("gain", [](const DecisionPathRecord &r) { return r.gain; })
        .def_property_readonly("support", [](const DecisionPathRecord &r) { return r.support; })
        .def_property_readonly("round_id", [](const DecisionPathRecord &r) { return r.round_id; })
        .def_property_readonly("candidate_id", [](const DecisionPathRecord &r) { return r.candidate_id; });
    m.def(
        "find_decision_path_candidates",
        &find_decision_path_candidates,
        py::arg("X"),
        py::arg("y"),
        py::arg("feature_ids") = py::none(),
        py::arg("max_depth") = 1,
        py::arg("max_paths") = 32,
        py::arg("max_bins_per_feature") = 0,
        py::arg("min_leaf") = 8,
        py::arg("rounds") = 1,
        py::arg("learning_rate") = 1.0,
        "Native whole-data GBDT split finder (decision_path family): greedy variance-reduction "
        "depth-k trees + residual boosting; returns root->leaf conjunction paths. No subsampling.");
    m.def("cpu_dispatch_target", &gafime_cpu_dispatch_target, "Runtime CPU SIMD dispatch target.");
    m.def(
        "available_cpu_dispatch_targets",
        &gafime_available_cpu_dispatch_targets,
        "CPU SIMD dispatch targets available on this host.");
    m.def("precision_name", &precision_name, "Native C++ Core real_t precision.");
}
