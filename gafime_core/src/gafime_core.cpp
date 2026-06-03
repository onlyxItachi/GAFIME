#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gafime_native_data.h"
#include "gafime_real.h"
#include "gafime_simd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
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

std::string precision_name() {
#ifdef GAFIME_USE_DOUBLE_PRECISION
    return "float64";
#else
    return "float32";
#endif
}

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
        .def_buffer([](Matrix &X) {
            return py::buffer_info(
                X.data.data(),
                sizeof(real_t),
                py::format_descriptor<real_t>::format(),
                1,
                {static_cast<py::ssize_t>(X.data.size())},
                {static_cast<py::ssize_t>(sizeof(real_t))});
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
    m.def("cpu_dispatch_target", &gafime_cpu_dispatch_target, "Runtime CPU SIMD dispatch target.");
    m.def(
        "available_cpu_dispatch_targets",
        &gafime_available_cpu_dispatch_targets,
        "CPU SIMD dispatch targets available on this host.");
    m.def("precision_name", &precision_name, "Native C++ Core real_t precision.");
}
