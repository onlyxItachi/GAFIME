#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

#if defined(_MSC_VER)
#define GAFIME_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define GAFIME_FORCEINLINE __attribute__((always_inline)) inline
#else
#define GAFIME_FORCEINLINE inline
#endif

enum class MetricId : int {
    Pearson = 0,
    Spearman = 1,
    MutualInfo = 2,
    R2 = 3,
};

template <typename T>
struct Span2D {
    std::span<T> data;
    std::size_t width = 0;

    [[nodiscard]] GAFIME_FORCEINLINE T &operator()(std::size_t row, std::size_t col) const {
#ifndef NDEBUG
        assert(width > 0);
        assert(row * width + col < data.size());
#endif
        return data[row * width + col];
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

void validate_combo_buffers(
    const py::buffer_info &indices_info,
    const py::buffer_info &offsets_info,
    std::size_t n_features) {
    if (indices_info.ndim != 1 || offsets_info.ndim != 1) {
        throw std::invalid_argument("combo_indices and combo_offsets must be 1D arrays");
    }
    if (offsets_info.shape[0] < 1) {
        throw std::invalid_argument("combo_offsets must have at least one entry");
    }
    const auto *offsets = static_cast<const std::int64_t *>(offsets_info.ptr);
    std::size_t n_offsets = static_cast<std::size_t>(offsets_info.shape[0]);
    std::size_t n_indices = static_cast<std::size_t>(indices_info.shape[0]);

    if (offsets[0] != 0) {
        throw std::invalid_argument("combo_offsets must start at 0");
    }
    for (std::size_t i = 1; i < n_offsets; ++i) {
        if (offsets[i] < offsets[i - 1]) {
            throw std::invalid_argument("combo_offsets must be non-decreasing");
        }
    }
    if (static_cast<std::size_t>(offsets[n_offsets - 1]) > n_indices) {
        throw std::invalid_argument("combo_offsets exceed combo_indices length");
    }
    const auto *indices = static_cast<const std::int64_t *>(indices_info.ptr);
    for (std::size_t i = 0; i < n_indices; ++i) {
        if (indices[i] < 0 || static_cast<std::size_t>(indices[i]) >= n_features) {
            throw std::invalid_argument("combo index out of feature bounds");
        }
    }
}

std::vector<double> compute_means(
    Span2D<const double> X,
    std::size_t n_samples,
    std::size_t n_features) {
    std::vector<double> means(n_features, 0.0);
    if (n_samples == 0 || n_features == 0) {
        return means;
    }
    for (std::size_t col = 0; col < n_features; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < n_samples; ++row) {
            sum += X(row, col);
        }
        means[col] = sum / static_cast<double>(n_samples);
    }
    return means;
}

struct RankScratch {
    std::vector<std::size_t> indices;
};

void rankdata(std::span<const double> data, RankScratch &scratch, std::vector<double> &out) {
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
        double avg_rank = 0.5 * static_cast<double>(i + j - 1);
        for (std::size_t k = i; k < j; ++k) {
            out[scratch.indices[k]] = avg_rank;
        }
        i = j;
    }
}

double pearson_from_sums(
    double sum_x,
    double sum_x2,
    double dot_xy,
    double var_y,
    std::size_t n) {
    if (n == 0) {
        return 0.0;
    }
    double mean_x = sum_x / static_cast<double>(n);
    double var_x = sum_x2 - (sum_x * mean_x);
    if (var_x <= 0.0 || var_y <= 0.0) {
        return 0.0;
    }
    double denom = std::sqrt(var_x * var_y);
    if (denom == 0.0) {
        return 0.0;
    }
    return dot_xy / denom;
}

bool build_bins(std::span<const double> values, int bins, std::vector<int> &out_bins) {
    std::size_t n = values.size();
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
    out_bins.resize(n);
    double scale = static_cast<double>(bins) / (vmax - vmin);
    for (std::size_t i = 0; i < n; ++i) {
        int bin = static_cast<int>((values[i] - vmin) * scale);
        if (bin < 0) {
            bin = 0;
        } else if (bin >= bins) {
            bin = bins - 1;
        }
        out_bins[i] = bin;
    }
    return true;
}

double mutual_info_from_vector(
    std::span<const double> x,
    int bins,
    std::span<const int> y_bins,
    struct MiScratch &scratch);

constexpr int kSmallMiBins = 32;

struct MiScratch {
    std::array<double, static_cast<std::size_t>(kSmallMiBins) * static_cast<std::size_t>(kSmallMiBins)>
        joint_small{};
    std::array<double, static_cast<std::size_t>(kSmallMiBins)> p_x_small{};
    std::array<double, static_cast<std::size_t>(kSmallMiBins)> p_y_small{};
    std::vector<double> joint;
    std::vector<double> p_x;
    std::vector<double> p_y;

    void ensure(int bins) {
        if (bins > kSmallMiBins) {
            std::size_t size = static_cast<std::size_t>(bins);
            joint.resize(size * size);
            p_x.resize(size);
            p_y.resize(size);
        }
    }

    double *joint_ptr(int bins) {
        return (bins <= kSmallMiBins) ? joint_small.data() : joint.data();
    }

    double *p_x_ptr(int bins) {
        return (bins <= kSmallMiBins) ? p_x_small.data() : p_x.data();
    }

    double *p_y_ptr(int bins) {
        return (bins <= kSmallMiBins) ? p_y_small.data() : p_y.data();
    }
};

double mutual_info_from_vector(
    std::span<const double> x,
    int bins,
    std::span<const int> y_bins,
    MiScratch &scratch) {
    std::size_t n = x.size();
    if (bins < 2 || n == 0 || y_bins.size() != n) {
        return 0.0;
    }

    double xmin = x[0];
    double xmax = x[0];
    for (std::size_t i = 1; i < n; ++i) {
        xmin = std::min(xmin, x[i]);
        xmax = std::max(xmax, x[i]);
    }
    if (xmin == xmax) {
        return 0.0;
    }

    scratch.ensure(bins);
    double *joint = scratch.joint_ptr(bins);
    double *p_x = scratch.p_x_ptr(bins);
    double *p_y = scratch.p_y_ptr(bins);

    std::size_t bins_size = static_cast<std::size_t>(bins);
    std::fill_n(joint, bins_size * bins_size, 0.0);
    std::fill_n(p_x, bins_size, 0.0);
    std::fill_n(p_y, bins_size, 0.0);

    double scale = static_cast<double>(bins) / (xmax - xmin);
    for (std::size_t i = 0; i < n; ++i) {
        int x_bin = static_cast<int>((x[i] - xmin) * scale);
        if (x_bin < 0) {
            x_bin = 0;
        } else if (x_bin >= bins) {
            x_bin = bins - 1;
        }
        int y_bin = y_bins[i];
#ifndef NDEBUG
        assert(y_bin >= 0 && y_bin < bins);
#endif
        std::size_t x_idx = static_cast<std::size_t>(x_bin);
        std::size_t y_idx = static_cast<std::size_t>(y_bin);
        joint[x_idx * bins_size + y_idx] += 1.0;
        p_x[x_idx] += 1.0;
        p_y[y_idx] += 1.0;
    }

    double inv_total = 1.0 / static_cast<double>(n);
    double inv_total_sq = inv_total * inv_total;
    double mi = 0.0;
    for (std::size_t bx = 0; bx < bins_size; ++bx) {
        for (std::size_t by = 0; by < bins_size; ++by) {
            double count = joint[bx * bins_size + by];
            if (count <= 0.0) {
                continue;
            }
            double p = count * inv_total;
            double expected = (p_x[bx] * p_y[by]) * inv_total_sq;
            if (expected > 0.0) {
                mi += p * std::log(p / expected);
            }
        }
    }
    return mi;
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

    py::array_t<std::int64_t> indices_arr(indices.size());
    py::array_t<std::int64_t> offsets_arr(offsets.size());
    auto indices_mut = indices_arr.mutable_unchecked<1>();
    auto offsets_mut = offsets_arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices_mut(static_cast<py::ssize_t>(i)) = indices[i];
    }
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        offsets_mut(static_cast<py::ssize_t>(i)) = offsets[i];
    }

    return py::make_tuple(indices_arr, offsets_arr);
}

py::array_t<double> interaction_matrix(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_in,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &combo_indices,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &combo_offsets) {
    auto X_info = X_in.request();
    if (X_info.ndim != 2) {
        throw std::invalid_argument("X must be a 2D array");
    }
    std::size_t n_samples = static_cast<std::size_t>(X_info.shape[0]);
    std::size_t n_features = static_cast<std::size_t>(X_info.shape[1]);

    auto idx_info = combo_indices.request();
    auto off_info = combo_offsets.request();
    validate_combo_buffers(idx_info, off_info, n_features);

    std::size_t n_combos = static_cast<std::size_t>(off_info.shape[0] - 1);
    py::array_t<double> output({static_cast<py::ssize_t>(n_samples),
                                static_cast<py::ssize_t>(n_combos)});
    auto out = output.mutable_unchecked<2>();

    const auto *X = static_cast<const double *>(X_info.ptr);
    const auto *indices = static_cast<const std::int64_t *>(idx_info.ptr);
    const auto *offsets = static_cast<const std::int64_t *>(off_info.ptr);

    std::span<const double> X_span(X, n_samples * n_features);
    Span2D<const double> X_view{X_span, n_features};

    std::vector<double> means;
    {
        py::gil_scoped_release release;
        means = compute_means(X_view, n_samples, n_features);

        for (std::size_t combo_idx = 0; combo_idx < n_combos; ++combo_idx) {
            std::size_t start = static_cast<std::size_t>(offsets[combo_idx]);
            std::size_t end = static_cast<std::size_t>(offsets[combo_idx + 1]);
            if (start >= end) {
                throw std::invalid_argument("combination entries must be non-empty");
            }
            std::size_t k = end - start;
            if (k == 1) {
                std::size_t feature = static_cast<std::size_t>(indices[start]);
                for (std::size_t row = 0; row < n_samples; ++row) {
                    out(static_cast<py::ssize_t>(row), static_cast<py::ssize_t>(combo_idx)) =
                        X_view(row, feature);
                }
            } else {
                for (std::size_t row = 0; row < n_samples; ++row) {
                    double prod = 1.0;
                    for (std::size_t j = start; j < end; ++j) {
                        std::size_t feature = static_cast<std::size_t>(indices[j]);
                        prod *= (X_view(row, feature) - means[feature]);
                    }
                    out(static_cast<py::ssize_t>(row), static_cast<py::ssize_t>(combo_idx)) = prod;
                }
            }
        }
    }

    return output;
}

py::array_t<double> score_combos(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &y_in,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &combo_indices,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &combo_offsets,
    const py::object &metric_ids,
    int mi_bins) {
    auto X_info = X_in.request();
    auto y_info = y_in.request();
    if (X_info.ndim != 2) {
        throw std::invalid_argument("X must be a 2D array");
    }
    if (y_info.ndim != 1) {
        throw std::invalid_argument("y must be a 1D array");
    }
    std::size_t n_samples = static_cast<std::size_t>(X_info.shape[0]);
    std::size_t n_features = static_cast<std::size_t>(X_info.shape[1]);
    if (static_cast<std::size_t>(y_info.shape[0]) != n_samples) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }

    auto idx_info = combo_indices.request();
    auto off_info = combo_offsets.request();
    validate_combo_buffers(idx_info, off_info, n_features);

    const auto *X = static_cast<const double *>(X_info.ptr);
    const auto *y = static_cast<const double *>(y_info.ptr);
    const auto *indices = static_cast<const std::int64_t *>(idx_info.ptr);
    const auto *offsets = static_cast<const std::int64_t *>(off_info.ptr);

    std::span<const double> X_span(X, n_samples * n_features);
    Span2D<const double> X_view{X_span, n_features};
    std::span<const double> y_span(y, n_samples);

    std::vector<int> metrics = parse_metric_ids(metric_ids);
    std::size_t n_metrics = metrics.size();
    std::size_t n_combos = static_cast<std::size_t>(off_info.shape[0] - 1);

    py::array_t<double> output({static_cast<py::ssize_t>(n_combos),
                                static_cast<py::ssize_t>(n_metrics)});
    auto out = output.mutable_unchecked<2>();

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

    std::vector<double> means = compute_means(X_view, n_samples, n_features);

    double sum_y = 0.0;
    double sum_y2 = 0.0;
    std::vector<double> y_centered;
    double var_y = 0.0;
    if (need_pearson) {
        y_centered.resize(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            sum_y += y_span[i];
            sum_y2 += y_span[i] * y_span[i];
        }
        double mean_y = (n_samples == 0) ? 0.0 : (sum_y / static_cast<double>(n_samples));
        for (std::size_t i = 0; i < n_samples; ++i) {
            y_centered[i] = y_span[i] - mean_y;
        }
        var_y = sum_y2 - (sum_y * mean_y);
    }

    std::vector<double> y_rank;
    std::vector<double> y_rank_centered;
    RankScratch rank_scratch;
    double var_y_rank = 0.0;
    if (need_spearman) {
        rankdata(y_span, rank_scratch, y_rank);
        y_rank_centered.resize(n_samples);
        double sum_r = 0.0;
        double sum_r2 = 0.0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            sum_r += y_rank[i];
            sum_r2 += y_rank[i] * y_rank[i];
        }
        double mean_r = (n_samples == 0) ? 0.0 : (sum_r / static_cast<double>(n_samples));
        for (std::size_t i = 0; i < n_samples; ++i) {
            y_rank_centered[i] = y_rank[i] - mean_r;
        }
        var_y_rank = sum_r2 - (sum_r * mean_r);
    }

    std::vector<int> y_bins;
    bool mi_ready = false;
    if (need_mi) {
        mi_ready = build_bins(y_span, mi_bins, y_bins);
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
            std::vector<double> interaction(n_samples);
            std::span<const double> interaction_span(interaction.data(), n_samples);
            std::vector<double> x_rank;
            RankScratch thread_rank_scratch;
            MiScratch mi_scratch;
            if (need_mi && mi_bins >= 2) {
                mi_scratch.ensure(mi_bins);
            }

#ifdef GAFIME_CORE_OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (std::size_t combo_idx = 0; combo_idx < n_combos; ++combo_idx) {
                std::size_t start = static_cast<std::size_t>(offsets[combo_idx]);
                std::size_t end = static_cast<std::size_t>(offsets[combo_idx + 1]);
                if (start >= end) {
                    throw std::invalid_argument("combination entries must be non-empty");
                }

                double sum_x = 0.0;
                double sum_x2 = 0.0;
                double dot_xy = 0.0;

                std::size_t k = end - start;
                if (k == 1) {
                    std::size_t feature = static_cast<std::size_t>(indices[start]);
                    for (std::size_t row = 0; row < n_samples; ++row) {
                        double value = X_view(row, feature);
                        interaction[row] = value;
                        if (need_pearson) {
                            sum_x += value;
                            sum_x2 += value * value;
                            dot_xy += value * y_centered[row];
                        }
                    }
                } else {
                    for (std::size_t row = 0; row < n_samples; ++row) {
                        double value = 1.0;
                        for (std::size_t j = start; j < end; ++j) {
                            std::size_t feature = static_cast<std::size_t>(indices[j]);
                            value *= (X_view(row, feature) - means[feature]);
                        }
                        interaction[row] = value;
                        if (need_pearson) {
                            sum_x += value;
                            sum_x2 += value * value;
                            dot_xy += value * y_centered[row];
                        }
                    }
                }

                for (std::size_t m = 0; m < n_metrics; ++m) {
                    int id = metrics[m];
                    if (id == static_cast<int>(MetricId::Pearson)) {
                        out(static_cast<py::ssize_t>(combo_idx), static_cast<py::ssize_t>(m)) =
                            pearson_from_sums(sum_x, sum_x2, dot_xy, var_y, n_samples);
                    } else if (id == static_cast<int>(MetricId::R2)) {
                        double corr = pearson_from_sums(sum_x, sum_x2, dot_xy, var_y, n_samples);
                        out(static_cast<py::ssize_t>(combo_idx), static_cast<py::ssize_t>(m)) =
                            corr * corr;
                    } else if (id == static_cast<int>(MetricId::Spearman)) {
                        rankdata(interaction_span, thread_rank_scratch, x_rank);
                        double sum_r = 0.0;
                        double sum_r2 = 0.0;
                        double dot_r = 0.0;
                        for (std::size_t i = 0; i < n_samples; ++i) {
                            double val = x_rank[i];
                            sum_r += val;
                            sum_r2 += val * val;
                            dot_r += val * y_rank_centered[i];
                        }
                        double corr = pearson_from_sums(sum_r, sum_r2, dot_r, var_y_rank, n_samples);
                        out(static_cast<py::ssize_t>(combo_idx), static_cast<py::ssize_t>(m)) = corr;
                    } else if (id == static_cast<int>(MetricId::MutualInfo)) {
                        double mi = 0.0;
                        if (mi_ready && mi_bins >= 2) {
                            mi = mutual_info_from_vector(interaction_span, mi_bins, y_bins_span, mi_scratch);
                        }
                        out(static_cast<py::ssize_t>(combo_idx), static_cast<py::ssize_t>(m)) = mi;
                    } else {
                        throw std::invalid_argument("unknown metric id");
                    }
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(gafime_core, m) {
    m.doc() = "C++23 kernels for GAFIME interaction and metric scoring.";

    py::enum_<MetricId>(m, "MetricId")
        .value("Pearson", MetricId::Pearson)
        .value("Spearman", MetricId::Spearman)
        .value("MutualInfo", MetricId::MutualInfo)
        .value("R2", MetricId::R2)
        .export_values();

    m.def("pack_combos", &pack_combos, "Pack combos into flat indices and offsets.");
    m.def(
        "interaction_matrix",
        &interaction_matrix,
        py::arg("X"),
        py::arg("combo_indices"),
        py::arg("combo_offsets"),
        "Build interaction vectors for a batch of combos.");
    m.def(
        "score_combos",
        &score_combos,
        py::arg("X"),
        py::arg("y"),
        py::arg("combo_indices"),
        py::arg("combo_offsets"),
        py::arg("metric_ids") = py::none(),
        py::arg("mi_bins") = 16,
        "Compute metrics for a batch of combos in a single call.");
}
