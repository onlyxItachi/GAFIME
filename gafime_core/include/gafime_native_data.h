#pragma once

#include "gafime_real.h"

#include <cstddef>
#include <vector>

struct Matrix {
    std::vector<real_t> data;
    std::size_t n_samples = 0;
    std::size_t n_features = 0;

    [[nodiscard]] inline real_t value(std::size_t row, std::size_t col) const {
        return data[row * n_features + col];
    }

    [[nodiscard]] inline std::size_t size() const {
        return data.size();
    }

    [[nodiscard]] inline std::size_t nbytes() const {
        return data.size() * sizeof(real_t);
    }
};

struct Vector {
    std::vector<real_t> data;

    [[nodiscard]] inline std::size_t size() const {
        return data.size();
    }

    [[nodiscard]] inline std::size_t nbytes() const {
        return data.size() * sizeof(real_t);
    }
};
