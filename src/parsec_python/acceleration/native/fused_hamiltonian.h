#pragma once

#include "finite_difference.h"

#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace parsec_accelerated_native {

class FusedHamiltonian {
public:
    FusedHamiltonian(
        const IndexArray& a_indptr,
        const IndexArray& a_indices,
        const FloatArray& a_data,
        const IndexArray& b_indptr,
        const IndexArray& b_indices,
        const FloatArray& b_data,
        const FloatArray& signs,
        const FloatArray& local_potential
    );

    void update_local(const FloatArray& local_potential);
    py::array apply(const FloatArray& vectors) const;

    [[nodiscard]] std::pair<std::int64_t, std::int64_t> shape() const noexcept;
    [[nodiscard]] std::int64_t size() const noexcept;
    [[nodiscard]] std::int64_t projector_count() const noexcept;

private:
    std::int64_t size_ = 0;
    std::int64_t projector_count_ = 0;
    std::vector<std::int64_t> a_indptr_;
    std::vector<std::int64_t> a_indices_;
    std::vector<double> a_data_;
    std::vector<std::int64_t> b_indptr_;
    std::vector<std::int64_t> b_indices_;
    std::vector<double> b_data_;
    std::vector<double> signs_;
    std::vector<std::int64_t> b_row_indptr_;
    std::vector<std::int64_t> b_row_projectors_;
    std::vector<double> b_row_data_;
    std::vector<double> local_potential_;
    mutable std::mutex mutex_;
};

}  // namespace parsec_accelerated_native
