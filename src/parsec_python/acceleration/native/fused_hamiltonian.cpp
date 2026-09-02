#include "fused_hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace parsec_accelerated_native {
namespace {

template <typename T, int Flags>
std::vector<T> copy_vector(const py::array_t<T, Flags>& array, const char* name) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    }
    const auto count = static_cast<std::size_t>(array.shape(0));
    return std::vector<T>(array.data(), array.data() + count);
}

void validate_indptr(
    const std::vector<std::int64_t>& indptr,
    const std::size_t data_size,
    const char* name
) {
    if (indptr.empty() || indptr.front() != 0) {
        throw std::invalid_argument(std::string(name) + " must start at zero");
    }
    for (std::size_t index = 1; index < indptr.size(); ++index) {
        if (indptr[index] < indptr[index - 1]) {
            throw std::invalid_argument(
                std::string(name) + " must be monotonically nondecreasing"
            );
        }
    }
    if (
        indptr.back() < 0 ||
        static_cast<std::size_t>(indptr.back()) != data_size
    ) {
        throw std::invalid_argument(
            std::string(name) + " final entry must equal the number of values"
        );
    }
}

void validate_finite(const std::vector<double>& values, const char* name) {
    if (!std::all_of(values.begin(), values.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(
            std::string(name) + " must contain only finite values"
        );
    }
}

}  // namespace

FusedHamiltonian::FusedHamiltonian(
    const IndexArray& a_indptr,
    const IndexArray& a_indices,
    const FloatArray& a_data,
    const IndexArray& b_indptr,
    const IndexArray& b_indices,
    const FloatArray& b_data,
    const FloatArray& signs,
    const FloatArray& local_potential
) :
    a_indptr_(copy_vector(a_indptr, "a_indptr")),
    a_indices_(copy_vector(a_indices, "a_indices")),
    a_data_(copy_vector(a_data, "a_data")),
    b_indptr_(copy_vector(b_indptr, "b_indptr")),
    b_indices_(copy_vector(b_indices, "b_indices")),
    b_data_(copy_vector(b_data, "b_data")),
    signs_(copy_vector(signs, "signs")),
    local_potential_(copy_vector(local_potential, "local_potential")) {
    if (a_indptr_.size() < 2) {
        throw std::invalid_argument("a_indptr must describe a nonempty matrix");
    }
    size_ = static_cast<std::int64_t>(a_indptr_.size() - 1);
    if (a_indices_.size() != a_data_.size()) {
        throw std::invalid_argument("A indices and values must have equal lengths");
    }
    validate_indptr(a_indptr_, a_data_.size(), "a_indptr");
    for (const std::int64_t column : a_indices_) {
        if (column < 0 || column >= size_) {
            throw std::invalid_argument("A contains a column outside its shape");
        }
    }

    if (b_indptr_.empty()) {
        throw std::invalid_argument("b_indptr must contain at least one entry");
    }
    projector_count_ = static_cast<std::int64_t>(b_indptr_.size() - 1);
    if (b_indices_.size() != b_data_.size()) {
        throw std::invalid_argument("B indices and values must have equal lengths");
    }
    validate_indptr(b_indptr_, b_data_.size(), "b_indptr");
    if (signs_.size() != static_cast<std::size_t>(projector_count_)) {
        throw std::invalid_argument("one sign is required per projector column");
    }
    if (local_potential_.size() != static_cast<std::size_t>(size_)) {
        throw std::invalid_argument("local potential length must match A");
    }
    for (std::int64_t projector = 0; projector < projector_count_; ++projector) {
        std::int64_t previous_row = -1;
        for (
            std::int64_t offset = b_indptr_[static_cast<std::size_t>(projector)];
            offset < b_indptr_[static_cast<std::size_t>(projector + 1)];
            ++offset
        ) {
            const std::int64_t row = b_indices_[static_cast<std::size_t>(offset)];
            if (row < 0 || row >= size_) {
                throw std::invalid_argument("B contains a row outside its shape");
            }
            if (row <= previous_row) {
                throw std::invalid_argument(
                    "each B projector column must have sorted, unique row indices"
                );
            }
            previous_row = row;
        }
    }
    validate_finite(a_data_, "a_data");
    validate_finite(b_data_, "b_data");
    validate_finite(signs_, "signs");
    validate_finite(local_potential_, "local_potential");

    // Construct a deterministic CSR-like view of B.  Traversing projector
    // columns in ascending order makes every row later accumulate q=0,1,...
    // regardless of the number of OpenMP threads used during apply().
    b_row_indptr_.assign(static_cast<std::size_t>(size_ + 1), 0);
    for (const std::int64_t row : b_indices_) {
        ++b_row_indptr_[static_cast<std::size_t>(row + 1)];
    }
    for (std::int64_t row = 0; row < size_; ++row) {
        b_row_indptr_[static_cast<std::size_t>(row + 1)] +=
            b_row_indptr_[static_cast<std::size_t>(row)];
    }
    b_row_projectors_.resize(b_data_.size());
    b_row_data_.resize(b_data_.size());
    std::vector<std::int64_t> cursors = b_row_indptr_;
    for (std::int64_t projector = 0; projector < projector_count_; ++projector) {
        for (
            std::int64_t offset = b_indptr_[static_cast<std::size_t>(projector)];
            offset < b_indptr_[static_cast<std::size_t>(projector + 1)];
            ++offset
        ) {
            const auto source = static_cast<std::size_t>(offset);
            const std::int64_t row = b_indices_[source];
            const auto destination = static_cast<std::size_t>(
                cursors[static_cast<std::size_t>(row)]++
            );
            b_row_projectors_[destination] = projector;
            b_row_data_[destination] = b_data_[source];
        }
    }
}

void FusedHamiltonian::update_local(const FloatArray& local_potential) {
    auto replacement = copy_vector(local_potential, "local_potential");
    if (replacement.size() != static_cast<std::size_t>(size_)) {
        throw std::invalid_argument("local potential length must match A");
    }
    validate_finite(replacement, "local_potential");
    std::lock_guard<std::mutex> guard(mutex_);
    local_potential_.swap(replacement);
}

py::array FusedHamiltonian::apply(const FloatArray& vectors) const {
    if (vectors.ndim() != 1 && vectors.ndim() != 2) {
        throw std::invalid_argument("vectors must have shape (n,) or (n, block_size)");
    }
    if (vectors.shape(0) != size_) {
        throw std::invalid_argument("vector row count must match the Hamiltonian");
    }
    const std::int64_t vector_count = vectors.ndim() == 1
        ? 1
        : static_cast<std::int64_t>(vectors.shape(1));
    if (vector_count < 1) {
        throw std::invalid_argument("a vector block must contain at least one column");
    }

    py::array_t<double> result;
    if (vectors.ndim() == 1) {
        result = py::array_t<double>(static_cast<py::ssize_t>(size_));
    } else {
        result = py::array_t<double>({
            static_cast<py::ssize_t>(size_),
            static_cast<py::ssize_t>(vector_count),
        });
    }

    const double* input = vectors.data();
    double* output = result.mutable_data();
    std::vector<double> coefficients(
        static_cast<std::size_t>(projector_count_ * vector_count),
        0.0
    );

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> guard(mutex_);

        // Compute A@Q first and add diag(V)@Q second, matching the reference
        // Hamiltonian's binary addition order.  Each CSR sum is serial.
#pragma omp parallel for schedule(static) if(size_ * vector_count >= 4096)
        for (std::int64_t row = 0; row < size_; ++row) {
            for (std::int64_t vector = 0; vector < vector_count; ++vector) {
                double value = 0.0;
                for (
                    std::int64_t offset = a_indptr_[static_cast<std::size_t>(row)];
                    offset < a_indptr_[static_cast<std::size_t>(row + 1)];
                    ++offset
                ) {
                    const auto position = static_cast<std::size_t>(offset);
                    const std::int64_t column = a_indices_[position];
                    value += a_data_[position] * input[
                        static_cast<std::size_t>(column * vector_count + vector)
                    ];
                }
                const auto output_index = static_cast<std::size_t>(
                    row * vector_count + vector
                );
                output[output_index] = value + local_potential_[
                    static_cast<std::size_t>(row)
                ] * input[output_index];
            }
        }

        // Each overlap is independent; its inner reduction follows CSC row
        // order and therefore does not depend on the OpenMP thread count.
#pragma omp parallel for schedule(static) if(projector_count_ * vector_count >= 128)
        for (std::int64_t projector = 0; projector < projector_count_; ++projector) {
            for (std::int64_t vector = 0; vector < vector_count; ++vector) {
                double overlap = 0.0;
                for (
                    std::int64_t offset = b_indptr_[static_cast<std::size_t>(projector)];
                    offset < b_indptr_[static_cast<std::size_t>(projector + 1)];
                    ++offset
                ) {
                    const auto position = static_cast<std::size_t>(offset);
                    const std::int64_t row = b_indices_[position];
                    overlap += b_data_[position] * input[
                        static_cast<std::size_t>(row * vector_count + vector)
                    ];
                }
                coefficients[static_cast<std::size_t>(
                    projector * vector_count + vector
                )] = signs_[static_cast<std::size_t>(projector)] * overlap;
            }
        }

        // Rows are independent and each nonlocal sum visits projector columns
        // in ascending order.  Add the complete V_NL contribution once.
#pragma omp parallel for schedule(static) if(size_ * vector_count >= 4096)
        for (std::int64_t row = 0; row < size_; ++row) {
            for (std::int64_t vector = 0; vector < vector_count; ++vector) {
                double nonlocal_value = 0.0;
                for (
                    std::int64_t offset = b_row_indptr_[static_cast<std::size_t>(row)];
                    offset < b_row_indptr_[static_cast<std::size_t>(row + 1)];
                    ++offset
                ) {
                    const auto position = static_cast<std::size_t>(offset);
                    const std::int64_t projector = b_row_projectors_[position];
                    nonlocal_value += b_row_data_[position] * coefficients[
                        static_cast<std::size_t>(
                            projector * vector_count + vector
                        )
                    ];
                }
                output[static_cast<std::size_t>(row * vector_count + vector)] +=
                    nonlocal_value;
            }
        }
    }
    return result;
}

std::pair<std::int64_t, std::int64_t> FusedHamiltonian::shape() const noexcept {
    return {size_, size_};
}

std::int64_t FusedHamiltonian::size() const noexcept {
    return size_;
}

std::int64_t FusedHamiltonian::projector_count() const noexcept {
    return projector_count_;
}

}  // namespace parsec_accelerated_native
