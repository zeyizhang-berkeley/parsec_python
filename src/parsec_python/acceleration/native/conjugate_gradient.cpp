#include "conjugate_gradient.h"
#include "openmp_workload.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

constexpr std::size_t kReductionBlockSize = 4096;
std::size_t reduction_block_count(std::size_t count) noexcept {
    return (count + kReductionBlockSize - 1) / kReductionBlockSize;
}

double merge_reduction_blocks(const std::vector<double>& partial) noexcept {
    double result = 0.0;
    for (const double value : partial) {
        result += value;
    }
    return result;
}

double deterministic_parallel_dot(
    const double* left,
    const double* right,
    std::size_t count,
    std::vector<double>& partial
) noexcept {
    const std::size_t block_count = reduction_block_count(count);
#pragma omp parallel for schedule(static) if(count >= kReductionBlockSize) \
    num_threads(grid_vector_worker_count(count))
    for (std::int64_t block = 0;
         block < static_cast<std::int64_t>(block_count); ++block) {
        const std::size_t start =
            static_cast<std::size_t>(block) * kReductionBlockSize;
        const std::size_t stop = std::min(count, start + kReductionBlockSize);
        double value = 0.0;
        for (std::size_t index = start; index < stop; ++index) {
            value += left[index] * right[index];
        }
        partial[static_cast<std::size_t>(block)] = value;
    }
    // A fixed block layout and serial block-order merge make the result
    // independent of the number of OpenMP workers.
    return merge_reduction_blocks(partial);
}

double residual_and_dot(
    const double* right_hand_side,
    const double* operator_vector,
    double* residual,
    std::size_t count,
    std::vector<double>& partial
) noexcept {
    const std::size_t block_count = reduction_block_count(count);
#pragma omp parallel for schedule(static) if(count >= kReductionBlockSize) \
    num_threads(grid_vector_worker_count(count))
    for (std::int64_t block = 0;
         block < static_cast<std::int64_t>(block_count); ++block) {
        const std::size_t start =
            static_cast<std::size_t>(block) * kReductionBlockSize;
        const std::size_t stop = std::min(count, start + kReductionBlockSize);
        double squared = 0.0;
        for (std::size_t index = start; index < stop; ++index) {
            const double value =
                right_hand_side[index] - operator_vector[index];
            residual[index] = value;
            squared += value * value;
        }
        partial[static_cast<std::size_t>(block)] = squared;
    }
    return merge_reduction_blocks(partial);
}

double update_solution_residual_and_dot(
    double* solution,
    double* residual,
    const double* direction,
    const double* operator_direction,
    double alpha,
    std::size_t count,
    std::vector<double>& partial
) noexcept {
    const std::size_t block_count = reduction_block_count(count);
#pragma omp parallel for schedule(static) if(count >= kReductionBlockSize) \
    num_threads(grid_vector_worker_count(count))
    for (std::int64_t block = 0;
         block < static_cast<std::int64_t>(block_count); ++block) {
        const std::size_t start =
            static_cast<std::size_t>(block) * kReductionBlockSize;
        const std::size_t stop = std::min(count, start + kReductionBlockSize);
        double squared = 0.0;
        for (std::size_t index = start; index < stop; ++index) {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * operator_direction[index];
            squared += residual[index] * residual[index];
        }
        partial[static_cast<std::size_t>(block)] = squared;
    }
    return merge_reduction_blocks(partial);
}

}  // namespace

ConjugateGradientSolver::ConjugateGradientSolver(
    const IndexArray& indptr,
    const IndexArray& indices,
    const FloatArray& data
) :
    indptr_(copy_vector(indptr, "indptr")),
    indices_(copy_vector(indices, "indices")),
    data_(copy_vector(data, "data")) {
    if (indptr_.size() < 2) {
        throw std::invalid_argument("indptr must describe a nonempty square matrix");
    }
    size_ = static_cast<std::int64_t>(indptr_.size() - 1);
    if (indices_.size() != data_.size()) {
        throw std::invalid_argument("indices and data must have equal lengths");
    }
    if (indptr_.front() != 0) {
        throw std::invalid_argument("indptr must start at zero");
    }
    for (std::size_t row = 1; row < indptr_.size(); ++row) {
        if (indptr_[row] < indptr_[row - 1]) {
            throw std::invalid_argument(
                "indptr must be monotonically nondecreasing"
            );
        }
    }
    if (
        indptr_.back() < 0 ||
        static_cast<std::size_t>(indptr_.back()) != data_.size()
    ) {
        throw std::invalid_argument(
            "the final indptr entry must equal the number of values"
        );
    }
    for (const std::int64_t column : indices_) {
        if (column < 0 || column >= size_) {
            throw std::invalid_argument("CSR column index lies outside the matrix");
        }
    }
    if (!std::all_of(data_.begin(), data_.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("CSR data must contain only finite values");
    }

    // A centered finite-difference Poisson matrix contains only one center
    // value plus one value per stencil shell.  Retain canonical CSR order but
    // replace each repeated float64 coefficient by a one-byte palette code,
    // and use int32 column rows whenever the grid fits.  This is lossless: the
    // palette stores the original coefficient bits exactly.
    if (size_ <= std::numeric_limits<std::int32_t>::max()) {
        std::unordered_map<std::uint64_t, std::uint16_t> palette_lookup;
        palette_lookup.reserve(32);
        coefficient_codes_.reserve(data_.size());
        coefficient_palette_.reserve(32);
        bool palette_fits = true;
        for (const double value : data_) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const auto found = palette_lookup.find(bits);
            std::uint16_t code = 0;
            if (found == palette_lookup.end()) {
                if (coefficient_palette_.size() >= 256) {
                    palette_fits = false;
                    break;
                }
                code = static_cast<std::uint16_t>(
                    coefficient_palette_.size()
                );
                palette_lookup.emplace(bits, code);
                coefficient_palette_.push_back(value);
            } else {
                code = found->second;
            }
            coefficient_codes_.push_back(static_cast<std::uint8_t>(code));
        }
        if (palette_fits) {
            compact_indices_.resize(indices_.size());
#pragma omp parallel for schedule(static) if(size_ >= 4096)
            for (std::int64_t offset = 0;
                 offset < static_cast<std::int64_t>(indices_.size()); ++offset) {
                compact_indices_[static_cast<std::size_t>(offset)] =
                    static_cast<std::int32_t>(
                        indices_[static_cast<std::size_t>(offset)]
                    );
            }
            compact_storage_ = true;
            indices_.clear();
            indices_.shrink_to_fit();
            data_.clear();
            data_.shrink_to_fit();
        } else {
            coefficient_codes_.clear();
            coefficient_palette_.clear();
        }
    }

}

void ConjugateGradientSolver::matvec(
    const double* input,
    double* output
) const noexcept {
    // Rows can run concurrently, but each short stencil sum visits canonical
    // CSR columns in ascending order and is therefore deterministic.
    if (compact_storage_) {
#pragma omp parallel for schedule(static) if(size_ >= 4096) \
    num_threads(grid_vector_worker_count(static_cast<std::size_t>(size_)))
        for (std::int64_t row = 0; row < size_; ++row) {
            double value = 0.0;
            for (
                std::int64_t offset = indptr_[static_cast<std::size_t>(row)];
                offset < indptr_[static_cast<std::size_t>(row + 1)];
                ++offset
            ) {
                const auto position = static_cast<std::size_t>(offset);
                value += coefficient_palette_[coefficient_codes_[position]] *
                    input[static_cast<std::size_t>(compact_indices_[position])];
            }
            output[static_cast<std::size_t>(row)] = value;
        }
    } else {
#pragma omp parallel for schedule(static) if(size_ >= 4096) \
    num_threads(grid_vector_worker_count(static_cast<std::size_t>(size_)))
        for (std::int64_t row = 0; row < size_; ++row) {
            double value = 0.0;
            for (
                std::int64_t offset = indptr_[static_cast<std::size_t>(row)];
                offset < indptr_[static_cast<std::size_t>(row + 1)];
                ++offset
            ) {
                const auto position = static_cast<std::size_t>(offset);
                value += data_[position] * input[
                    static_cast<std::size_t>(indices_[position])
                ];
            }
            output[static_cast<std::size_t>(row)] = value;
        }
    }
}

double ConjugateGradientSolver::matvec_and_dot(
    const double* input,
    double* output,
    std::vector<double>& partial
) const noexcept {
    // Form A*p and p.T*(A*p) in one traversal.  The sparse row sum is
    // identical to matvec(), while the scalar reduction uses the same fixed
    // 4096-row blocks and serial block merge as deterministic_parallel_dot().
    // Consequently this removes one memory pass and one OpenMP region without
    // changing the floating-point summation topology used by the solver.
    const std::size_t count = static_cast<std::size_t>(size_);
    const std::size_t block_count = reduction_block_count(count);
    if (compact_storage_) {
#pragma omp parallel for schedule(static) if(count >= kReductionBlockSize) \
    num_threads(grid_vector_worker_count(count))
        for (std::int64_t block = 0;
             block < static_cast<std::int64_t>(block_count); ++block) {
            const std::size_t start =
                static_cast<std::size_t>(block) * kReductionBlockSize;
            const std::size_t stop = std::min(
                count, start + kReductionBlockSize
            );
            double dot = 0.0;
            for (std::size_t row = start; row < stop; ++row) {
                double value = 0.0;
                for (
                    std::int64_t offset = indptr_[row];
                    offset < indptr_[row + 1];
                    ++offset
                ) {
                    const auto position = static_cast<std::size_t>(offset);
                    value += coefficient_palette_[coefficient_codes_[position]] *
                        input[static_cast<std::size_t>(compact_indices_[position])];
                }
                output[row] = value;
                dot += input[row] * value;
            }
            partial[static_cast<std::size_t>(block)] = dot;
        }
    } else {
#pragma omp parallel for schedule(static) if(count >= kReductionBlockSize) \
    num_threads(grid_vector_worker_count(count))
        for (std::int64_t block = 0;
             block < static_cast<std::int64_t>(block_count); ++block) {
            const std::size_t start =
                static_cast<std::size_t>(block) * kReductionBlockSize;
            const std::size_t stop = std::min(
                count, start + kReductionBlockSize
            );
            double dot = 0.0;
            for (std::size_t row = start; row < stop; ++row) {
                double value = 0.0;
                for (
                    std::int64_t offset = indptr_[row];
                    offset < indptr_[row + 1];
                    ++offset
                ) {
                    const auto position = static_cast<std::size_t>(offset);
                    value += data_[position] * input[
                        static_cast<std::size_t>(indices_[position])
                    ];
                }
                output[row] = value;
                dot += input[row] * value;
            }
            partial[static_cast<std::size_t>(block)] = dot;
        }
    }
    return merge_reduction_blocks(partial);
}

py::dict ConjugateGradientSolver::solve(
    const FloatArray& right_hand_side,
    const FloatArray& initial,
    const double relative_tolerance,
    const double absolute_tolerance,
    const std::int64_t max_iterations
) const {
    if (
        right_hand_side.ndim() != 1 ||
        right_hand_side.shape(0) != size_
    ) {
        throw std::invalid_argument("right_hand_side must have shape (matrix_size,)");
    }
    if (initial.ndim() != 1 || initial.shape(0) != size_) {
        throw std::invalid_argument("initial must have shape (matrix_size,)");
    }
    if (!std::isfinite(relative_tolerance) || relative_tolerance <= 0.0) {
        throw std::invalid_argument("relative_tolerance must be finite and positive");
    }
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0) {
        throw std::invalid_argument("absolute_tolerance must be finite and nonnegative");
    }
    if (max_iterations < 1) {
        throw std::invalid_argument("max_iterations must be positive");
    }

    const auto count = static_cast<std::size_t>(size_);
    std::vector<double> rhs(
        right_hand_side.data(), right_hand_side.data() + count
    );
    std::vector<double> solution(initial.data(), initial.data() + count);
    std::vector<double> residual(count);
    std::vector<double> direction(count);
    std::vector<double> operator_direction(count);
    std::vector<double> reduction_scratch(reduction_block_count(count));

    bool converged = false;
    bool breakdown = false;
    std::int64_t iterations = 0;
    std::int64_t matrix_vector_products = 0;
    double initial_norm = 0.0;
    double residual_norm = 0.0;
    double tolerance = 0.0;

    {
        py::gil_scoped_release release;

        // This recurrence intentionally mirrors
        // parsec_python.Hartree.poisson._conjugate_gradient, including
        // its matrix-vector-product budget and explicit final residual.
        matvec(solution.data(), operator_direction.data());
        matrix_vector_products = 1;
        double residual_squared = residual_and_dot(
            rhs.data(),
            operator_direction.data(),
            residual.data(),
            count,
            reduction_scratch
        );
        initial_norm = std::sqrt(residual_squared);
        residual_norm = initial_norm;
        tolerance = relative_tolerance * initial_norm + absolute_tolerance;

        if (initial_norm <= tolerance) {
            converged = true;
        } else {
#pragma omp parallel for schedule(static) if(size_ >= 4096) \
    num_threads(grid_vector_worker_count(count))
            for (std::int64_t index = 0; index < size_; ++index) {
                direction[static_cast<std::size_t>(index)] =
                    residual[static_cast<std::size_t>(index)];
            }

            while (matrix_vector_products < max_iterations) {
                const double denominator = matvec_and_dot(
                    direction.data(),
                    operator_direction.data(),
                    reduction_scratch
                );
                ++matrix_vector_products;

                if (denominator <= 0.0 || !std::isfinite(denominator)) {
                    breakdown = true;
                    break;
                }

                const double alpha = residual_squared / denominator;
                const double new_residual_squared =
                    update_solution_residual_and_dot(
                        solution.data(),
                        residual.data(),
                        direction.data(),
                        operator_direction.data(),
                        alpha,
                        count,
                        reduction_scratch
                    );
                ++iterations;
                residual_norm = std::sqrt(new_residual_squared);
                if (residual_norm <= tolerance) {
                    converged = true;
                    break;
                }

                // PARSEC's SPARSKIT CG stores fpar(5)=||r_new|| and reuses
                // fpar(5)**2 here; it does not perform a second residual scan.
                const double beta = new_residual_squared / residual_squared;
#pragma omp parallel for schedule(static) if(size_ >= 4096) \
    num_threads(grid_vector_worker_count(count))
                for (std::int64_t index = 0; index < size_; ++index) {
                    const auto position = static_cast<std::size_t>(index);
                    direction[position] = residual[position]
                        + beta * direction[position];
                }
                residual_squared = new_residual_squared;
            }

            // As in the Python reference, this explicit check is performed
            // after convergence, budget exhaustion, or breakdown and counts
            // as one more matrix-vector product.
            matvec(solution.data(), operator_direction.data());
            ++matrix_vector_products;
            residual_norm = std::sqrt(residual_and_dot(
                rhs.data(),
                operator_direction.data(),
                residual.data(),
                count,
                reduction_scratch
            ));
        }
    }

    py::array_t<double> solution_array(static_cast<py::ssize_t>(size_));
    std::memcpy(
        solution_array.mutable_data(),
        solution.data(),
        count * sizeof(double)
    );
    py::dict result;
    result["solution"] = std::move(solution_array);
    result["converged"] = converged;
    result["iterations"] = iterations;
    result["matrix_vector_products"] = matrix_vector_products;
    result["residual_norm"] = residual_norm;
    result["initial_residual_norm"] = initial_norm;
    result["tolerance"] = tolerance;
    result["breakdown"] = breakdown;
    return result;
}

std::pair<std::int64_t, std::int64_t>
ConjugateGradientSolver::shape() const noexcept {
    return {size_, size_};
}

std::int64_t ConjugateGradientSolver::size() const noexcept {
    return size_;
}

int ConjugateGradientSolver::worker_count() const noexcept {
    return grid_vector_worker_count(static_cast<std::size_t>(size_));
}

std::string ConjugateGradientSolver::storage_mode() const {
    return compact_storage_ ? "int32_columns_uint8_coefficient_palette"
                            : "float64_int64_csr";
}

std::size_t
ConjugateGradientSolver::coefficient_palette_size() const noexcept {
    return coefficient_palette_.size();
}

}  // namespace parsec_accelerated_native
