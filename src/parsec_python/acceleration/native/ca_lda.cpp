#include "ca_lda.h"
#include "openmp_workload.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace parsec_accelerated_native {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::size_t kReductionBlockSize = 4096;

}  // namespace

CALDAEvaluator::CALDAEvaluator(
    const FloatArray& core_density,
    double volume_element
) :
    core_density_(
        core_density.data(),
        core_density.data() + static_cast<std::size_t>(core_density.size())
    ),
    volume_element_(volume_element) {
    if (core_density.ndim() != 1) {
        throw std::invalid_argument("core_density must be one-dimensional");
    }
    if (!(volume_element_ > 0.0) || !std::isfinite(volume_element_)) {
        throw std::invalid_argument("volume_element must be finite and positive");
    }
    if (!std::all_of(
            core_density_.begin(),
            core_density_.end(),
            [](double value) { return std::isfinite(value); }
        )) {
        throw std::invalid_argument("core_density must contain finite values");
    }
    integration_weights_.assign(core_density_.size(), 1.0);
}

CALDAEvaluator::CALDAEvaluator(
    const FloatArray& core_density,
    double volume_element,
    const IndexArray& integration_weights
) : CALDAEvaluator(core_density, volume_element) {
    if (
        integration_weights.ndim() != 1 ||
        static_cast<std::size_t>(integration_weights.shape(0)) !=
            core_density_.size()
    ) {
        throw std::invalid_argument(
            "integration_weights must match the cached core density"
        );
    }
    for (std::size_t index = 0; index < core_density_.size(); ++index) {
        const std::int64_t weight = integration_weights.data()[index];
        if (weight <= 0) {
            throw std::invalid_argument("integration_weights must be positive");
        }
        integration_weights_[index] = static_cast<double>(weight);
    }
}

py::dict CALDAEvaluator::evaluate(const FloatArray& valence_density) const {
    if (
        valence_density.ndim() != 1 ||
        static_cast<std::size_t>(valence_density.shape(0)) !=
            core_density_.size()
    ) {
        throw std::invalid_argument(
            "valence_density does not match the cached core density"
        );
    }
    const std::size_t count = core_density_.size();
    const double* valence = valence_density.data();
    for (std::size_t index = 0; index < count; ++index) {
        const double density = valence[index] + core_density_[index];
        if (!std::isfinite(density)) {
            throw std::invalid_argument("CA-LDA density must contain finite values");
        }
        if (density < -1.0e-14) {
            throw std::invalid_argument("CA-LDA requires a nonnegative density");
        }
    }

    py::array_t<double> potential_array(static_cast<py::ssize_t>(count));
    py::array_t<double> epsilon_array(static_cast<py::ssize_t>(count));
    py::array_t<double> energy_density_array(static_cast<py::ssize_t>(count));
    double* potential = potential_array.mutable_data();
    double* epsilon = epsilon_array.mutable_data();
    double* energy_density = energy_density_array.mutable_data();

    const std::size_t reduction_blocks =
        (count + kReductionBlockSize - 1) / kReductionBlockSize;
    std::vector<double> partial_energy(reduction_blocks, 0.0);

    {
        py::gil_scoped_release release;
        const double a0 = std::cbrt(4.0 / (9.0 * kPi));
#pragma omp parallel for schedule(static) if(count >= kReductionBlockSize) \
    num_threads(grid_vector_worker_count(count))
        for (std::int64_t block = 0;
             block < static_cast<std::int64_t>(reduction_blocks); ++block) {
            const std::size_t start =
                static_cast<std::size_t>(block) * kReductionBlockSize;
            const std::size_t stop =
                std::min(count, start + kReductionBlockSize);
            double block_energy = 0.0;
            for (std::size_t index = start; index < stop; ++index) {
                const double density = valence[index] + core_density_[index];
                if (!(density > 0.0)) {
                    potential[index] = 0.0;
                    epsilon[index] = 0.0;
                    energy_density[index] = 0.0;
                    continue;
                }

                const double rs = std::cbrt(0.75 / (kPi * density));
                const double exchange_potential = -2.0 / (kPi * a0 * rs);
                const double exchange_epsilon = 0.75 * exchange_potential;
                double correlation_epsilon = 0.0;
                double correlation_potential = 0.0;
                if (rs >= 1.0) {
                    const double square_root = std::sqrt(rs);
                    constexpr double g = -0.2846;
                    constexpr double b1 = 1.0529;
                    constexpr double b2 = 0.3334;
                    correlation_epsilon =
                        g / (1.0 + b1 * square_root + b2 * rs);
                    correlation_potential =
                        (correlation_epsilon * correlation_epsilon / g) *
                        (
                            1.0 + (7.0 / 6.0) * b1 * square_root +
                            (4.0 / 3.0) * b2 * rs
                        );
                } else {
                    const double logarithm = std::log(rs);
                    constexpr double c1 = 0.0622;
                    constexpr double c2 = 0.096;
                    constexpr double c3 = 0.004;
                    constexpr double c4 = 0.0232;
                    constexpr double c5 = 0.0192;
                    correlation_epsilon =
                        c1 * logarithm - c2 +
                        (c3 * logarithm - c4) * rs;
                    correlation_potential =
                        correlation_epsilon -
                        (c1 + (c3 * logarithm - c5) * rs) / 3.0;
                }

                epsilon[index] = exchange_epsilon + correlation_epsilon;
                potential[index] =
                    exchange_potential + correlation_potential;
                energy_density[index] = density * epsilon[index];
                block_energy +=
                    integration_weights_[index] * energy_density[index];
            }
            partial_energy[static_cast<std::size_t>(block)] = block_energy;
        }
    }

    double total_energy = 0.0;
    for (const double value : partial_energy) {
        total_energy += value;
    }
    total_energy *= volume_element_;

    py::dict result;
    result["potential"] = std::move(potential_array);
    result["energy_per_electron"] = std::move(epsilon_array);
    result["energy_density"] = std::move(energy_density_array);
    result["total_energy"] = total_energy;
    return result;
}

std::size_t CALDAEvaluator::size() const noexcept {
    return core_density_.size();
}

}  // namespace parsec_accelerated_native
