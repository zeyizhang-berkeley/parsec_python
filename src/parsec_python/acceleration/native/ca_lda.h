#pragma once

#include "finite_difference.h"

#include <cstddef>
#include <vector>

namespace parsec_accelerated_native {

class CALDAEvaluator {
public:
    CALDAEvaluator(const FloatArray& core_density, double volume_element);
    CALDAEvaluator(
        const FloatArray& core_density,
        double volume_element,
        const IndexArray& integration_weights
    );

    py::dict evaluate(const FloatArray& valence_density) const;

    std::size_t size() const noexcept;

private:
    std::vector<double> core_density_;
    std::vector<double> integration_weights_;
    double volume_element_ = 0.0;
};

}  // namespace parsec_accelerated_native
