#pragma once

#include "finite_difference.h"

#include <cstddef>
#include <vector>

namespace parsec_accelerated_native {

class RadialGridEvaluator {
public:
    explicit RadialGridEvaluator(const FloatArray& coordinates);

    FloatArray local_potential(
        const FloatArray& atom_position,
        const FloatArray& radii,
        const FloatArray& values,
        double ionic_charge,
        const FloatArray& spline_knots,
        const FloatArray& spline_values,
        const FloatArray& spline_second_derivatives
    ) const;

    FloatArray density(
        const FloatArray& atom_position,
        const FloatArray& radii,
        const FloatArray& values,
        const FloatArray& spline_knots,
        const FloatArray& spline_values,
        const FloatArray& spline_second_derivatives
    ) const;

    py::dict projector_channel(
        const FloatArray& atom_position,
        const FloatArray& radii,
        const FloatArray& radial_projector,
        double support_radius,
        int angular_momentum,
        double square_root_volume,
        const FloatArray& spline_knots,
        const FloatArray& spline_values,
        const FloatArray& spline_second_derivatives
    ) const;

    std::size_t size() const noexcept;

private:
    std::vector<double> coordinates_;
};

}  // namespace parsec_accelerated_native
