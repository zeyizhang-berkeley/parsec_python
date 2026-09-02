#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace parsec_accelerated_native {

/**
 * Choose a useful OpenMP team for repeated grid-vector kernels.
 *
 * Two 4096-point deterministic reduction blocks per worker provide enough
 * work to amortize team entry and barrier costs.  The result is bounded by
 * the user's OMP_NUM_THREADS setting (or the extension default), so large
 * grids still use every configured worker and explicit user caps are always
 * respected.
 */
inline int grid_vector_worker_count(std::size_t count) noexcept {
#ifdef _OPENMP
    constexpr std::size_t minimum_items_per_worker = 8192;
    const std::size_t useful_workers = std::max<std::size_t>(
        1,
        (count + minimum_items_per_worker - 1) / minimum_items_per_worker
    );
    const std::size_t bounded = std::min<std::size_t>(
        useful_workers,
        static_cast<std::size_t>(std::numeric_limits<int>::max())
    );
    return std::max(1, std::min(omp_get_max_threads(), static_cast<int>(bounded)));
#else
    static_cast<void>(count);
    return 1;
#endif
}

}  // namespace parsec_accelerated_native
