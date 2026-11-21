"""Real arithmetic for the k-points whose Bloch Hamiltonian is real (``PARSEC_KPOINT_REAL_GAMMA``).

The Bloch Hamiltonian of the u gauge, ``H_k = -nabla^2 - 2ik.grad + |k|^2 + V + V_NL,k``, is complex Hermitian at a
general k-point; at ``k = 0`` (the Gamma point of every Gamma-centred mesh) the gradient term and ``|k|^2`` vanish and
the Bloch projectors carry no phase, so ``H_0`` is the real symmetric Hamiltonian of the Gamma point: the eigenvectors
can be taken real, and the eigensolver of the k-points (the device solver of one process,
:func:`..Eigensolvers.chebff_spinor.solve_spinor_eigval`, and the pools of MPI ranks,
:func:`..Eigensolvers.chebff_columns.solve_column_eigval`) runs on float64 blocks: half the bytes of the basis and of
every filter step, a quarter of the flops of its dense products.  The sectors of the little group of ``k = 0``
(:mod:`.kpoint_sectors`) have real bases (characters ``+-1``), so their reduced Hamiltonians ``U^T H_0 U`` are real as
well and are solved the same way.  The spectrum is that of the complex solve: the eigenvalues agree to the tolerance
of the eigensolver (the start vectors differ: real ones in place of complex ones).

``PARSEC_KPOINT_REAL_GAMMA``: ``auto`` (the default) solves the Gamma point of a k-point SCF in real arithmetic where
its operator is real (a scalar Hamiltonian without the exchange of a hybrid, LDA+U or a meta-GGA's orbital term, on a
whole grid or a sector of it, not the row-distributed solver of the spin-orbit and non-collinear drivers) and the
basis of a k-point (grid points times states) reaches the floor of the k-point sectors, ``2^24``
(:func:`real_gamma_applies`: below it the first compile of the real kernels costs more than Gamma's share saves);
``off`` keeps the complex solve of before, bit for bit, and so does ``auto`` below the floor.  Only the device code of
the k-point SCF takes it (CuPy, or the NumPy double of the tests); the band structures (also the band paths in sectors
and the psi gauge) and the CPU routes keep their complex solves.

A time-reversal invariant point other than Gamma (``k = G/2``) has a real Hamiltonian in the psi gauge
``psi = e^{ik.r} u`` (Bloch phases ``+-1`` across the faces of the cell) where its kinetic term is the Bloch form, the
discretization of the supercell (``PARSEC_KPOINT_KINETIC=bloch``, ``auto`` where it applies, or ``trim``: that form at
these points only).  The gradient form of PARSEC 1.5 (the default) is not real there: it would need an operation that
maps ``k`` onto ``-k`` exactly on the grid.  ``PARSEC_KPOINT_TRIM_REAL`` (:func:`trim_real_setting`, ``auto`` by
default) takes the real structure where the Bloch form is used: a k-point solved whole runs in the psi gauge in real
arithmetic (:class:`PsiGauge`, from a basis of ``2^22`` entries, :func:`psi_gauge_for`; on the pools the twisted real
stencil kernel), a k-point in sectors solves one sector of each conjugate pair (:func:`trim_partners`): the
antiunitary ``e^{-iG.r} conj`` (complex conjugation in the psi gauge) carries a sector into its partner, whose levels
and densities are those of the sector solved.  Both give the
energies of the u gauge to round-off and the tolerance of the eigensolver; ``off`` keeps the u gauge and every
sector, bit for bit.  The Bloch form itself changes the discretization (``PARSEC_KPOINT_KINETIC``, ``gradient`` by
default).
"""

from __future__ import annotations

import os

import numpy as np

GAMMA_SWITCH = "PARSEC_KPOINT_REAL_GAMMA"
TRIM_SWITCH = "PARSEC_KPOINT_TRIM_REAL"
# The smallest basis of a k-point (grid points times states) whose Gamma point is solved in real arithmetic: the
# number of the floor of the k-point sectors (kpoint_sectors._MINIMUM_ENTRIES), a constant of its own (a test that
# splits a small cell into sectors keeps the complex solve).  Below it the first compile of the real kernels (a run
# with an empty kernel cache) costs more than Gamma's share of the mesh saves.
_MINIMUM_ENTRIES = 1 << 24
# The same for the psi gauge of the time-reversal invariant points solved whole (PsiGauge), which take most of a
# small mesh (seven of the eight points of 2 x 2 x 2): with empty kernel caches the 8-atom silicon cube (8,000 points,
# 24 states) was 7 % slower in it, 64 silicon atoms (32,768 and 85,184 points, 144 states) and 64 carbon atoms (46,656
# points, 144 states) 1-1.5 % faster.
_PSI_MINIMUM_ENTRIES = 1 << 22


def real_gamma_setting() -> str:
    """``PARSEC_KPOINT_REAL_GAMMA``: ``auto`` (default) or ``off`` (see the module text)."""

    value = os.environ.get(GAMMA_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{GAMMA_SWITCH} must be auto or off, not {value!r}")
    return value


def real_gamma_applies(job, grid_points) -> bool:
    """Whether the solvers of the k-point job ``job`` take its Gamma point in real arithmetic: ``auto``
    (:func:`real_gamma_setting`), an SCF job (not the band k-points of a Gamma cell, whose geometry carries their
    invariance: one first solve that leaves its buffer states unconverged, whose empty levels would follow the start
    vectors), and a basis of a k-point (``grid_points`` rows of the whole grid times the states of the job) of at least
    ``2^24`` (:data:`_MINIMUM_ENTRIES`, the number of the floor of the k-point sectors).  Below it the first compile of
    the real kernels (a run with an empty kernel cache) costs more than Gamma's share of the mesh saves (the 2-atom
    cells of the examples on one A100: 4-10 % slower; 64 silicon atoms at 85,184 points, 144 states, one k-point per
    GPU: no gain), so the complex solve of before is kept there, bit for bit.  Every rank takes the same answer: the
    job and the environment are those of every rank."""

    if real_gamma_setting() != "auto" or grid_points is None:
        return False
    geometry = getattr(job, "geometry", None)
    if isinstance(geometry, dict) and geometry.get("invariance") is not None:
        return False
    return int(grid_points) * int(job.number_of_states) >= int(_MINIMUM_ENTRIES)


def trim_real_setting() -> str:
    """``PARSEC_KPOINT_TRIM_REAL``: ``auto`` (default) or ``off`` (:class:`PsiGauge`, :func:`trim_partners`).

    ``auto``: a time-reversal invariant k-point other than Gamma (``k = G/2``) that takes the Bloch form of the kinetic
    term (``PARSEC_KPOINT_KINETIC=bloch``, ``auto`` where it applies, or ``trim``) is solved whole in the psi gauge in
    real arithmetic (the device solvers of the k-point SCF, a plain Hamiltonian without the exchange of a hybrid or
    LDA+U, a basis of at least ``2^22`` grid points times states, not the band paths), and where it splits into
    sectors in an unpolarized k-point SCF without the exchange of a hybrid one sector of each conjugate pair is solved;
    ``off``: the u gauge and every sector, as before.  The default kinetic form (``gradient``) is not affected.
    """

    value = os.environ.get(TRIM_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{TRIM_SWITCH} must be auto or off, not {value!r}")
    return value


def check_switches() -> None:
    """Read the switches of this module (and the kinetic form of the k-points, which ``trim`` extends), so that a
    value they do not know stops the run at its preparation."""

    from parsec_python.Laplacian.finite_difference import kpoint_kinetic_setting

    real_gamma_setting()
    trim_real_setting()
    kpoint_kinetic_setting()


def trim_partners(cell, elements, generators, k_point) -> np.ndarray | None:
    """The partner of every sector of a time-reversal invariant k-point in the Bloch form, or ``None``.

    At ``k = G/2`` the Bloch form of the u gauge satisfies ``H_k^* = H_{-k} = e^{iG.r} H_k e^{-iG.r}`` exactly (the
    Bloch Laplacian ``e^{-ik.r} L e^{ik.r}`` of the finite-difference ``L``, the Bloch projectors ``e^{-ik.r} B``
    with ``B`` real there; the potential is real), so the antiunitary ``A u = e^{-iG.r} conj(u)`` commutes with
    ``H_k``: in the psi gauge ``e^{ik.r} u`` it is complex conjugation, and the Hamiltonian is real.  ``A`` carries
    the sector of character ``chi`` of the commuting involutions ``H_k`` (:mod:`.kpoint_sectors`; each ``h: r -> R r
    + t`` with ``R k = k``) into that of ``chi lambda``, ``lambda(h) = e^{-iG.t}``, a character of ``H_k`` (``-1`` on
    the half-cell translations along the axes where ``k`` sits at ``pi/L``): with a reduced Hamiltonian
    ``H_{chi lambda} = Phi conj(H_chi) Phi^*`` (``Phi`` diagonal phases ``e^{-iG.p}`` of the representative points),
    the same spectrum, orbitals ``A u`` of the same density.  ``elements`` and ``generators`` are the indices in
    ``cell`` (:class:`.kpoint_sectors.CellGroup`) of ``H_k`` and its generators (element ``p`` the product of the
    generators of the bits of ``p``); the sector of the parities ``q`` (``itertools.product`` order, the first
    generator the highest bit) has the partner ``q XOR m``.  ``None`` where ``k`` is not such a point, ``lambda`` is
    trivial (every sector its own partner) or not ``+-1``.
    """

    from parsec_python.Laplacian.finite_difference import is_trim_point

    grid = cell.grid
    if not is_trim_point(grid, k_point) or not generators:
        return None
    reciprocal = 2.0 * np.asarray(k_point, dtype=np.float64).reshape(3)
    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    rank = len(generators)
    mask = 0
    for position, index in enumerate(generators):
        image = cell.row_map(int(index), rows=np.zeros(1, dtype=np.int64))
        if image is None:
            return None
        matrix = np.asarray(cell.matrices[int(index)], dtype=np.float64)
        translation = coordinates[int(np.asarray(image).reshape(-1)[0])] - matrix @ coordinates[0]
        value = np.exp(-1j * float(reciprocal @ translation))
        if abs(value - 1.0) < 1.0e-8:
            continue
        if abs(value + 1.0) >= 1.0e-8:
            return None
        mask |= 1 << (rank - 1 - position)
    if mask == 0:
        return None
    return np.arange(1 << rank, dtype=np.int64) ^ mask


def is_gamma(k_point) -> bool:
    """Whether ``k_point`` (Cartesian, any shape of three numbers; ``None`` an isolated system) is exactly zero."""

    if k_point is None:
        return False
    values = np.asarray(k_point, dtype=np.float64).reshape(-1)
    return values.size == 3 and bool(np.all(values == 0.0))


def _is_complex(array) -> bool:
    return np.dtype(array.dtype).kind == "c"


def matched_state(state, real: bool, gauge: str | None = None):
    """The Ritz state ``state`` of a task for a solve whose arithmetic is real (``real``) or complex, in ``gauge``
    (``None`` the u gauge, ``"psi"`` that of :class:`PsiGauge`).

    A real subspace goes into a complex solve as it is, its vectors cast to complex (the same space: the exchange loop
    of a hybrid after its semilocal start, whose operator at Gamma is complex); a complex subspace into a real solve
    is dropped (``None``: the solve starts afresh), since its span is not that of real vectors, and so is a subspace
    of another gauge.  A state of the host eigensolver (no ``vectors``) and a state that matches come back as they
    are.
    """

    if state is None:
        return None
    vectors = getattr(state, "vectors", None)
    if vectors is None:
        return state
    if getattr(state, "gauge", None) != gauge:
        return None
    if real:
        return None if _is_complex(vectors) else state
    if _is_complex(vectors):
        return state
    from dataclasses import replace

    order = "F" if vectors.flags.f_contiguous and not vectors.flags.c_contiguous else "C"
    return replace(state, vectors=vectors.astype(np.complex128, order=order))


def as_complex(vectors):
    """``vectors`` as complex128 (host or device, the same memory order): the orbitals a real solve hands to callers
    that take Bloch orbitals."""

    if vectors is None or _is_complex(vectors):
        return vectors
    order = "F" if vectors.flags.f_contiguous and not vectors.flags.c_contiguous else "C"
    return vectors.astype(np.complex128, order=order)


class PsiGauge:
    """The psi gauge of a time-reversal invariant point ``k = G/2`` in the Bloch form, where the Hamiltonian is real.

    ``psi = D u`` with ``D = diag(e^{ik.r})`` on the grid points ``r``: ``H_psi = D H_u D^*`` is the twisted
    finite-difference Laplacian of the cell (every neighbour across a face along an axis where ``k = pi/L`` taken
    with ``-1``, :func:`parsec_python.Laplacian.finite_difference.build_bloch_negative_laplacian` with
    ``gauge="psi"``), the real Bloch projectors ``D B_u`` (``sum_R e^{ik.R} beta(r - tau - R)``, signs ``+-1``) and
    the potential.  The density ``|psi|^2 = |u|^2`` needs no conversion; the orbitals handed out and the forces take
    ``u = D^* psi`` (:meth:`to_u`).  Built once per k-point (its Laplacian and projectors kept for the objects of the
    u gauge they came from).
    """

    def __init__(self, grid, k_point) -> None:
        self.grid = grid
        self.k_point = np.asarray(k_point, dtype=np.float64).reshape(3)
        self.phases = np.exp(1j * (np.asarray(grid.coordinates, dtype=np.float64) @ self.k_point))
        self._laplacian = None
        self._projectors: tuple | None = None  # (u-gauge operator, psi-gauge operator)
        self._device: dict[int, object] = {}

    def laplacian(self):
        if self._laplacian is None:
            import scipy.sparse as sp

            from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian

            matrix = sp.csr_matrix(build_bloch_negative_laplacian(self.grid, self.k_point, "psi"))
            if np.iscomplexobj(matrix.data):
                if matrix.nnz and float(np.max(np.abs(matrix.data.imag))) > 1.0e-12 * float(np.max(np.abs(matrix.data))):
                    raise RuntimeError("the Bloch Laplacian of a time-reversal invariant point is not real in the psi gauge")
                matrix = sp.csr_matrix((matrix.data.real.copy(), matrix.indices, matrix.indptr), shape=matrix.shape)
            self._laplacian = matrix
        return self._laplacian

    def projectors(self, nonlocal_operator):
        """The real projectors ``D B_u`` of the u-gauge projector operator ``nonlocal_operator``."""
        cached = self._projectors
        if cached is not None and cached[0] is nonlocal_operator:
            return cached[1]
        import scipy.sparse as sp
        from dataclasses import replace

        matrix = sp.csc_matrix(nonlocal_operator.projectors)
        if matrix.shape[1]:
            rows = matrix.indices
            values = self.phases[rows] * matrix.data
            scale = float(np.max(np.abs(values))) if values.size else 0.0
            if values.size and float(np.max(np.abs(values.imag))) > 1.0e-10 * max(scale, 1.0e-300):
                raise RuntimeError("the Bloch projectors of a time-reversal invariant point are not real in the psi gauge")
            matrix = sp.csc_matrix((values.real.copy(), rows, matrix.indptr), shape=matrix.shape)
        else:
            matrix = sp.csc_matrix(matrix.shape, dtype=np.float64)
        operator = replace(nonlocal_operator, projectors=matrix)
        self._projectors = (nonlocal_operator, operator)
        return operator

    def hamiltonian(self, hamiltonian):
        """The psi-gauge Hamiltonian of the u-gauge Bloch Hamiltonian ``hamiltonian`` (its potential and projectors)."""
        return PsiGaugeHamiltonian(
            hamiltonian.negative_laplacian,
            hamiltonian.gradient,
            hamiltonian.effective_potential,
            self.projectors(hamiltonian.nonlocal_operator),
            hamiltonian.k_point,
            bloch_laplacian=self.laplacian(),
        )

    def to_u(self, vectors, xp=np):
        """``u = D^* psi`` (complex) of a block of psi-gauge orbitals, where the block is (host or device)."""
        if xp is np or isinstance(vectors, np.ndarray):
            return np.conj(self.phases)[:, None] * np.asarray(vectors)
        device = int(vectors.device.id)
        phases = self._device.get(device)
        if phases is None:
            with xp.cuda.Device(device):
                phases = self._device[device] = xp.asarray(np.conj(self.phases))
        return phases[:, None] * vectors


def _psi_gauge_class():
    from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian

    class _PsiGaugeHamiltonian(KPointKohnShamHamiltonian):
        """A Bloch Hamiltonian in the psi gauge of a time-reversal invariant point: real (:class:`PsiGauge`)."""

        psi_gauge = True

    return _PsiGaugeHamiltonian


def PsiGaugeHamiltonian(*arguments, **keywords):  # noqa: N802 - a class made on first use (no import cycle)
    global _PSI_CLASS
    if _PSI_CLASS is None:
        _PSI_CLASS = _psi_gauge_class()
    return _PSI_CLASS(*arguments, **keywords)


_PSI_CLASS = None


def psi_gauge_for(job, task: int, hamiltonian, cache: dict):
    """The :class:`PsiGauge` of task ``task`` of a k-point job where its Hamiltonian is a plain u-gauge Bloch
    Hamiltonian (``bloch_laplacian``, no exchange, LDA+U or meta-GGA) at a time-reversal invariant point other than
    Gamma, ``PARSEC_KPOINT_TRIM_REAL=auto``, the job gives its grid (``geometry``), is an SCF job (not the band
    k-points of a Gamma cell, whose geometry carries their invariance: their one first solve keeps the complex start
    vectors of before) and its basis of a k-point (grid points times states) reaches :data:`_PSI_MINIMUM_ENTRIES`
    (``2^22``; below it the first compile of the real kernels costs more than the gauge saves); ``None`` elsewhere.
    ``cache`` keeps the gauge of each task (the solver's)."""

    if task in cache:
        return cache[task]
    found = None
    geometry = getattr(job, "geometry", None)
    grid = geometry.get("grid") if isinstance(geometry, dict) else None
    if (
        grid is not None
        and geometry.get("invariance") is None
        and int(grid.size) * int(job.number_of_states) >= int(_PSI_MINIMUM_ENTRIES)
        and trim_real_setting() == "auto"
        and getattr(hamiltonian, "bloch_laplacian", None) is not None
        and getattr(hamiltonian, "base", None) is None
        and getattr(hamiltonian, "kinetic_potential", None) is None
        and not getattr(hamiltonian, "psi_gauge", False)
        and not getattr(hamiltonian, "sector_stencil", False)
        and not getattr(getattr(hamiltonian, "bloch_laplacian", None), "sector_stencil", False)
        and getattr(hamiltonian, "scalar_hamiltonian", None) is None
    ):
        from parsec_python.Laplacian.finite_difference import is_trim_point

        if is_trim_point(grid, hamiltonian.k_point) and int(hamiltonian.negative_laplacian.shape[0]) == int(grid.size):
            found = PsiGauge(grid, hamiltonian.k_point)
    cache[task] = found
    return found


__all__ = [
    "PsiGauge",
    "psi_gauge_for",
    "GAMMA_SWITCH",
    "TRIM_SWITCH",
    "as_complex",
    "check_switches",
    "is_gamma",
    "matched_state",
    "real_gamma_applies",
    "real_gamma_setting",
    "trim_partners",
    "trim_real_setting",
]
